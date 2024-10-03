#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType;
    uint32_t fileSize;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offsetData;
};

struct DIBHeader {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
};
#pragma pack(pop)

__global__ void colorToGrayscaleConversion(unsigned char* Pout, const unsigned char* Pin, int width, int height, int pitch) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = row * pitch + col * 3;
        unsigned char R = Pin[rgbOffset];
        unsigned char G = Pin[rgbOffset + 1];
        unsigned char B = Pin[rgbOffset + 2];
        Pout[grayOffset] = static_cast<unsigned char>(0.21f * R + 0.71f * G + 0.07f * B);
    }
}

int main() {
    const char* imageFile = R"(C:\Users\HP\Pictures\lena-flipped.bmp)";
    const char* outImageFile = R"(C:\Users\HP\Pictures\graylena-flipped.bmp)";

    std::ifstream image(imageFile, std::ios::binary);
    if (!image) {
        std::cerr << "Error opening input file" << std::endl;
        return 1;
    }

    BMPHeader bmpHeader;
    DIBHeader dibHeader;

    // Read headers
    if (!image.read(reinterpret_cast<char*>(&bmpHeader), sizeof(BMPHeader)) ||
        !image.read(reinterpret_cast<char*>(&dibHeader), sizeof(DIBHeader))) {
        std::cerr << "Error reading headers" << std::endl;
        return 1;
    }

    // Check if it's a 24-bit BMP
    if (dibHeader.bitsPerPixel != 24 || dibHeader.compression != 0) {
        std::cerr << "Unsupported BMP format. Only uncompressed 24-bit BMPs are supported." << std::endl;
        return 1;
    }

    // Calculate paddings
    int rgbPadding = (4 - (dibHeader.width * 3) % 4) % 4;
    int grayPadding = (4 - dibHeader.width % 4) % 4;

    // Modify headers for grayscale
    dibHeader.bitsPerPixel = 8;
    uint32_t colorTableSize = 256 * 4;
    bmpHeader.offsetData = sizeof(BMPHeader) + sizeof(DIBHeader) + colorTableSize;

    uint64_t newFileSize = bmpHeader.offsetData +
        static_cast<uint64_t>(dibHeader.width + grayPadding) * std::abs(dibHeader.height);

    if (newFileSize > UINT32_MAX) {
        std::cerr << "Resulting file size is too large" << std::endl;
        return 1;
    }

    bmpHeader.fileSize = static_cast<uint32_t>(newFileSize);
    dibHeader.imageSize = static_cast<uint32_t>(newFileSize - bmpHeader.offsetData);

    // Open output file
    std::ofstream outImage(outImageFile, std::ios::binary);
    if (!outImage) {
        std::cerr << "Error opening output file" << std::endl;
        return 1;
    }

    // Write headers
    outImage.write(reinterpret_cast<char*>(&bmpHeader), sizeof(BMPHeader));
    outImage.write(reinterpret_cast<char*>(&dibHeader), sizeof(DIBHeader));

    // Write grayscale palette
    for (int i = 0; i < 256; ++i) {
        uint8_t gray[4] = { static_cast<uint8_t>(i), static_cast<uint8_t>(i), static_cast<uint8_t>(i), 0 };
        outImage.write(reinterpret_cast<char*>(gray), 4);
    }

    // Process pixels
    int width = dibHeader.width;
    int height = std::abs(dibHeader.height);
    int pitch = width * 3 + rgbPadding;
    size_t rgbSize = static_cast<size_t>(pitch) * height;
    size_t graySize = static_cast<size_t>(width) * height;

    std::vector<unsigned char> Pin(rgbSize);
    std::vector<unsigned char> Pout(graySize);

    // Read image data
    image.read(reinterpret_cast<char*>(Pin.data()), rgbSize);

    unsigned char* Pin_d, * Pout_d;

    cudaMalloc(&Pin_d, rgbSize);
    cudaMalloc(&Pout_d, graySize);

    cudaMemcpy(Pin_d, Pin.data(), rgbSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 1);
    dim3 gridSize(ceil(height / 16), ceil(width / 16), 1);

    colorToGrayscaleConversion << <gridSize, blockSize >> > (Pout_d, Pin_d, width, height, pitch);

    cudaMemcpy(Pout.data(), Pout_d, graySize, cudaMemcpyDeviceToHost);

    // Write grayscale image data
    for (int y = 0; y < height; ++y) {
        outImage.write(reinterpret_cast<char*>(Pout.data() + y * width), width);
        outImage.write(std::vector<char>(grayPadding, 0).data(), grayPadding);
    }

    image.close();
    outImage.close();

    cudaFree(Pin_d);
    cudaFree(Pout_d);

    std::cout << "Conversion completed successfully." << std::endl;
    return 0;
}