#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

#pragma pack(push, 1)
struct ImageHeader {
    uint16_t fileType;
    uint32_t fileSize;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offsetData;
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


#define BLUR_SIZE 3

__global__ void blurGrayscaleKernel(const unsigned char* in, unsigned char* out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixelValue = 0;
        int pixelCount = 0;
        //Get average of surrounding BLUR_SIZE X BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int currentRow = row + blurRow;
                int currentCol = col + blurCol;
                //verify that we have a valid image pixel
                if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {
                    pixelValue += in[currentRow * width + currentCol];
                    pixelCount += 1;
                }
            }
        }
        out[row * width + col] = (unsigned char)(pixelValue / pixelCount);
    }
}

__global__ void blurRGBkernel(const unsigned char* in, unsigned char* out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixelCount = 0;
        int pixelValue = 0;
        //Get average of surrounding RED, GREEB, or BLUE Pixel.
        //The loop goes through each column in skips of three to average out corresponding channels.
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -(BLUR_SIZE * 3); blurCol < (BLUR_SIZE * 3) + 1; blurCol += 3) {
                int currentRow = row + blurRow;
                int currentCol = col + blurCol;
                //verify that we have a valid image pixel
                if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {
                    pixelValue += in[currentRow * width + currentCol];
                    pixelCount += 1;
                }
            }
        }
        out[row * width + col] = (unsigned char)(pixelValue / pixelCount);
    }
}

int main() {
    const char* imageFile = R"(C:\Users\HP\Pictures\lena-flipped.bmp)";
    const char* outImageFile = R"(C:\Users\HP\Pictures\blurredlena-flipped1.bmp)";

    /*const char* imageFile = R"(C:\Users\HP\Pictures\graylena-flipped.bmp)";
    const char* outImageFile = R"(C:\Users\HP\Pictures\blurredlena-flipped.bmp)";*/

    std::ifstream image(imageFile, std::ios::binary);
    if (!image) {
        std::cerr << "Error opening input file" << std::endl;
        return 1;
    }

    std::ofstream outImage(outImageFile, std::ios::binary);
    if (!outImage) {
        std::cerr << "Error opening output file" << std::endl;
        return 1;
    }

    ImageHeader bmpHeader;

    // Read input image header
    if (!image.read(reinterpret_cast<char*>(&bmpHeader), sizeof(ImageHeader))) {
        std::cerr << "Error reading headers" << std::endl;
        return 1;
    }

    // Check if it's an 8-bit or  24-bit BMP
    if ((bmpHeader.bitsPerPixel != 8 && bmpHeader.bitsPerPixel != 24) || bmpHeader.compression != 0) {
        std::cerr << "Unsupported BMP format. Only uncompressed 8-bit or 24-bit BMPs are supported." << std::endl;
        return 1;
    }

    // Write header
    outImage.write(reinterpret_cast<char*>(&bmpHeader), sizeof(ImageHeader));

    if (bmpHeader.bitsPerPixel == 8) {
        //read and write color palette
        uint8_t gray[1024] = { 0 };
        image.read(reinterpret_cast<char*>(gray), 1024);
        outImage.write(reinterpret_cast<char*>(gray), 1024);
    }


    // Calculate paddings
    int imagePadding = bmpHeader.bitsPerPixel == 8 ? (4 - (bmpHeader.width) % 4) % 4 : (4 - (bmpHeader.width * 3) % 4) % 4;

    // Process pixels
    //int width = bmpHeader.width;
    int width = bmpHeader.bitsPerPixel == 8 ? bmpHeader.width : bmpHeader.width * 3;
    int height = std::abs(bmpHeader.height);
    size_t imageSize = (width + imagePadding) * height;

    std::vector<unsigned char> Pin(imageSize);
    std::vector<unsigned char> Pout(imageSize);

    // Read image data
    image.read(reinterpret_cast<char*>(Pin.data()), imageSize);

    unsigned char* Pin_d, * Pout_d;

    cudaMalloc(&Pin_d, imageSize);
    cudaMalloc(&Pout_d, imageSize);

    cudaMemcpy(Pin_d, Pin.data(), imageSize, cudaMemcpyHostToDevice);

    if (bmpHeader.bitsPerPixel == 8) {
        dim3 blockSize(16, 16, 1);
        dim3 gridSize(ceil(height / 16), ceil(width / 16), 1);
        blurGrayscaleKernel << <gridSize, blockSize >> > (Pin_d, Pout_d, width, height);
    }
    else {
        dim3 blockSize(32, 32, 1);
        dim3 gridSize(ceil(height / 8), ceil(width / 8), 1);
        blurRGBkernel << <gridSize, blockSize >> > (Pin_d, Pout_d, width, height);
    }

    cudaMemcpy(Pout.data(), Pout_d, imageSize, cudaMemcpyDeviceToHost);

    // Write blurred image pixels
    for (int y = 0; y < height; ++y) {
        outImage.write(reinterpret_cast<char*>(Pout.data() + y * width), width);
        outImage.write(std::vector<char>(imagePadding, 0).data(), imagePadding);
    }

    image.close();
    outImage.close();

    cudaFree(Pin_d);
    cudaFree(Pout_d);

    std::cout << "Image blurring successful." << std::endl;
    return 0;
}