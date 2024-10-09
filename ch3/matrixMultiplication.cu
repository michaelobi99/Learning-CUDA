#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>


__global__ void MatrixMulKernel(float* M, float* N, float* P, int width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < width) && (col < width)) {
		float Pvalue = 0;
		for (int k = 0; k < width; ++k) {
			Pvalue += M[row * width + k] * N[k * width + col];
		}
		P[row * width + col] = Pvalue;
	}
}

int main() {
	int row, col;
	printf("Dimensions of matrix:\n");
	printf("row: ");
	std::cin >> row;
	printf("col: ");
	std::cin >> col;

	float* M = (float*)malloc(row * col * sizeof(float));
	float* N = (float*)malloc(row * col * sizeof(float));
	float* P = (float*)malloc(col * row * sizeof(float));
	int i = 0;
	printf("En`ter values of MAt1 (row major)\n");
	for (i = 0; i < (row * col); ++i) {
		std::cin >> M[i];
	}

	printf("Enter values of MAt2 (row major)\n");
	for (i = 0; i < (row * col); ++i) {
		std::cin >> N[i];
	}

	float* M_d, * N_d, * P_d;

	cudaMalloc(&M_d, row * col * sizeof(float));
	cudaMalloc(&N_d, row * col * sizeof(float));
	cudaMalloc(&P_d, col * row * sizeof(float));

	cudaMemcpy(M_d, M, row * col * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(N_d, N, row * col * sizeof(float), cudaMemcpyHostToDevice);

	dim3 gridSize(1, 1, 1);
	dim3 blockSize(2, 2, 1);

	MatrixMulKernel << <gridSize, blockSize >> > (M_d, N_d, P_d, row);

	cudaMemcpy(P, P_d, row * col * sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < (row * col); ++i) {
		printf("%f ", P[i]);
	}

	free(M);
	free(N);
	free(P);

	cudaFree(M_d);
	cudaFree(N_d);
	cudaFree(P_d);
}