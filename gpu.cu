#include "gpu.cuh"

#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_header.cuh"

#define BLOCK_SIZE 16

__device__ int calcNeighbor(const int* in_data, const int row, const int col, const unsigned int width, const unsigned int height) {
    if (row < 0 || row >= height || col < 0 || col >= width) {
        return 0;
    }

    return in_data[row + col * width];
}

__global__
void cuda_calc_neighbors_kernel(const int* in_data, int* out_data, const unsigned int width, const unsigned int height) {
    // Calculate the global index
    const unsigned int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const unsigned int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Check if the index is out of bounds
    if (row >= height || col >= width) {
        return;
    }

    const unsigned int idx = row + col * width;

    // Calculate the number of neighbors
    int neighbors = 0;
    neighbors += calcNeighbor(in_data, row - 1, col - 1, width, height);
    neighbors += calcNeighbor(in_data, row - 1, col, width, height);
    neighbors += calcNeighbor(in_data, row - 1, col + 1, width, height);
    neighbors += calcNeighbor(in_data, row, col - 1, width, height);
    neighbors += calcNeighbor(in_data, row, col + 1, width, height);
    neighbors += calcNeighbor(in_data, row + 1, col - 1, width, height);
    neighbors += calcNeighbor(in_data, row + 1, col, width, height);
    neighbors += calcNeighbor(in_data, row + 1, col + 1, width, height);

    out_data[idx] = neighbors;
}

float cuda_call_calc_neighbors_kernel(const int* in_data, int* out_data, const unsigned int width, const unsigned int height) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    float time = 0.0f;

    // Allocate GPU memory
    int* gpu_in_data;
    int* gpu_out_data;
    cudaMalloc((void**)&gpu_in_data, width * height * sizeof(int));
    cudaMalloc((void**)&gpu_out_data, width * height * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(gpu_in_data, in_data, width * height * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    cudaEventRecord(start_gpu);
    cuda_calc_neighbors_kernel<<<gridSize, blockSize>>>(gpu_in_data, gpu_out_data, width, height);
    cudaEventRecord(stop_gpu);

    // Copy data back to CPU
    cudaMemcpy(out_data, gpu_out_data, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate time
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time, start_gpu, stop_gpu);

    // Clean up
    cudaFree(gpu_in_data);
    cudaFree(gpu_out_data);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return time;
}

__global__
void cuda_update_kernel(const int* in_data, const int* neighbors, int* out_data, const unsigned int width, const unsigned int height) {
    // Calculate the global index
    const unsigned int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const unsigned int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Check if the index is out of bounds
    if (row >= height || col >= width) {
        return;
    }

    const unsigned int idx = row + col * width;

    // Update the cell
    const int neighbor = neighbors[idx];
    if (in_data[idx] == 1) {
        if (neighbor < 2 || neighbor > 3) {
            out_data[idx] = 0;
        } else {
            out_data[idx] = 1;
        }
    } else {
        if (neighbor == 3) {
            out_data[idx] = 1;
        } else {
            out_data[idx] = 0;
        }
    }
}

float cuda_call_update_kernel(const int* in_data, int* out_data, const unsigned int width, const unsigned int height) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    float time = 0.0f;

    // Allocate GPU memory
    int* gpu_in_data;
    int* gpu_neighbors;
    int *gpu_out_neighbors;
    int* gpu_out_data;
    cudaMalloc((void**)&gpu_in_data, width * height * sizeof(int));
    cudaMalloc((void**)&gpu_neighbors, width * height * sizeof(int));
    cudaMalloc((void**)&gpu_out_neighbors, width * height * sizeof(int));   
    cudaMalloc((void**)&gpu_out_data, width * height * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(gpu_in_data, in_data, width * height * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernels
    cuda_calc_neighbors_kernel<<<numBlocks, blockSize>>>(gpu_in_data, gpu_neighbors, width, height);
    cuda_update_kernel<<<numBlocks, blockSize>>>(gpu_in_data, gpu_neighbors, gpu_out_data, width, height);

    // Copy data back to CPU
    cudaMemcpy(out_data, gpu_out_data, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(gpu_in_data);
    cudaFree(gpu_neighbors);
    cudaFree(gpu_out_neighbors);
    cudaFree(gpu_out_data);

    // Calculate time
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time, start_gpu, stop_gpu);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return time;
}