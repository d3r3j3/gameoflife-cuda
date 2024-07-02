#include "imgui.h"
#include <stdio.h>
#include <random>
#include <iostream>
#include "gpu.cuh"
#include <cuda_runtime.h>

using namespace std;
#include <vector>
#include "objects.hpp"

Grid::Grid(int width, int height) {
    this->width = width;
    this->height = height;
    this->cells = new int[width * height];
    this->cell_size = 10;
    
    // initialize the cells to be dead
    clear();
}

Grid::Grid(int width, int height, int* state) {
    this->width = width;
    this->height = height;
    this->cell_size = 10;

    // copy the state of the cells
    this->cells = new int[width * height];
    memcpy(cells, state, width * height * sizeof(int));
}

void Grid::draw() {
    for (int idx = 0; idx < width * height; idx++) {
        int row = idx % height;
        int col = idx / height;
        drawCell(row, col, cell_size);
    }
}

// get index of cell in 1D array in column major order
int Grid::getIdx(int row, int col) {
    return col * height + row;
}

void Grid::drawCell(int row, int col, int size) {
    if (isCellAlive(row, col)) {
        // shift by 200 to make the cells visible
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(col * size + 200, row * size), ImVec2((col + 1) * size + 200, (row + 1) * size), IM_COL32(255, 255, 255, 255));
    }
}

void Grid::clear() {
    memset(cells, 0, width * height * sizeof(int));
}

void Grid::randomize() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);

    for (int idx = 0; idx < width * height; idx++) {
        cells[idx] = dis(gen);
    }
}

void Grid::randomize(int seed, float density) {
    mt19937 gen(seed);
    uniform_real_distribution<> dis(0, 1);

    for (int idx = 0; idx < width * height; idx++) {
        cells[idx] = dis(gen) < density ? 1 : 0;
    }
}

void Grid::setState(int* state) {
    memcpy(cells, state, width * height * sizeof(int));
}

void Grid::setCellAlive(int row, int col, bool alive) {
    cells[getIdx(row, col)] = alive ? 1 : 0;
}

bool Grid::isCellAlive(int row, int col) {
    return cells[getIdx(row, col)] == 1;
}

int Grid::getWidth() {
    return width;
}

int Grid::getHeight() {
    return height;
}

void Grid::resize(int width, int height, int size) {
    this->width = width;
    this->height = height;
    this->cell_size = size;
    delete[] cells;
    cells = new int[width * height];
    clear();
}

void Grid::calcNeighbors(int* out) {
    for (int idx = 0; idx < width * height; idx++) {
        int row = idx % height;
        int col = idx / height;

        int liveNeighbors = 0;
        liveNeighbors += (row > 0 && col > 0) ? isCellAlive(row - 1, col - 1) : 0;
        liveNeighbors += (row > 0) ? isCellAlive(row - 1, col) : 0;
        liveNeighbors += (row > 0 && col < width - 1) ? isCellAlive(row - 1, col + 1) : 0;
        liveNeighbors += (col > 0) ? isCellAlive(row, col - 1) : 0;
        liveNeighbors += (col < width - 1) ? isCellAlive(row, col + 1) : 0;
        liveNeighbors += (row < height - 1 && col > 0) ? isCellAlive(row + 1, col - 1) : 0;
        liveNeighbors += (row < height - 1) ? isCellAlive(row + 1, col) : 0;
        liveNeighbors += (row < height - 1 && col < width - 1) ? isCellAlive(row + 1, col + 1) : 0;

        out[idx] = liveNeighbors;
    }
}

void Grid::gpuCalcNeighbors(int* out) {
    int size = width * height;
    int *h_cells = new int[size];
    int *h_output = new int[size];

    // copy the state of the cells to the host memory - column major
    memcpy(h_cells, cells, size * sizeof(int));
    memset(h_output, 0, size * sizeof(int));

    // call the kernel
    cuda_call_calc_neighbors_kernel(h_cells, h_output, width, height);

    // copy the output back to the cells
    memcpy(out, h_output, size * sizeof(int));

    delete[] h_cells;
    delete[] h_output;
}

void Grid::compareNeighbors(int* cpu, int* gpu) {
    for (int idx = 0; idx < width * height; idx++) {
        if (cpu[idx] != gpu[idx]) {
            cout << "Mismatch at index " << idx << " CPU: " << cpu[idx] << " GPU: " << gpu[idx] << "\n";
        }
    }
}

float Grid::update() {
    // cuda time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float time = 0.0f;

    int* newState = new int[width * height];
    memset(newState, 0, width * height * sizeof(int));

    for (int idx = 0; idx < width * height; idx++) {
        int row = idx % height;
        int col = idx / height;

        int liveNeighbors = 0;

        // count the number of live neighbors
        for (int x = col - 1; x <= col + 1; x++) {
            for (int y = row - 1; y <= row + 1; y++) {
                if (x >= 0 && x < width && y >= 0 && y < height && !(x == col && y == row)) {
                    if (isCellAlive(y, x) && !(x == col && y == row)) {
                        liveNeighbors++;
                    }
                }
            }
        }

        // apply the rules
        if (isCellAlive(row, col)) {
            if (liveNeighbors < 2 || liveNeighbors > 3) {
                newState[idx] = 0;
            } else {
                newState[idx] = 1;
            }
        } else {
            if (liveNeighbors == 3) {
                newState[idx] = 1;
            } else {
                newState[idx] = 0;
            }
        }
    }

    // update the state of the cells
    setState(newState);
    delete[] newState;

    // stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

float Grid::updateGPU() {
    int size = width * height;
    int *h_cells = new int[size];
    int *h_output = new int[size];

    // copy the state of the cells to the host memory - column major
    memcpy(h_cells, cells, size * sizeof(int));
    memset(h_output, 0, size * sizeof(int));

    // call the kernel
    float time = cuda_call_update_kernel(h_cells, h_output, width, height);

    // copy the output back to the cells
    setState(h_output);

    delete[] h_cells;
    delete[] h_output;

    return time;
}

#include <cudnn.h>

#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        cout << "Error at " << __FILE__ << ":" << __LINE__ << " " << cudnnGetErrorString(status) << "\n"; \
    } \
}

float* Grid::cudnnCalcNeighbors() {
    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1, 1,
                                          height, width));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1, 1,
                                          height, width));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          1, 1,
                                          3, 3));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                              1, 1, 1, 1,
                                              1, 1,
                                              CUDNN_CONVOLUTION,
                                              CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;
    int requested_algo_count = 1;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                                      input_descriptor,
                                                      kernel_descriptor,
                                                      convolution_descriptor,
                                                      output_descriptor,
                                                      1,
                                                      &requested_algo_count,
                                                      &convolution_algorithm));

    size_t workspace_bytes;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                      input_descriptor,
                                                      kernel_descriptor,
                                                      convolution_descriptor,
                                                      output_descriptor,
                                                      convolution_algorithm.algo,
                                                      &workspace_bytes));

    void* d_workspace;
    cudaMalloc(&d_workspace, workspace_bytes);

    float* h_cells = new float[width * height];
    float* h_output = new float[width * height];

    // copy the state of the cells to the host memory - column major
    for (int idx = 0; idx < width * height; idx++) {
        h_cells[idx] = cells[idx];
    }
    
    memset(h_output, 0, width * height * sizeof(float));

    float* d_input;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMemcpy(d_input, h_cells, width * height * sizeof(float), cudaMemcpyHostToDevice);

    float* d_output;
    cudaMalloc(&d_output, width * height * sizeof(float));

    float kernel[3][3] = {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    float* d_kernel;
    cudaMalloc(&d_kernel, 3 * 3 * sizeof(float));
    cudaMemcpy(d_kernel, kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                      &alpha,
                                      input_descriptor,
                                      d_input,
                                      kernel_descriptor,
                                      d_kernel,
                                      convolution_descriptor,
                                      convolution_algorithm.algo,
                                      d_workspace,
                                      workspace_bytes,
                                      &beta,
                                      output_descriptor,
                                      d_output));

    // copy the output back to the host memory
    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_workspace);

    delete[] h_cells;

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
    checkCUDNN(cudnnDestroy(cudnn));

    return h_output;
}

void Grid::compareCUDNN(float* cudnn_output, int* cpu_output) {
    for (int idx = 0; idx < width * height; idx++) {
        if (cudnn_output[idx] != cpu_output[idx]) {
            cout << "Mismatch at index " << idx << " CPU: " << cpu_output[idx] << " cuDNN: " << cudnn_output[idx] << "\n";
        }

        // print matching values
        if (cudnn_output[idx] == cpu_output[idx]) {
            cout << "Matching values at index " << idx << " CPU: " << cpu_output[idx] << " cuDNN: " << cudnn_output[idx] << "\n";
        }
    }
}

float Grid::updateCUDNN() {
    // cuda time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float time = 0.0f;

    float* h_output = cudnnCalcNeighbors();

    // apply the rules
    int *newState = new int[width * height];
    memset(newState, 0, width * height * sizeof(int));

    for (int idx = 0; idx < width * height; idx++) {
        int row = idx % height;
        int col = idx / height;

        if (isCellAlive(row, col)) {
            if (h_output[idx] < 2.0f || h_output[idx] > 3.0f) {
                newState[idx] = 0;
            } else {
                newState[idx] = 1;
            }
        } else {
            if (h_output[idx] == 3.0f) {
                newState[idx] = 1;
            } else {
                newState[idx] = 0;
            }
        }
    }

    // update the state of the cells
    setState(newState);
    delete[] newState;

    delete[] h_output;

    // stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}
