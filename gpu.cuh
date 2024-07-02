#include "cuda_header.cuh"

// function to call the cuda kernel
float cuda_call_update_kernel(const int* in_data, int* out_data, const unsigned int width, const unsigned int height);
float cuda_call_calc_neighbors_kernel(const int* in_data, int* out_data, const unsigned int width, const unsigned int height);