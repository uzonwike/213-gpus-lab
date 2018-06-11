#include <stdint.h>
#include <stdio.h>

#define N 32
#define THREADS_PER_BLOCK 32

__global__ void dotproduct(float* x, float* y, float* result) {
    // Compute the index this thread should use to access elements
    size_t index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

    // Create space for a shared array that all threads in this block will
    // use to store pairwise products
    __shared__ float temp[THREADS_PER_BLOCK];

    // Compute pairwise products
    temp[threadIdx.x] = x[index] * y[index];

    __syncthreads();
    // The thread with index zero will sum up the values in temp
    if(threadIdx.x == 0) {
        float sum = 0;
        int i;
        for(i=0; i<THREADS_PER_BLOCK; i++) {
            atomicAdd(&sum, temp[i]);
        }

        // Add the sum for this block to the result
        *result += sum;
    }
}

int main() {
    // Allocate arrays for X and Y on the CPU
    float* cpu_x = (float*)malloc(sizeof(float) * N);
    float* cpu_y = (float*)malloc(sizeof(float) * N);

    // Initialize X and Y
    int i;
    for(i=0; i<N; i++) {
        cpu_x[i] = (float)i;
        cpu_y[i] = (float)i;
    }

    // Allocate space for X and Y on the GPU
    float* gpu_x;
    float* gpu_y;
    float gpu_result = 0.0;

    if(cudaMalloc(&gpu_x, sizeof(float) * N) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate X array on GPU\n");
        exit(2);
    }

    if(cudaMalloc(&gpu_y, sizeof(float) * N) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate Y array on GPU\n");
        exit(2);
    }

    // Copy the host X and Y arrays to the device X and Y
    // arrays
    if(cudaMemcpy(gpu_x, cpu_x, sizeof(float) * N, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        fprintf(stderr, "Failed to copy X to the GPU\n");
    }

    if(cudaMemcpy(gpu_y, cpu_y, sizeof(float) * N,
                cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy Y to the GPU\n");
    }

    // How many blocks should be run, rounding up to
    // include all threads?
    size_t blocks = (N + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK;

    // Run the saxpy kernel
    dotproduct<<<blocks, THREADS_PER_BLOCK>>>(gpu_x, gpu_y, &gpu_result);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();


    printf("%f\n", gpu_result);

    cudaFree(gpu_x);
    cudaFree(gpu_y);
    free(cpu_x);
    free(cpu_y);

    return 0;
}
