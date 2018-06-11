#include <stdio.h>

__global__ void kernel() {
    printf("Hello world!\n");
}

int main() {
    kernel <<< 2, 5 >>>();
    cudaDeviceSynchronize();
    return 0;
}
