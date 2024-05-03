#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000

__global__ void add(int *a, int *b, int *c) {
    int tid = threadIdx.x; // this thread handles the data at its thread index
    for (int i = tid; i < N; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    struct timespec t_start, t_end;
    double elapsedTimeCPU;

    // Start CPU timing
    clock_gettime(CLOCK_REALTIME, &t_start);

    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    // Allocate memory on the CPU
    a = (int *)malloc(N * sizeof(int));
    b = (int *)malloc(N * sizeof(int));
    c = (int *)malloc(N * sizeof(int));

    // Allocate memory on the GPU
    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));

    // Fill the arrays 'a' and 'b' on the CPU
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 256;
        b[i] = rand() % 256;
    }

    // Copy arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // CUDA event creation and start recording
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // GPU kernel function
    add<<<1, 1024>>>(dev_a, dev_b, dev_c);

    // CUDA event end and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Compute execution time for GPU
    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, start, stop);

    // Copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the GPU computation
    bool success = true;
    for (int i = 0; i < N; i++) {
        if ((a[i] + b[i]) != c[i]) {
            printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
            success = false;
        }
    }
    if (success) {
        printf("We did it!\n");
    }

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Stop CPU timing
    clock_gettime(CLOCK_REALTIME, &t_end);

    // Compute and print the elapsed time in millisec for CPU
    elapsedTimeCPU = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsedTimeCPU += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;

    printf("GPU time: %f msec\n", elapsedTimeGPU);
    printf("CPU elapsedTime: %lf ms\n", elapsedTimeCPU);

    return 0;
}
