#include <cmath>
#include <chrono>

#define N 10000000

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}


__global__ void vector_add(float* out, float* a, float* b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Handling arbitrary vector size
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

__global__ void vector_add_stride(float* out, float* a, float* b, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

// works, but wrong implementation
__global__ void vector_add_block(float* out, float* a, float* b, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    int block_size = N / blockDim.x + 1;
    int end = min(N, (index + 1) * block_size);

    for (int i = index * block_size; i < end; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float* a, * b, * out;

    float* d_a, *d_b, *d_out;

    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f; b[i] = 2.0f;
    }

    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    // vector_add_block << <1, 32 >> > (d_out, d_a, d_b, N);
    int block_size = 256;
    int grid_size = (N / block_size) + 1;
    vector_add << <grid_size, block_size >> > (d_out, d_a, d_b, N);
	
    auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; i++) {
        printf("out[%d] = %f\n", i, out[i]);
    }

    for (int i = 0; i < 100; i++) {
        printf("out[%d] = %f\n", N - 100 + i, out[N - 100 + i]);
    }

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a);
    free(b);
    free(out);

    printf("Addition took: %d microseconds\n", duration.count());

    cuda_hello << <1, 1 >> > ();
    return 0;
}