
#include <iostream>

__global__ void matMulNaiveCuda(float* a, int height_a, float* b, int width_b, int n, float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= height_a * width_b) {
        return;
    }
    
    int i_a = tid % height_a;
    int i_b = tid % n;
    
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i + i_a * n] * b[i_b + i * width_b];
    }
    c[tid] = sum;
}

void multMulNaive(float* a, int height_a, float* b, int width_b, int n, float* c) {
    int a_size = height_a * n;
    int b_size = width_b * n;
    int c_size = height_a * width_b;

    float* d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, sizeof(float) * a_size);
    cudaMalloc((void**)&d_b, sizeof(float) * b_size);
    cudaMalloc((void**)&d_c, sizeof(float) * c_size);
    cudaMemcpy(d_a, a, sizeof(float) * a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * b_size, cudaMemcpyHostToDevice);
    
    
    int block_size = 256;
    int grid_size = (c_size / block_size) + 1;
    matMulNaiveCuda << <grid_size, block_size >> > (d_a, height_a, d_b, width_b, n, d_c);

    cudaMemcpy(c, d_c, sizeof(float) * c_size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Make a version where matric b is column based?
// version where the block size is sensitive to matrix size

int main() {
    float* a, * b, * c;

    int a_h = 2;
    int b_w = 2;
    int n = 3;
    
    a = (float*)malloc(sizeof(float) * a_h * n);
    b = (float*)malloc(sizeof(float) * b_w * n);
    c = (float*)malloc(sizeof(float) * a_h * b_w);
    
    // a:
    // [4, 2.3, 0]
    // [2, 1, -0.5]
    a[0] = 4;
    a[1] = 2.3;
    a[2] = 0;
    a[3] = 2;
    a[4] = 1;
    a[5] = -0.5;
    
    // b:
    // [-2, 1]
    // [2, 0.25]
    // [0, 10000]
    b[0] = -2;
    b[1] = 1;
    b[2] = 2;
    b[3] = 0.25;
    b[4] = 0;
    b[5] = 10000;
    
    multMulNaive(a, a_h, b, b_w, n, c);

    std::cout << "Run complete. Output:" << std::endl;
    for (int i = 0; i < a_h; i++) {
        for (int j = 0; j < b_w; j++) {
            std::cout << c[i + j * b_w] << " ";
        }
        std::cout << std::endl;
    }
    
    free(a);
    free(b);
    free(c);
    return 0;
}