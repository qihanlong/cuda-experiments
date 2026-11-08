
#include <iostream>

__global__ void matMulNaiveCuda(float* a, int height_a, float* b, int width_b, int n, float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= height_a * width_b) {
        return;
    }
    
    int i_a = tid / height_a;
    int i_b = tid % height_a;
    
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

void setupTest1(float* a, float* b) {
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
}

void runTest1() {
    float* a, * b, * c;

    int m = 2;
    int p = 2;
    int n = 3;
    
    a = (float*)malloc(sizeof(float) * m * n);
    b = (float*)malloc(sizeof(float) * p * n);
    c = (float*)malloc(sizeof(float) * m * p);
    
    setupTest1(a, b);
    
    multMulNaive(a, m, b, p, n, c);

    std::cout << "Naive run complete. Output:" << std::endl;
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << c[i * p + j] << " ";
        }
        std::cout << std::endl;
    }
    
    free(a);
    free(b);
    free(c);
}

void setupTest2(float* a, float* b) {
    // a:
    // [4, 2.3]
    // [0, 2]
    // [1, -0.5]
    // [10000, -99.99]
    a[0] = 4;
    a[1] = 2.3;
    a[2] = 0;
    a[3] = 2;
    a[4] = 1;
    a[5] = -0.5;
    a[6] = 10000;
    a[7] = -99.99;
    
    // b: Identity
    // [1, 0]
    // [0, 1]
    b[0] = 1;
    b[1] = 0;
    b[2] = 0;
    b[3] = 1;
}

void runTest2() {
    float* a, * b, * c;

    int m = 4;
    int p = 2;
    int n = 2;
    
    a = (float*)malloc(sizeof(float) * m * n);
    b = (float*)malloc(sizeof(float) * n * p);
    c = (float*)malloc(sizeof(float) * m * p);
    
    setupTest2(a, b);
    
    multMulNaive(a, m, b, p, n, c);

    std::cout << "Naive run complete. Output:" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            std::cout << c[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
    
    free(a);
    free(b);
    free(c);
}

int main() {
    std::cout << "Test 1" << std::endl;
    runTest1();
    std::cout << std::endl << "Test 2" << std::endl;
    runTest2();
    return 0;
}