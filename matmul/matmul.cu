#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define ERROR_THRESHOLD 0.0001f

__global__ void matMulNaiveCuda(float* a, int height_a, float* b, int width_b, int n, float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= height_a * width_b) {
        return;
    }
    
    int i_a = tid / width_b;
    int i_b = tid % width_b;
    
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

void matMulCPUAux(float* a, float* b, float* c, int m, int p, int n, int tid) {
    if (tid >= m * p) {
        return;
    }
    
    int a_row = tid / p;
    int b_column = tid % p;
    
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i + a_row * n] * b[b_column + i * p];
    }
    c[tid] = sum;
}

void multMulCPU(float* a, float* b, float* c, int m, int p, int n) {
    int threads = m * p + 1;
    for (int i = 0; i < threads; i++) {
        matMulCPUAux(a, b, c, m, p, n, i);
    }
}

// Make a version where matrix b is column based?
// version where the block size is sensitive to matrix size

void printMatrix(float* a, int m, int n) {
    for (int i = 0; i < m; i++) {
        std::cout << "[";
        for (int j = 0; j < n; j++) {
            std::cout << a[i * n + j] << " ";
        }
        std::cout << "]" << std::endl;
    }
}

void setupTest1(float* a, float* b, float* c) {
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
    
    for (int i = 0; i < 4; i++) {
        c[i] = 0;
    }
}

void runTest1() {
    float* a, * b, * c, *cpu;

    int m = 2;
    int p = 2;
    int n = 3;
    
    a = (float*)malloc(sizeof(float) * m * n);
    b = (float*)malloc(sizeof(float) * p * n);
    c = (float*)malloc(sizeof(float) * m * p);
    cpu = (float*)malloc(sizeof(float) * m * p);
    
    setupTest1(a, b, c);
    

    auto start = std::chrono::high_resolution_clock::now();
    multMulCPU(a, b, cpu, m, p, n);
	
    auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "CPU run complete. Took " << duration.count() << " milliseconds. Output:" << std::endl;
    printMatrix(cpu, m, p);
    std::cout << std::endl;
    
    


    start = std::chrono::high_resolution_clock::now();
    multMulNaive(a, m, b, p, n, c);
	
    end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Naive run complete. Took " << duration.count() << " milliseconds." << std::endl;
    
    bool match = true;
    for (int i = 0; i < m * p; i++) {
        match = match && (c[i] == cpu[i]);
        if (!match) {
            break;
        }
    }
    if (match) {
        std::cout << "Matches CPU output" << std::endl;
    } else {
        std::cout << "ERROR: Does not match CPU output" << std::endl;
    }
    
    std::cout << "Output:" << std::endl;
    printMatrix(c, m, p);
    
    free(a);
    free(b);
    free(c);
    free(cpu);
}

void setupTest2(float* a, float* b, float* c) {
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
    
    for (int i = 0; i < 8; i++) {
        c[i] = 0;
    }
}

void runTest2() {
    float* a, * b, * c, * cpu;

    int m = 4;
    int p = 2;
    int n = 2;
    
    a = (float*)malloc(sizeof(float) * m * n);
    b = (float*)malloc(sizeof(float) * n * p);
    c = (float*)malloc(sizeof(float) * m * p);
    cpu = (float*)malloc(sizeof(float) * m * p);
    
    setupTest2(a, b, c);
    
    auto start = std::chrono::high_resolution_clock::now();
    multMulCPU(a, b, cpu, m, p, n);
	
    auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "CPU run complete. Took " << duration.count() << " microseconds. Output:" << std::endl;
    printMatrix(cpu, m, p);
    
    

    start = std::chrono::high_resolution_clock::now();
    multMulNaive(a, m, b, p, n, c);
	
    end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Naive run complete. Took " << duration.count() << " microseconds." << std::endl;
    
    bool match = true;
    for (int i = 0; i < m * p; i++) {
        match = match && (std::abs(c[i] - cpu[i]) < ERROR_THRESHOLD);
    }
    if (match) {
        std::cout << "Matches CPU output" << std::endl;
    } else {
        std::cout << "ERROR: Does not match CPU output" << std::endl;
    }
    
    std::cout << "Output:" << std::endl;
    printMatrix(c, m, p);
    
    free(a);
    free(b);
    free(c);
    free(cpu);
}

float randfloat(){
    return (float)rand() / (float)(RAND_MAX);
}

void setupTestRandom(float* a, float* b, float* c, int size_a, int size_b, int size_c) {
    for (int i = 0; i < size_a; i++) {
        a[i] = randfloat();
    }

    for (int i = 0; i < size_b; i++) {
        b[i] = randfloat();
    }
    
    for (int i = 0; i < size_c; i++) {
        c[i] = 0;
    }
}

void runTest3() {
    float* a, * b, * c, * cpu;

    int m = 256;
    int p = 192;
    int n = 128;
    
    a = (float*)malloc(sizeof(float) * m * n);
    b = (float*)malloc(sizeof(float) * n * p);
    c = (float*)malloc(sizeof(float) * m * p);
    cpu = (float*)malloc(sizeof(float) * m * p);
    
    std::srand(3);
    setupTestRandom(a, b, c, m * n, n * p, m * p);
    
    auto start = std::chrono::high_resolution_clock::now();
    multMulCPU(a, b, cpu, m, p, n);
	
    auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "CPU run complete. Took " << duration.count() << " microseconds." << std::endl;
    
    

    start = std::chrono::high_resolution_clock::now();
    multMulNaive(a, m, b, p, n, c);
	
    end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Naive run complete. Took " << duration.count() << " microseconds." << std::endl;
    
    bool match = true;
    int index = -1;
    for (int i = 0; i < m * p; i++) {
        match = match && (std::abs(c[i] - cpu[i]) < ERROR_THRESHOLD);
        if (!match) {
            index = i;
            break;
        }
    }
    if (match) {
        std::cout << "Matches CPU output" << std::endl;
    } else {
        std::cout << "ERROR: Does not match CPU output. First error found at index " << index << std::endl;
        std::cout << "CPU got " << cpu[index] << " vs Cuda: " << c[index] << std::endl;
    }
    
    free(a);
    free(b);
    free(c);
    free(cpu);
}

void runTest4() {
    float* a, * b, * c, * cpu;

    int m = 4000;
    int p = 3000;
    int n = 1;
    
    a = (float*)malloc(sizeof(float) * m * n);
    b = (float*)malloc(sizeof(float) * n * p);
    c = (float*)malloc(sizeof(float) * m * p);
    cpu = (float*)malloc(sizeof(float) * m * p);
    
    std::srand(4);
    setupTestRandom(a, b, c, m * n, n * p, m * p);
    
    auto start = std::chrono::high_resolution_clock::now();
    multMulCPU(a, b, cpu, m, p, n);
	
    auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "CPU run complete. Took " << duration.count() << " microseconds." << std::endl;
    
    

    start = std::chrono::high_resolution_clock::now();
    multMulNaive(a, m, b, p, n, c);
	
    end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Naive run complete. Took " << duration.count() << " microseconds." << std::endl;
    
    bool match = true;
    int index = -1;
    for (int i = 0; i < m * p; i++) {
        match = match && (std::abs(c[i] - cpu[i]) < ERROR_THRESHOLD);
        if (!match) {
            index = i;
            break;
        }
    }
    if (match) {
        std::cout << "Matches CPU output" << std::endl;
    } else {
        std::cout << "ERROR: Does not match CPU output. First error found at index " << index << std::endl;
        std::cout << "CPU got " << cpu[index] << " vs Cuda: " << c[index] << std::endl;
    }
    
    free(a);
    free(b);
    free(c);
    free(cpu);
}

void runTest5() {
    float* a, * b, * c, * cpu;

    int m = 400;
    int p = 600;
    int n = 1000;
    
    a = (float*)malloc(sizeof(float) * m * n);
    b = (float*)malloc(sizeof(float) * n * p);
    c = (float*)malloc(sizeof(float) * m * p);
    cpu = (float*)malloc(sizeof(float) * m * p);
    
    std::srand(5);
    setupTestRandom(a, b, c, m * n, n * p, m * p);
    
    auto start = std::chrono::high_resolution_clock::now();
    multMulCPU(a, b, cpu, m, p, n);
	
    auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "CPU run complete. Took " << duration.count() << " microseconds." << std::endl;
    
    

    start = std::chrono::high_resolution_clock::now();
    multMulNaive(a, m, b, p, n, c);
	
    end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Naive run complete. Took " << duration.count() << " microseconds." << std::endl;
    
    bool match = true;
    int index = -1;
    for (int i = 0; i < m * p; i++) {
        match = match && (std::abs(c[i] - cpu[i]) < ERROR_THRESHOLD);
        if (!match) {
            index = i;
            break;
        }
    }
    if (match) {
        std::cout << "Matches CPU output" << std::endl;
    } else {
        std::cout << "ERROR: Does not match CPU output. First error found at index " << index << std::endl;
        std::cout << "CPU got " << cpu[index] << " vs Cuda: " << c[index] << std::endl;
    }
    
    free(a);
    free(b);
    free(c);
    free(cpu);
}

int main() {
    std::cout << "Test 1" << std::endl;
    runTest1();
    std::cout << std::endl << "Test 2" << std::endl;
    runTest2();
    std::cout << std::endl << "Test 3" << std::endl;
    runTest3();
    std::cout << std::endl << "Test 4" << std::endl;
    runTest4();
    std::cout << std::endl << "Test 5" << std::endl;
    runTest5();
    return 0;
}