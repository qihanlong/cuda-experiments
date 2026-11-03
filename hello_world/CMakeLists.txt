cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_ARCHITECTURES "native")

include("${CMAKE_CURRENT_SOURCE_DIR}/../cuda-samples/cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(matrixMul LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../cuda-samples/cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../cuda-samples/Common)

add_executable(cudaHello hello.cu)

target_compile_options(cudaHello PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cudaHello PRIVATE cxx_std_17 cuda_std_17)

include(InstallSamples)
setup_samples_install()
