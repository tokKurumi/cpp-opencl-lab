cmake_minimum_required(VERSION 3.10)

project(opencl-lab LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCLHeadersCpp REQUIRED)
find_package(OpenCLICDLoader REQUIRED)
find_package(spdlog REQUIRED)

add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=300)

add_executable(opencl-lab
    main.cpp
    services/jacobi_runner.cpp
    services/jacobi_gpu_texture.cpp
    services/jacobi_gpu_global.cpp
    services/jacobi_gpu_local.cpp
    services/jacobi_benchmark.cpp
)

target_include_directories(opencl-lab PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(opencl-lab
    OpenCL::HeadersCpp
    OpenCL::OpenCL
    spdlog::spdlog
)
