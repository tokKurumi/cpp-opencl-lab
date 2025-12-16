cmake_minimum_required(VERSION 3.10)

project(opencl-lab LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCLHeadersCpp QUIET)
find_package(OpenCLICDLoader QUIET)
find_package(spdlog REQUIRED)

set(USE_CONAN_OPENCL ${OpenCLHeadersCpp_FOUND})

# Always cap OpenCL C++ headers to 1.2 on macOS (Apple provides OpenCL 1.2)
if(APPLE)
    find_package(OpenCL REQUIRED)
    add_compile_definitions(
        CL_HPP_TARGET_OPENCL_VERSION=120
        CL_HPP_MINIMUM_OPENCL_VERSION=120
    )
else()
    # On non-Apple, prefer Conan-provided headers/loader with OpenCL 3.0 API
    if(USE_CONAN_OPENCL)
        add_compile_definitions(
            CL_HPP_TARGET_OPENCL_VERSION=300
            CL_HPP_MINIMUM_OPENCL_VERSION=120
        )
    else()
        find_package(OpenCL REQUIRED)
        add_compile_definitions(
            CL_HPP_TARGET_OPENCL_VERSION=300
            CL_HPP_MINIMUM_OPENCL_VERSION=120
        )
    endif()
endif()

add_executable(opencl-lab
    main.cpp
    config.cpp
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

# macOS specific includes
if(APPLE AND NOT USE_CONAN_OPENCL)
    target_include_directories(opencl-lab PRIVATE 
        /opt/homebrew/opt/opencl-clhpp-headers/include # Adjust path as necessary, now use homebrew
    )
endif()

if(USE_CONAN_OPENCL)
    target_link_libraries(opencl-lab
        OpenCL::HeadersCpp
        OpenCL::OpenCL
        spdlog::spdlog
    )
else()
    target_link_libraries(opencl-lab
        OpenCL::OpenCL
        spdlog::spdlog
    )
endif()
