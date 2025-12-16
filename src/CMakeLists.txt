cmake_minimum_required(VERSION 3.10)

project(opencl-lab LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Try to find OpenCL packages from Conan first
find_package(OpenCLHeadersCpp QUIET)
find_package(OpenCLICDLoader QUIET)
find_package(spdlog REQUIRED)

# If Conan packages not found (macOS), use system OpenCL
if(NOT OpenCLHeadersCpp_FOUND OR NOT OpenCLICDLoader_FOUND)
    find_package(OpenCL REQUIRED)
    
    # On macOS, use Homebrew headers for C++ API
    if(APPLE)
        set(OPENCL_HPP_INCLUDE_DIR "/opt/homebrew/opt/opencl-clhpp-headers/include")
        add_compile_definitions(
            CL_HPP_TARGET_OPENCL_VERSION=120
            CL_HPP_MINIMUM_OPENCL_VERSION=120
        )
    endif()
else()
    # Using Conan packages (Linux/Windows)
    add_compile_definitions(
        CL_HPP_TARGET_OPENCL_VERSION=300
        CL_HPP_MINIMUM_OPENCL_VERSION=120
    )
endif()

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

# Link libraries: use Conan packages if available, otherwise system OpenCL
if(OpenCLHeadersCpp_FOUND AND OpenCLICDLoader_FOUND)
    # Using Conan packages
    target_link_libraries(opencl-lab
        OpenCL::HeadersCpp
        OpenCL::OpenCL
        spdlog::spdlog
    )
else()
    # Using system OpenCL
    if(APPLE AND OPENCL_HPP_INCLUDE_DIR)
        target_include_directories(opencl-lab PRIVATE ${OPENCL_HPP_INCLUDE_DIR})
    endif()
    target_link_libraries(opencl-lab
        OpenCL::OpenCL
        spdlog::spdlog
    )
endif()
