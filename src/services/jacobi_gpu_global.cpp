#include "services/jacobi_gpu_global.h"

#include <CL/opencl.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <chrono>

class JacobiGpuGlobal::Impl
{
public:
    uint32_t grid_size;
    uint32_t iterations;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    bool is_initialized = false;
    float *result_buffer = nullptr;

    Impl(uint32_t grid_size, uint32_t iterations)
        : grid_size(grid_size), iterations(iterations)
    {
        init_opencl();
    }

    ~Impl()
    {
        if (result_buffer)
        {
            delete[] result_buffer;
        }
    }

    void init_opencl()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            spdlog::error("JacobiGpuGlobal: No OpenCL platforms found");
            return;
        }

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty())
        {
            spdlog::warn("JacobiGpuGlobal: No GPU devices, trying CPU");
            platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
        }

        if (devices.empty())
        {
            spdlog::error("JacobiGpuGlobal: No devices found");
            return;
        }

        device = devices[0];
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        std::string kernel_source = load_kernel_source();
        cl::Program::Sources sources;
        sources.push_back({kernel_source.c_str(), kernel_source.length()});

        cl::Program program(context, sources);
        if (program.build({device}) != CL_SUCCESS)
        {
            spdlog::error("JacobiGpuGlobal: Build error: {}", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
            return;
        }

        kernel = cl::Kernel(program, "jacobi_global");
        is_initialized = true;
        spdlog::info("JacobiGpuGlobal: Initialized with {}", device.getInfo<CL_DEVICE_NAME>());
    }

    std::string load_kernel_source()
    {
        std::ifstream file("src/services/jacobi_kernels.cl");
        if (!file.is_open())
        {
            spdlog::warn("jacobi_kernels.cl not found, using embedded kernel");
            return "__kernel void jacobi_global(__global const float *input, __global float *output, uint grid_size) {}";
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    float *run()
    {
        if (!is_initialized)
        {
            spdlog::error("JacobiGpuGlobal: Not initialized");
            return nullptr;
        }

        auto start = std::chrono::high_resolution_clock::now();

        size_t size = grid_size * grid_size;
        float *data = new float[size];

        for (uint32_t i = 0; i < grid_size; ++i)
            for (uint32_t j = 0; j < grid_size; ++j)
                data[i * grid_size + j] = (i == 0 || i == grid_size - 1 || j == 0 || j == grid_size - 1) ? 1.0f : 0.0f;

        size_t bytes = size * sizeof(float);
        cl::Buffer buf_in(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, data);
        cl::Buffer buf_out(context, CL_MEM_READ_WRITE, bytes);

        for (uint32_t iter = 0; iter < iterations; ++iter)
        {
            kernel.setArg(0, buf_in);
            kernel.setArg(1, buf_out);
            kernel.setArg(2, grid_size);
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grid_size, grid_size), cl::NullRange);
            queue.finish();
            std::swap(buf_in, buf_out);
        }

        queue.enqueueReadBuffer(buf_in, CL_TRUE, 0, bytes, data);
        result_buffer = data;

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        spdlog::info("JacobiGpuGlobal: {} iterations in {} ms", iterations, ms);

        return data;
    }
};

JacobiGpuGlobal::JacobiGpuGlobal(uint32_t grid_size, uint32_t iterations)
    : JacobiRunner(grid_size, iterations), _impl(std::make_unique<Impl>(grid_size, iterations))
{
}

JacobiGpuGlobal::~JacobiGpuGlobal() = default;

float *JacobiGpuGlobal::run()
{
    return _impl->run();
}

uint32_t JacobiGpuGlobal::grid_size() const
{
    return _impl->grid_size;
}

uint32_t JacobiGpuGlobal::iterations() const
{
    return _impl->iterations;
}
