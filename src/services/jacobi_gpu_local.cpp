#include "services/jacobi_gpu_local.h"
#include "services/kernels.h"

#include <CL/opencl.hpp>
#include <spdlog/spdlog.h>
#include <chrono>

class JacobiGpuLocal::Impl
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
        : grid_size(grid_size), iterations(iterations),
          result_buffer(nullptr)
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
        // Get platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            spdlog::error("JacobiGpuLocal: No OpenCL platforms found");
            return;
        }

        // Get GPU device
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty())
        {
            spdlog::warn("JacobiGpuLocal: No GPU devices found, trying CPU");
            platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
        }

        if (devices.empty())
        {
            spdlog::error("JacobiGpuLocal: No devices found");
            return;
        }

        device = devices[0];
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        // Load kernel source
        std::string kernel_source = load_kernel_source();
        cl::Program::Sources sources;
        sources.push_back({kernel_source.c_str(), kernel_source.length()});

        cl::Program program(context, sources);
        if (program.build({device}) != CL_SUCCESS)
        {
            spdlog::error("JacobiGpuLocal: Error building kernel: {}",
                          program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
            return;
        }

        kernel = cl::Kernel(program, "jacobi_local");
        is_initialized = true;

        spdlog::info("JacobiGpuLocal: OpenCL initialized with device: {}",
                     device.getInfo<CL_DEVICE_NAME>());
    }

    std::string load_kernel_source()
    {
        return std::string(kernels::JACOBI_KERNELS);
    }

    float *run()
    {
        if (!is_initialized)
        {
            spdlog::error("JacobiGpuLocal: OpenCL not initialized");
            return nullptr;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Allocate host memory
        size_t buffer_size = grid_size * grid_size * sizeof(float);
        float *host_data = new float[grid_size * grid_size];
        float *host_data_swap = new float[grid_size * grid_size];

        // Initialize: boundaries = 1.0, interior = 0.0
        for (uint32_t i = 0; i < grid_size; ++i)
        {
            for (uint32_t j = 0; j < grid_size; ++j)
            {
                if (i == 0 || i == grid_size - 1 || j == 0 || j == grid_size - 1)
                {
                    host_data[i * grid_size + j] = 1.0f;
                }
                else
                {
                    host_data[i * grid_size + j] = 0.0f;
                }
            }
        }

        // Create device buffers
        cl::Buffer buffer_input(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                buffer_size, host_data);
        cl::Buffer buffer_output(context, CL_MEM_READ_WRITE,
                                 buffer_size);

        // Local memory size (8x8 work group + 1 halo element on each side)
        size_t local_size_x = 8;
        size_t local_size_y = 8;
        size_t local_mem_size = (local_size_x + 2) * (local_size_y + 2) * sizeof(float);

        // Jacobi iterations
        for (uint32_t iter = 0; iter < iterations; ++iter)
        {
            kernel.setArg(0, buffer_input);
            kernel.setArg(1, buffer_output);
            kernel.setArg(2, grid_size);
            kernel.setArg(3, cl::LocalSpaceArg(local_mem_size));

            queue.enqueueNDRangeKernel(kernel,
                                       cl::NullRange,
                                       cl::NDRange(grid_size, grid_size),
                                       cl::NDRange(local_size_x, local_size_y));
            queue.finish();

            // Swap buffers
            std::swap(buffer_input, buffer_output);
        }

        // Read result back
        queue.enqueueReadBuffer(buffer_input, CL_TRUE, 0,
                                buffer_size, host_data);

        result_buffer = host_data;
        delete[] host_data_swap;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        spdlog::info("JacobiGpuLocal: Completed {} iterations in {} ms",
                     iterations, duration.count());

        return host_data;
    }
};

JacobiGpuLocal::JacobiGpuLocal(uint32_t grid_size, uint32_t iterations)
    : JacobiRunner(grid_size, iterations), _impl(std::make_unique<Impl>(grid_size, iterations))
{
}

JacobiGpuLocal::~JacobiGpuLocal() = default;

float *JacobiGpuLocal::run()
{
    return _impl->run();
}

uint32_t JacobiGpuLocal::grid_size() const
{
    return _impl->grid_size;
}

uint32_t JacobiGpuLocal::iterations() const
{
    return _impl->iterations;
}
