#include "services/jacobi_benchmark.h"
#include "services/jacobi_gpu_global.h"
#include "services/jacobi_gpu_local.h"
#include "services/jacobi_gpu_texture.h"

#include <spdlog/spdlog.h>
#include <memory>

int main()
{
    // Configuration
    uint32_t grid_size = 512;
    uint32_t iterations = 100;

    spdlog::info("========================================");
    spdlog::info("Jacobi Benchmark: Grid {}x{}, {} iterations",
                 grid_size, grid_size, iterations);
    spdlog::info("========================================");
    spdlog::info("");

    // Run Global Memory benchmark
    {
        std::shared_ptr<JacobiRunner> global_solver = std::make_shared<JacobiGpuGlobal>(grid_size, iterations);
        JacobiBenchmark benchmark(global_solver, "Global Memory");
        benchmark.run();
    }

    spdlog::info("");

    // Run Local Memory benchmark
    {
        std::shared_ptr<JacobiRunner> local_solver = std::make_shared<JacobiGpuLocal>(grid_size, iterations);
        JacobiBenchmark benchmark(local_solver, "Local (Shared) Memory");
        benchmark.run();
    }

    spdlog::info("");

    // Run Texture Memory benchmark
    {
        std::shared_ptr<JacobiRunner> texture_solver = std::make_shared<JacobiGpuTexture>(grid_size, iterations);
        JacobiBenchmark benchmark(texture_solver, "Texture Memory");
        benchmark.run();
    }

    spdlog::info("");
    spdlog::info("========================================");
    spdlog::info("Benchmark Complete");
    spdlog::info("========================================");

    return 0;
}
