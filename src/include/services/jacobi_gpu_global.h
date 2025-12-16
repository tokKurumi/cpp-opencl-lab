#pragma once

#include "services/jacobi_runner.h"
#include <memory>
#include <cstdint>

// OpenCL global memory implementation
class JacobiGpuGlobal final : public JacobiRunner
{
public:
    JacobiGpuGlobal(uint32_t grid_size, uint32_t iterations);
    ~JacobiGpuGlobal();

    JacobiGpuGlobal(const JacobiGpuGlobal &) = delete;
    JacobiGpuGlobal &operator=(const JacobiGpuGlobal &) = delete;

    // Run the Jacobi solver using global memory
    // Returns the result array
    float *run();

    uint32_t grid_size() const;
    uint32_t iterations() const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};
