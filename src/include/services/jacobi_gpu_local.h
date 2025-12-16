#pragma once

#include "services/jacobi_runner.h"
#include <memory>
#include <cstdint>

// OpenCL local (shared) memory implementation
class JacobiGpuLocal final : public JacobiRunner
{
public:
    JacobiGpuLocal(uint32_t grid_size, uint32_t iterations);
    ~JacobiGpuLocal();

    JacobiGpuLocal(const JacobiGpuLocal &) = delete;
    JacobiGpuLocal &operator=(const JacobiGpuLocal &) = delete;

    // Run the Jacobi solver using local memory
    // Returns the result array
    float *run();

    uint32_t grid_size() const;
    uint32_t iterations() const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};
