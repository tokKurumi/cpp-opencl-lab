#pragma once

#include "services/jacobi_runner.h"
#include <memory>
#include <cstdint>

// OpenCL texture memory implementation
class JacobiGpuTexture final : public JacobiRunner
{
public:
    JacobiGpuTexture(uint32_t grid_size, uint32_t iterations);
    ~JacobiGpuTexture();

    JacobiGpuTexture(const JacobiGpuTexture &) = delete;
    JacobiGpuTexture &operator=(const JacobiGpuTexture &) = delete;

    // Run the Jacobi solver using texture memory
    // Returns the result array
    float *run();

    uint32_t grid_size() const;
    uint32_t iterations() const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};
