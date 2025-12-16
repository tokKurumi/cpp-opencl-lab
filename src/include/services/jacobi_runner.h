#pragma once

#include <memory>
#include <cstdint>

// Abstract interface for Jacobi solver implementations
class JacobiRunner
{
public:
    JacobiRunner(uint32_t grid_size, uint32_t iterations);
    virtual ~JacobiRunner();

    JacobiRunner(const JacobiRunner &) = delete;
    JacobiRunner &operator=(const JacobiRunner &) = delete;

    // Run the Jacobi solver
    // Returns the result array (for verification purposes)
    virtual float *run() = 0;

    uint32_t grid_size() const;
    uint32_t iterations() const;

protected:
    uint32_t _grid_size;
    uint32_t _iterations;
};
