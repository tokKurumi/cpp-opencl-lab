#include "services/jacobi_runner.h"

JacobiRunner::JacobiRunner(uint32_t grid_size, uint32_t iterations)
    : _grid_size(grid_size), _iterations(iterations)
{
}

JacobiRunner::~JacobiRunner() = default;

uint32_t JacobiRunner::grid_size() const
{
    return _grid_size;
}

uint32_t JacobiRunner::iterations() const
{
    return _iterations;
}
