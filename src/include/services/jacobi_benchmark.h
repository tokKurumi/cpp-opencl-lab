#pragma once

#include <memory>
#include <cstdint>
#include <string>

class JacobiRunner;

// Benchmark that measures performance of a single Jacobi solver implementation
class JacobiBenchmark
{
public:
    JacobiBenchmark(std::shared_ptr<JacobiRunner> solver, const std::string &name);
    ~JacobiBenchmark();

    JacobiBenchmark(const JacobiBenchmark &) = delete;
    JacobiBenchmark &operator=(const JacobiBenchmark &) = delete;

    // Run the solver and log execution time
    void run();

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};
