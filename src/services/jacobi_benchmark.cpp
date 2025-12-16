#include "services/jacobi_benchmark.h"
#include "services/jacobi_runner.h"

#include <spdlog/spdlog.h>
#include <chrono>

class JacobiBenchmark::Impl
{
public:
    std::shared_ptr<JacobiRunner> solver;
    std::string name;

    Impl(std::shared_ptr<JacobiRunner> solver, const std::string &name)
        : solver(solver), name(name)
    {
    }

    void run()
    {
        if (!solver)
        {
            spdlog::error("JacobiBenchmark: Solver is null");
            return;
        }

        spdlog::info("--- Running Jacobi with {} ---", name);
        auto start = std::chrono::high_resolution_clock::now();

        auto result = solver->run();

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (result)
        {
            spdlog::info("{}: Completed {} iterations in {} ms",
                         name, solver->iterations(), ms);
        }
        else
        {
            spdlog::warn("{}: Failed to execute", name);
        }
    }
};

JacobiBenchmark::JacobiBenchmark(std::shared_ptr<JacobiRunner> solver, const std::string &name)
    : _impl(std::make_unique<Impl>(solver, name))
{
}

JacobiBenchmark::~JacobiBenchmark() = default;

void JacobiBenchmark::run()
{
    _impl->run();
}
