#include "config.h"

#include <spdlog/spdlog.h>
#include <iostream>
#include <string>

class Config::Impl
{
public:
    uint32_t grid_size = 512;
    uint32_t iterations = 100;
    bool help_requested = false;

    void parse(int argc, char *argv[])
    {
        if (argc > 1)
        {
            std::string arg1(argv[1]);
            if (arg1 == "--help" || arg1 == "-h")
            {
                help_requested = true;
                return;
            }

            try
            {
                grid_size = std::stoul(arg1);
            }
            catch (const std::exception &e)
            {
                spdlog::warn("Invalid grid_size: {}. Using default (512).", arg1);
                grid_size = 512;
            }
        }

        if (argc > 2)
        {
            try
            {
                iterations = std::stoul(argv[2]);
            }
            catch (const std::exception &e)
            {
                spdlog::warn("Invalid iterations: {}. Using default (100).", argv[2]);
                iterations = 100;
            }
        }

        if (argc > 3)
        {
            spdlog::warn("Too many arguments. Expected at most 2, got {}.", argc - 1);
        }
    }
};

Config::Config(int argc, char *argv[])
    : pimpl(std::make_unique<Impl>())
{
    pimpl->parse(argc, argv);
}

Config::~Config() = default;

uint32_t Config::get_grid_size() const
{
    return pimpl->grid_size;
}

uint32_t Config::get_iterations() const
{
    return pimpl->iterations;
}

bool Config::is_help_requested() const
{
    return pimpl->help_requested;
}

void Config::print_usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [grid_size] [iterations]\n"
              << "  grid_size:  Size of the grid (default: 512)\n"
              << "  iterations: Number of iterations (default: 100)\n"
              << "\nExample: " << program_name << " 1024 200\n";
}
