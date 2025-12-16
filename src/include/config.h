#pragma once

#include <memory>
#include <cstdint>

class Config
{
public:
    /**
     * Parse configuration from command line arguments.
     * @param argc Argument count
     * @param argv Argument values
     */
    Config(int argc, char *argv[]);
    ~Config();

    Config(const Config &) = delete;
    Config &operator=(const Config &) = delete;

    /**
     * Get the grid size (width and height).
     * @return Grid size (default: 512)
     */
    uint32_t get_grid_size() const;

    /**
     * Get the number of iterations.
     * @return Number of iterations (default: 100)
     */
    uint32_t get_iterations() const;

    /**
     * Check if help was requested.
     * @return true if --help or -h was provided
     */
    bool is_help_requested() const;

    /**
     * Print usage information.
     * @param program_name The name of the program
     */
    static void print_usage(const char *program_name);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};
