#pragma once

#include <string_view>

namespace kernels
{
    constexpr std::string_view JACOBI_KERNELS = R"(
__kernel void jacobi_texture(
    __read_only image2d_t input,
    __write_only image2d_t output,
    uint grid_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x == 0 || x == grid_size - 1 || y == 0 || y == grid_size - 1)
    {
        write_imagef(output, (int2)(x, y), (float4)(1.0f, 0.0f, 0.0f, 0.0f));
        return;
    }

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float top = read_imagef(input, sampler, (int2)(x, y - 1)).x;
    float bottom = read_imagef(input, sampler, (int2)(x, y + 1)).x;
    float left = read_imagef(input, sampler, (int2)(x - 1, y)).x;
    float right = read_imagef(input, sampler, (int2)(x + 1, y)).x;

    float avg = (top + bottom + left + right) * 0.25f;
    write_imagef(output, (int2)(x, y), (float4)(avg, 0.0f, 0.0f, 0.0f));
}

__kernel void jacobi_global(
    __global const float *input,
    __global float *output,
    uint grid_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x == 0 || x == grid_size - 1 || y == 0 || y == grid_size - 1)
    {
        output[y * grid_size + x] = 1.0f;
        return;
    }

    int idx = y * grid_size + x;
    float top = input[(y - 1) * grid_size + x];
    float bottom = input[(y + 1) * grid_size + x];
    float left = input[y * grid_size + (x - 1)];
    float right = input[y * grid_size + (x + 1)];

    output[idx] = (top + bottom + left + right) * 0.25f;
}

__kernel void jacobi_local(
    __global const float *input,
    __global float *output,
    uint grid_size,
    __local float *local_data)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);

    // Load data into local memory (including halo)
    int local_idx = ly * (local_size_x + 2) + lx + 1;
    local_data[local_idx] = input[y * grid_size + x];

    // Load halo elements
    if (lx == 0 && x > 0)
    {
        local_data[ly * (local_size_x + 2)] = input[y * grid_size + (x - 1)];
    }
    if (lx == local_size_x - 1 && x < grid_size - 1)
    {
        local_data[ly * (local_size_x + 2) + local_size_x + 1] = input[y * grid_size + (x + 1)];
    }
    if (ly == 0 && y > 0)
    {
        local_data[lx + 1] = input[(y - 1) * grid_size + x];
    }
    if (ly == local_size_y - 1 && y < grid_size - 1)
    {
        local_data[(local_size_y + 1) * (local_size_x + 2) + lx + 1] = input[(y + 1) * grid_size + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Boundary conditions
    if (x == 0 || x == grid_size - 1 || y == 0 || y == grid_size - 1)
    {
        output[y * grid_size + x] = 1.0f;
        return;
    }

    // Compute average from local memory
    float top = local_data[(ly) * (local_size_x + 2) + lx + 1];
    float bottom = local_data[(ly + 2) * (local_size_x + 2) + lx + 1];
    float left = local_data[(ly + 1) * (local_size_x + 2) + lx];
    float right = local_data[(ly + 1) * (local_size_x + 2) + lx + 2];

    output[y * grid_size + x] = (top + bottom + left + right) * 0.25f;
}
)";

} // namespace kernels
