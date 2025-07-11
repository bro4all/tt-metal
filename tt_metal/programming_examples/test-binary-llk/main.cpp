// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <fmt/core.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

struct DFormat_Float32 {
    using Host = float;
    static constexpr tt::DataFormat Device = tt::DataFormat::Float32;
};

struct DFormat_Bfloat16 {
    using Host = bfloat16;
    static constexpr tt::DataFormat Device = tt::DataFormat::Float16_b;
};

template <typename DFormat>
class UnaryLLKKernel {
private:
    Program program;
    CoreCoord core;

    // Buffers
    std::shared_ptr<Buffer> input0_buffer;
    std::shared_ptr<Buffer> input1_buffer;
    std::shared_ptr<Buffer> output_buffer;

    // Kernels
    KernelHandle reader_kernel;
    KernelHandle writer_kernel;
    KernelHandle compute_kernel;

    // Parameters
    uint32_t num_tiles;
    uint32_t tile_size;
    uint32_t elements_per_tile;

    // Circular buffer indices
    uint32_t input0_cb_index;
    uint32_t input1_cb_index;
    uint32_t output_cb_index;

public:
    UnaryLLKKernel(uint32_t num_tiles) :
        num_tiles(num_tiles),
        tile_size(TILE_WIDTH * TILE_HEIGHT * sizeof(typename DFormat::Host)),
        elements_per_tile(TILE_WIDTH * TILE_HEIGHT),
        core({0, 0}),
        input0_cb_index(CBIndex::c_0),
        input1_cb_index(CBIndex::c_1),
        output_cb_index(CBIndex::c_2) {}

    ~UnaryLLKKernel() {}

    void Setup(IDevice* device) {
        // Initialize device

        // Get command queue
        CommandQueue* cq = &device->command_queue();

        // Program setup
        program = CreateProgram();

        // Create DRAM buffers
        tt_metal::InterleavedBufferConfig input_buffer_config{
            .device = device,
            .size = num_tiles * elements_per_tile * sizeof(typename DFormat::Host),
            .page_size = tile_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        tt_metal::InterleavedBufferConfig output_buffer_config{
            .device = device,
            .size = num_tiles * elements_per_tile * sizeof(typename DFormat::Host),
            .page_size = tile_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        input0_buffer = CreateBuffer(input_buffer_config);
        input1_buffer = CreateBuffer(input_buffer_config);
        output_buffer = CreateBuffer(output_buffer_config);

        // Create circular buffers
        constexpr uint32_t num_cb_tiles = 2;  // Double buffering
        const tt::DataFormat cb_data_format = DFormat::Device;

        CircularBufferConfig input0_cb_config =
            CircularBufferConfig(num_cb_tiles * tile_size, {{input0_cb_index, cb_data_format}})
                .set_page_size(input0_cb_index, tile_size);
        auto input0_cb = CreateCircularBuffer(program, core, input0_cb_config);

        CircularBufferConfig input1_cb_config =
            CircularBufferConfig(num_cb_tiles * tile_size, {{input1_cb_index, cb_data_format}})
                .set_page_size(input1_cb_index, tile_size);
        auto input1_cb = CreateCircularBuffer(program, core, input1_cb_config);

        CircularBufferConfig output_cb_config =
            CircularBufferConfig(num_cb_tiles * tile_size, {{output_cb_index, cb_data_format}})
                .set_page_size(output_cb_index, tile_size);
        auto output_cb = CreateCircularBuffer(program, core, output_cb_config);

        // Kernel compile-time arguments
        std::vector<uint32_t> reader_compile_args = {num_tiles, input0_cb_index, input1_cb_index};
        std::vector<uint32_t> writer_compile_args = {num_tiles, output_cb_index};
        std::vector<uint32_t> compute_compile_args = {num_tiles, input0_cb_index, input1_cb_index, output_cb_index};

        // Create kernels
        reader_kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/test-binary-llk/kernels/dataflow/reader.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_args});

        writer_kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/test-binary-llk/kernels/dataflow/writer.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_args});

        compute_kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/test-binary-llk/kernels/compute/compute.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_args});
    }

    void Execute(
        IDevice* device,
        std::vector<typename DFormat::Host>& input0_data,
        std::vector<typename DFormat::Host> input1_data,
        std::vector<typename DFormat::Host>& output_data) {
        // Get command queue
        CommandQueue* cq = &device->command_queue();

        // Set runtime arguments
        SetRuntimeArgs(program, reader_kernel, core, {input0_buffer->address(), input1_buffer->address()});
        SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address()});
        // No runtime args needed for compute kernel

        // Upload input data
        EnqueueWriteBuffer(*cq, input0_buffer, input0_data.data(), false);
        EnqueueWriteBuffer(*cq, input1_buffer, input1_data.data(), false);

        // Execute program

        EnqueueProgram(*cq, program, false);
        Finish(*cq);

        // Read output data
        EnqueueReadBuffer(*cq, output_buffer, output_data.data(), true);
    }
};

int main() {
    // Test parameters
    constexpr uint32_t num_tiles = 1;  // Number of tiles to process
    constexpr uint32_t elements_per_tile = TILE_WIDTH * TILE_HEIGHT;

    // Create random input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<float> input0_data_f32(num_tiles * elements_per_tile);
    std::vector<float> input1_data_f32(num_tiles * elements_per_tile);

    std::vector<bfloat16> input0_data_bf16(num_tiles * elements_per_tile);
    std::vector<bfloat16> input1_data_bf16(num_tiles * elements_per_tile);

    std::vector<float> data0 = {9.f, 100000.f, 5.f};
    std::vector<float> data1 = {2.f, 1.7984f, 3.f};

    for (size_t i = 0; i < num_tiles * elements_per_tile; i++) {
        float val0_f32 = 9.f;
        float val1_f32 = 2.f;

        if (i < data0.size()) {
            val0_f32 = data0[i];
        }

        if (i < data1.size()) {
            val1_f32 = data1[i];
        }

        input0_data_bf16[i] = bfloat16(val0_f32);
        input1_data_bf16[i] = bfloat16(val1_f32);

        input0_data_f32[i] = val0_f32;
        input1_data_f32[i] = val1_f32;
    }

    // Create output data vector
    std::vector<bfloat16> output_data_bf16(num_tiles * elements_per_tile);
    std::vector<float> output_data_f32(num_tiles * elements_per_tile);

    constexpr int device_id = 0;

    // Create and setup kernel

    // Execute kernel
    {
        UnaryLLKKernel<DFormat_Float32> kernel_f32(num_tiles);
        IDevice* device = CreateDevice(device_id);
        kernel_f32.Setup(device);
        kernel_f32.Execute(device, input0_data_f32, input1_data_f32, output_data_f32);
        CloseDevice(device);
    }

    {
        UnaryLLKKernel<DFormat_Bfloat16> kernel_bf16(num_tiles);

        IDevice* device = CreateDevice(device_id);
        kernel_bf16.Setup(device);
        kernel_bf16.Execute(device, input0_data_bf16, input1_data_bf16, output_data_bf16);
        CloseDevice(device);
    }

    // Print results
    fmt::print("Test Binary LLK Example Results:\n");
    fmt::print("Number of tiles processed: {}\n", num_tiles);
    fmt::print("Elements per tile: {}\n", elements_per_tile);

    size_t elements_to_print = data0.size();

    // Print first few elements for verification
    fmt::print("\nFirst {} elements:\n", elements_to_print);
    for (int i = 0; i < elements_to_print && i < input0_data_f32.size(); i++) {
        fmt::print(
            "[float32] Input[{}]: {:.6f}, {:6f} -> Output[{}]: {:.6f}\n",
            i,
            input0_data_f32[i],
            input1_data_f32[i],
            i,
            output_data_f32[i]);
    }
    fmt::print("-----");

    for (int i = 0; i < elements_to_print && i < input0_data_bf16.size(); i++) {
        fmt::print(
            "[bfloat16] Input[{}]: {:.6f}, {:6f} -> Output[{}]: {:.6f}\n",
            i,
            input0_data_bf16[i].to_float(),
            input1_data_bf16[i].to_float(),
            i,
            output_data_bf16[i].to_float());
    }
    fmt::print("-----");

    for (int i = 0; i < elements_to_print && i < input0_data_f32.size(); i++) {
        float truth = powf(input0_data_f32[i], input1_data_f32[i]);
        fmt::print(
            "[float32 reference] Input[{}]: {:.6f}, {:6f} -> Output[{}]: {:.6f}, bf16 = {:6f}\n",
            i,
            input0_data_f32[i],
            input1_data_f32[i],
            i,
            truth,
            bfloat16(truth).to_float());
    }

    return 0;
}
