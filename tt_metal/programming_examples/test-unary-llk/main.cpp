// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

class UnaryLLKKernel {
private:
    IDevice* device;
    CommandQueue* cq;
    Program program;
    CoreCoord core;

    // Buffers
    std::shared_ptr<Buffer> input_buffer;
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
    uint32_t input_cb_index;
    uint32_t output_cb_index;

public:
    UnaryLLKKernel(uint32_t num_tiles) :
        num_tiles(num_tiles),
        tile_size(TILE_WIDTH * TILE_HEIGHT * sizeof(bfloat16)),
        elements_per_tile(TILE_WIDTH * TILE_HEIGHT),
        core({0, 0}),
        input_cb_index(CBIndex::c_0),
        output_cb_index(CBIndex::c_16),
        device(nullptr),
        cq(nullptr) {}

    ~UnaryLLKKernel() {
        if (device) {
            CloseDevice(device);
        }
    }

    void Setup() {
        // Initialize device
        constexpr int device_id = 0;
        device = CreateDevice(device_id);

        // Get command queue
        cq = &device->command_queue();

        // Program setup
        program = CreateProgram();

        // Create DRAM buffers
        tt_metal::InterleavedBufferConfig input_buffer_config{
            .device = device,
            .size = num_tiles * elements_per_tile * sizeof(bfloat16),
            .page_size = tile_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        tt_metal::InterleavedBufferConfig output_buffer_config{
            .device = device,
            .size = num_tiles * elements_per_tile * sizeof(bfloat16),
            .page_size = tile_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        input_buffer = CreateBuffer(input_buffer_config);
        output_buffer = CreateBuffer(output_buffer_config);

        // Create circular buffers
        constexpr uint32_t num_cb_tiles = 2;  // Double buffering
        const tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

        CircularBufferConfig input_cb_config =
            CircularBufferConfig(num_cb_tiles * tile_size, {{input_cb_index, cb_data_format}})
                .set_page_size(input_cb_index, tile_size);
        auto input_cb = CreateCircularBuffer(program, core, input_cb_config);

        CircularBufferConfig output_cb_config =
            CircularBufferConfig(num_cb_tiles * tile_size, {{output_cb_index, cb_data_format}})
                .set_page_size(output_cb_index, tile_size);
        auto output_cb = CreateCircularBuffer(program, core, output_cb_config);

        // Kernel compile-time arguments
        std::vector<uint32_t> reader_compile_args = {num_tiles, input_cb_index};
        std::vector<uint32_t> writer_compile_args = {num_tiles, output_cb_index};
        std::vector<uint32_t> compute_compile_args = {num_tiles, input_cb_index, output_cb_index};

        // Create kernels
        reader_kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/test-unary-llk/kernels/dataflow/reader.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_args});

        writer_kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/test-unary-llk/kernels/dataflow/writer.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_args});

        compute_kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/test-unary-llk/kernels/compute/compute.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_args});
    }

    void Execute(std::vector<bfloat16>& input_data, std::vector<bfloat16>& output_data) {
        // Set runtime arguments
        SetRuntimeArgs(program, reader_kernel, core, {input_buffer->address()});
        SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address()});
        // No runtime args needed for compute kernel

        // Upload input data
        EnqueueWriteBuffer(*cq, input_buffer, input_data.data(), false);

        // Execute program
        EnqueueProgram(*cq, program, false);

        // Read output data
        EnqueueReadBuffer(*cq, output_buffer, output_data.data(), true);
    }
};

int main() {
    // Test parameters
    constexpr uint32_t num_tiles = 4;  // Number of tiles to process
    constexpr uint32_t elements_per_tile = TILE_WIDTH * TILE_HEIGHT;

    // Create random input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<bfloat16> input_data(num_tiles * elements_per_tile);
    for (auto& val : input_data) {
        val = bfloat16(dis(gen));
    }

    // Create output data vector
    std::vector<bfloat16> output_data(num_tiles * elements_per_tile);

    // Create and setup kernel
    UnaryLLKKernel kernel(num_tiles);
    kernel.Setup();

    // Execute kernel
    kernel.Execute(input_data, output_data);

    // Print results
    fmt::print("Test Unary LLK Example Results:\n");
    fmt::print("Number of tiles processed: {}\n", num_tiles);
    fmt::print("Elements per tile: {}\n", elements_per_tile);

    // Print first few elements for verification
    fmt::print("\nFirst 10 elements:\n");
    for (int i = 0; i < 10 && i < input_data.size(); i++) {
        fmt::print(
            "Input[{}]: {:.4f} -> Output[{}]: {:.4f}\n", i, input_data[i].to_float(), i, output_data[i].to_float());
    }

    // Verify that output matches input (identity operation)
    bool passed = true;
    for (size_t i = 0; i < input_data.size(); i++) {
        if (std::abs(input_data[i].to_float() - output_data[i].to_float()) > 1e-6) {
            passed = false;
            break;
        }
    }

    if (passed) {
        fmt::print("\nTest PASSED: Identity operation successful!\n");
    } else {
        fmt::print("\nTest FAILED: Output does not match input!\n");
    }

    return passed ? 0 : 1;
}
