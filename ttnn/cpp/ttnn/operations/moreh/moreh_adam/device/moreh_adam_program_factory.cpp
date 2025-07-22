// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_adam_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "adam_cb_config.hpp"

union Scalar {
    float f;
    uint32_t u;
};

namespace ttnn::operations::moreh::moreh_adam {
MorehAdamOperation::ProgramFactory::cached_program_t MorehAdamOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    namespace cb = cb_adam;
    auto& param_in = tensor_args.param_in;
    auto& grad = tensor_args.grad;
    auto& exp_avg_in = tensor_args.exp_avg_in;
    auto& exp_avg_sq_in = tensor_args.exp_avg_sq_in;

    auto& output_tensors = output_tensor;

    auto max_exp_avg_sq_in = tensor_args.max_exp_avg_sq_in;

    auto& param_out = output_tensors.at(0).value();
    auto& exp_avg_out = output_tensors.at(1).value();
    auto& exp_avg_sq_out = output_tensors.at(2).value();
    auto max_exp_avg_sq_out = output_tensors.at(3);

    auto lr = operation_attributes.lr;
    auto beta1 = operation_attributes.beta1;
    auto beta2 = operation_attributes.beta2;
    auto eps = operation_attributes.eps;
    auto weight_decay = operation_attributes.weight_decay;
    auto step = operation_attributes.step;
    auto amsgrad = operation_attributes.amsgrad;

    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    uint32_t num_tiles = param_in.physical_volume() / tt::constants::TILE_HW;

    Scalar f2u_lr, f2u_beta1, f2u_beta2, f2u_eps, f2u_weight_decay, f2u_biascorr1, f2u_biascorr2;
    f2u_lr.f = lr;
    f2u_beta1.f = beta1;
    f2u_beta2.f = beta2;
    f2u_eps.f = eps;
    f2u_weight_decay.f = weight_decay;
    f2u_biascorr1.f = 1.0f - powf(beta1, step);
    f2u_biascorr2.f = 1.0f - powf(beta2, step);

    Program program{};

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::IDevice* device = param_in.device();
    auto grid = device->compute_with_storage_grid_size();

    uint32_t num_units[2];
    CoreRangeSet all_cores, core_groups[2];
    uint32_t num_cores;
    std::tie(num_cores, all_cores, core_groups[0], core_groups[1], num_units[0], num_units[1]) =
        split_work_to_cores(grid, num_tiles);

    auto arch = param_in.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(param_in.dtype());
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {cb::param_in, 2},
            {cb::grad, 2},
            {cb::exp_avg_in, 2},
            {cb::exp_avg_sq_in, 2},
            {cb::max_exp_avg_sq_in, amsgrad ? 2 : 0},
            {cb::grad_i, 1},
            {cb::exp_avg_i, 1},
            {cb::exp_avg_sq_i, 1},
            {cb::max_exp_avg_sq_i, amsgrad ? 1 : 0},
            {cb::param_out, 2},
            {cb::exp_avg_out, 2},
            {cb::exp_avg_sq_out, 2},
            {cb::max_exp_avg_sq_out, amsgrad ? 2 : 0},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(param_in)),
        static_cast<uint32_t>(is_dram(grad)),
        static_cast<uint32_t>(is_dram(exp_avg_in)),
        static_cast<uint32_t>(is_dram(exp_avg_sq_in)),
        static_cast<uint32_t>(is_dram(max_exp_avg_sq_in))};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(is_dram(param_out)),
        static_cast<uint32_t>(is_dram(exp_avg_out)),
        static_cast<uint32_t>(is_dram(exp_avg_sq_out)),
        static_cast<uint32_t>(max_exp_avg_sq_out.has_value() ? is_dram(max_exp_avg_sq_out.value()) : false)};

    const std::vector<uint32_t> compute_ctas{
        amsgrad,
    };

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/"
        "reader_moreh_adam.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/"
        "writer_moreh_adam.cpp";

    std::map<std::string, std::string> data_movement_defines{};
    if (amsgrad) {
        data_movement_defines["AMSGRAD"] = "1";
    }
    if (fp32_dest_acc_en) {
        data_movement_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, data_movement_defines);
    const auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, data_movement_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    if (amsgrad) {
        compute_defines["AMSGRAD"] = "1";
    }

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/"
        "moreh_adam.cpp";

    auto compute_kernel_id = CreateKernel(
        program,
        compute_kernel_file,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_ctas,
            .defines = compute_defines,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto param_in_addr = param_in.buffer()->address();
    const auto grad_addr = grad.buffer()->address();
    const auto exp_avg_in_addr = exp_avg_in.buffer()->address();
    const auto exp_avg_sq_in_addr = exp_avg_sq_in.buffer()->address();
    const auto max_exp_avg_sq_in_addr =
        max_exp_avg_sq_in.has_value() ? max_exp_avg_sq_in.value().buffer()->address() : 0;

    const auto param_out_addr = param_out.buffer()->address();
    const auto exp_avg_out_addr = exp_avg_out.buffer()->address();
    const auto exp_avg_sq_out_addr = exp_avg_sq_out.buffer()->address();
    const auto max_exp_avg_sq_out_addr = max_exp_avg_sq_out.has_value() ? max_exp_avg_sq_out->buffer()->address() : 0;

    tt::tt_metal::SetCommonRuntimeArgs(
        program,
        compute_kernel_id,
        {num_units[0],
         num_units[1],
         core_groups[0].num_cores(),
         core_groups[1].num_cores(),
         grid.y,
         step,
         f2u_lr.u,
         f2u_beta1.u,
         f2u_beta2.u,
         f2u_eps.u,
         f2u_weight_decay.u,
         f2u_biascorr1.u,
         f2u_biascorr2.u});

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / grid.y, i % grid.y};

        uint32_t num_tiles_per_core = 0;
        if (core_groups[0].contains(core)) {
            num_tiles_per_core = num_units[0];
        } else if (core_groups[1].contains(core)) {
            num_tiles_per_core = num_units[1];
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        const std::vector<uint32_t> reader_runtime_args{
            param_in_addr,
            grad_addr,
            exp_avg_in_addr,
            exp_avg_sq_in_addr,
            max_exp_avg_sq_in_addr,
            f2u_lr.u,
            f2u_beta1.u,
            f2u_beta2.u,
            f2u_eps.u,
            f2u_weight_decay.u,
            step,
            static_cast<uint32_t>(amsgrad),
            num_tiles_per_core,
            tile_offset};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{
            param_out_addr,
            exp_avg_out_addr,
            exp_avg_sq_out_addr,
            max_exp_avg_sq_out_addr,
            num_tiles_per_core,
            tile_offset};
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        tile_offset += num_tiles_per_core;
    }

    return {
        std::move(program),
        {reader_kernel_id,
         writer_kernel_id,
         compute_kernel_id,
         compute_kernel_id,
         core_groups[0],
         core_groups[1],
         num_cores,
         grid.y}};
}

void MorehAdamOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& compute_kernel_1_id = cached_program.shared_variables.compute_kernel_group1_id;
    auto& compute_kernel_2_id = cached_program.shared_variables.compute_kernel_group2_id;

    auto param_in_buffer = tensor_args.param_in.buffer();
    auto grad_buffer = tensor_args.grad.buffer();
    auto exp_avg_in_buffer = tensor_args.exp_avg_in.buffer();
    auto exp_avg_sq_in_buffer = tensor_args.exp_avg_sq_in.buffer();
    auto max_exp_avg_sq_in_buffer =
        tensor_args.max_exp_avg_sq_in.has_value() ? tensor_args.max_exp_avg_sq_in->buffer() : nullptr;

    auto param_out_buffer = tensor_return_value.at(0)->buffer();
    auto exp_avg_out_buffer = tensor_return_value.at(1)->buffer();
    auto exp_avg_sq_out_buffer = tensor_return_value.at(2)->buffer();
    auto max_exp_avg_sq_out_buffer = operation_attributes.amsgrad ? tensor_return_value.at(3)->buffer() : nullptr;

    auto& core_group_1 = cached_program.shared_variables.core_group_1;
    auto& core_group_2 = cached_program.shared_variables.core_group_2;

    auto num_cores = cached_program.shared_variables.num_cores;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    union {
        float f;
        uint32_t u;
    } f2u_lr;

    f2u_lr.f = operation_attributes.lr;

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = param_in_buffer->address();
            runtime_args[1] = grad_buffer->address();
            runtime_args[2] = exp_avg_in_buffer->address();
            runtime_args[3] = exp_avg_sq_in_buffer->address();
            if (max_exp_avg_sq_in_buffer != nullptr) {
                runtime_args[4] = max_exp_avg_sq_in_buffer->address();
            }
            runtime_args[5] = f2u_lr.u;
            runtime_args[10] = operation_attributes.step;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = param_out_buffer->address();
            runtime_args[1] = exp_avg_out_buffer->address();
            runtime_args[2] = exp_avg_sq_out_buffer->address();
            if (max_exp_avg_sq_out_buffer != nullptr) {
                runtime_args[3] = max_exp_avg_sq_out_buffer->address();
            }
        }
        {
            if (core_group_1.contains(core)) {
                tt::tt_metal::SetRuntimeArgs(program, compute_kernel_1_id, core, {operation_attributes.step});
            } else if (core_group_2.contains(core)) {
                tt::tt_metal::SetRuntimeArgs(program, compute_kernel_2_id, core, {operation_attributes.step});
            } else {
                TT_THROW("Core not in specified core ranges.");
            }
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_adam
