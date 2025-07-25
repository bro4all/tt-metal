// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// This header contains tile conversion functions used in tests on the host.
//

#pragma once

#include <tt_stl/span.hpp>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>
#include <iosfwd>

enum class TensorLayoutType {
    LIN_ROW_MAJOR = 0,   // standard element-wise row-major
    TILED_SWIZZLED = 1,  // row-major of tiles, each tile is row-major-swizzled
    TILED_NFACES = 2,    // row-major of tiles, each tile is N (N = 1, 2, or 4) faces, each face is
                         // row-major, faces are swizzled
};
std::ostream& operator<<(std::ostream& os, TensorLayoutType layout);

using PhysicalSize = std::array<uint32_t, 2>;

struct TensAddr {
    std::vector<std::uint32_t> sh;

    TensAddr(const std::vector<std::uint32_t>& shape);
    std::uint32_t numel() const;
    int offs(int n, int c, int h, int w);
};

std::uint32_t round_up_to_mul16(std::uint32_t val);

std::uint32_t round_up_to_mul32(std::uint32_t val);

std::uint32_t round_up_to_tile(int val, int tile_val);

template <class T>
std::vector<T> convert_layout_tile_swizzled_to_tile_nfaces(
    tt::stl::Span<const T> data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

template <class T>
std::vector<T> convert_layout_tile_nfaces_to_tile_swizzled(
    tt::stl::Span<const T> data,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

template <typename T>
std::vector<T> convert_layout(
    tt::stl::Span<const T> data,
    const PhysicalSize& shape,
    TensorLayoutType inL,
    TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_within_face = false,
    bool transpose_of_faces = false);

template <typename T>
std::vector<T> convert_layout(
    tt::stl::Span<const T> data,
    tt::stl::Span<const uint32_t> shape,
    TensorLayoutType inL,
    TensorLayoutType outL,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_within_face = false,
    bool transpose_of_faces = false);

// Those are specific cases that convert_layout can do, but left for compatibility with existing codebase
// Converts from/to row major
template <typename T>
std::vector<T> tilize_swizzled(const std::vector<T>& input, uint32_t m, uint32_t n);

template <typename T>
std::vector<T> untilize_swizzled(const std::vector<T>& input, uint32_t m, uint32_t n);

template <typename T>
std::vector<T> tilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n);

template <typename T>
std::vector<T> untilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n);

// Additional convert_layout function declarations
template <typename T>
std::vector<T> convert_layout_row_major_to_tile_swizzled(
    tt::stl::Span<const T> in_row_major, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape);

template <typename T>
std::vector<T> convert_layout_tile_swizzled_to_row_major(
    tt::stl::Span<const T> in_tile_swizzled, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape);

template <typename T>
std::vector<T> convert_layout_row_major_to_tile_nfaces(
    tt::stl::Span<const T> in_row_major,
    const PhysicalSize& shape,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    const bool transpose_face_order);

template <typename T>
std::vector<T> convert_layout_tile_nfaces_to_row_major(
    tt::stl::Span<const T> in_nfaces,
    const PhysicalSize& shape,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    bool transpose_face_order);

// Bool specializations for functions that handle std::vector<bool>
// These handle std::vector<bool> which doesn't support .data() or direct memory operations
template <>
std::vector<bool> tilize_swizzled<bool>(const std::vector<bool>& input, uint32_t m, uint32_t n);

template <>
std::vector<bool> untilize_swizzled<bool>(const std::vector<bool>& input, uint32_t m, uint32_t n);

template <>
std::vector<bool> tilize_nfaces<bool>(const std::vector<bool>& input, uint32_t m, uint32_t n);

template <>
std::vector<bool> untilize_nfaces<bool>(const std::vector<bool>& input, uint32_t m, uint32_t n);

// Bool specializations for convert_layout functions
template <>
std::vector<bool> convert_layout_row_major_to_tile_swizzled<bool>(
    tt::stl::Span<const bool> in_row_major, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape);

template <>
std::vector<bool> convert_layout_tile_swizzled_to_row_major<bool>(
    tt::stl::Span<const bool> in_tile_swizzled, const PhysicalSize& shape, std::optional<PhysicalSize> tile_shape);

template <>
std::vector<bool> convert_layout_tile_swizzled_to_tile_nfaces<bool>(
    tt::stl::Span<const bool> in_tile_swizzled,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    bool transpose_face_order);

template <>
std::vector<bool> convert_layout_tile_nfaces_to_tile_swizzled<bool>(
    tt::stl::Span<const bool> in_tile_nfaces,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    bool transpose_face_order);

template <>
std::vector<bool> convert_layout_row_major_to_tile_nfaces<bool>(
    tt::stl::Span<const bool> in_row_major,
    const PhysicalSize& shape,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    const bool transpose_face_order);

template <>
std::vector<bool> convert_layout_tile_nfaces_to_row_major<bool>(
    tt::stl::Span<const bool> in_nfaces,
    const PhysicalSize& shape,
    std::optional<PhysicalSize> tile_shape,
    std::optional<PhysicalSize> face_shape,
    const bool transpose_face,
    bool transpose_face_order);
