#pragma once

#include <array>
#include <optional>
#include <sycl/ext/oneapi/experimental/buffer_access_logic.hpp>
#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

struct buffer_access_logic_impl {
  bool Enabled = false;
  std::array<size_t, 3> Step{1, 1, 1};
  std::array<size_t, 3> ChunkLen{1, 1, 1};

private:
  constexpr size_t unit_count_1d(size_t Dim, size_t Phys) const noexcept {
    const size_t Chunk = ChunkLen[Dim];
    const size_t Stride = Step[Dim];

    if (Chunk == 0 || Stride == 0 || Phys < Chunk)
      return 0;

    return (Phys - Chunk) / Stride + 1;
  }

public:
  constexpr size_t map(size_t Dim, size_t Logical) const noexcept {
    return (Logical / ChunkLen[Dim]) * Step[Dim] + (Logical % ChunkLen[Dim]);
  }

  range<3>
  unit_count_from_physical(const range<3> &Phys) const noexcept {
    return range<3>{
        unit_count_1d(0, Phys[0]),
        unit_count_1d(1, Phys[1]),
        unit_count_1d(2, Phys[2]),
    };
  }

  range<3>
  logical_range_from_physical(const range<3> &Phys) const noexcept {
    const auto Count = unit_count_from_physical(Phys);
    return range<3>{
        Count[0] * ChunkLen[0],
        Count[1] * ChunkLen[1],
        Count[2] * ChunkLen[2],
    };
  }

  size_t
  total_unit_count_from_physical(const range<3> &Phys) const noexcept {
    const auto Count = unit_count_from_physical(Phys);
    return Count[0] * Count[1] * Count[2];
  }
};

template <int Dimensions>
inline buffer_access_logic_impl
make_buffer_access_logic_impl(
    const ext::oneapi::experimental::buffer_access_logic<Dimensions> &Logic) {
  buffer_access_logic_impl Impl;
  Impl.Enabled = true;
  for (int I = 0; I < 3; ++I) {
    Impl.Step[I] = 1;
    Impl.ChunkLen[I] = 1;
  }
  for (int I = 0; I < Dimensions; ++I) {
    Impl.Step[I] = Logic.ops[I].step;
    Impl.ChunkLen[I] = Logic.ops[I].chunk_len;
  }
  return Impl;
}

template <int Dimensions>
inline range<3> to_range3(const range<Dimensions> &R) {
  range<3> Out{1, 1, 1};
  for (int I = 0; I < Dimensions; ++I)
    Out[I] = R[I];
  return Out;
}

} // namespace detail
} // namespace _V1
} // namespace sycl