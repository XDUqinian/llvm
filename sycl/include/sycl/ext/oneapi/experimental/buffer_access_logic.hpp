#pragma once

#include <array>
#include <cstddef>
#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

class dim_operator {
public:
  size_t step = 1;
  size_t chunk_len = 1;

  constexpr dim_operator() = default;

  constexpr dim_operator(size_t Step, size_t ChunkLen)
      : step(Step), chunk_len(ChunkLen) {}

  constexpr bool valid() const noexcept {
    return step > 0 && chunk_len > 0;
  }

  constexpr size_t map(size_t logical) const noexcept {
    return (logical / chunk_len) * step + (logical % chunk_len);
  }

  constexpr size_t logical_extent(size_t physical_extent) const noexcept {
    if (step == 0 || chunk_len == 0)
      return 0;

    if (physical_extent < chunk_len)
      return 0;

    size_t num_chunks = (physical_extent - chunk_len) / step + 1;
    return num_chunks * chunk_len;
  }
};

template <int Dimensions>
class buffer_access_logic {
public:
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "buffer_access_logic only supports 1D, 2D, and 3D");

  constexpr buffer_access_logic() = default;

  constexpr buffer_access_logic(
      const std::array<dim_operator, Dimensions> &Ops) noexcept
      : ops(Ops) {}

  std::array<dim_operator, Dimensions> ops{};

  constexpr const dim_operator &operator[](size_t i) const noexcept {
    return ops[i];
  }

  constexpr dim_operator &operator[](size_t i) noexcept {
    return ops[i];
  }

  constexpr bool valid() const noexcept {
    for (int i = 0; i < Dimensions; ++i) {
      if (!ops[i].valid())
        return false;
    }
    return true;
  }

  constexpr sycl::range<Dimensions>
  logical_range_from_physical(const sycl::range<Dimensions> &phys) const
      noexcept {
    sycl::range<Dimensions> out = phys;
    for (int i = 0; i < Dimensions; ++i)
      out[i] = ops[i].logical_extent(phys[i]);
    return out;
  }
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl