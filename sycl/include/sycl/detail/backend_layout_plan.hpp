#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <sycl/detail/buffer_access_logic_impl.hpp>
#include <sycl/id.hpp>
#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

class context_impl;
class buffer_impl;

enum class backend_kind {
  cpu,
  gpu
};

enum class backend_layout_kind {
  unit_first_touch_permutation
};

struct backend_layout_plan {
  backend_kind Backend = backend_kind::cpu;
  backend_layout_kind Layout =
      backend_layout_kind::unit_first_touch_permutation;

  range<3> PhysicalRange{1, 1, 1};
  range<3> UnitGrid{1, 1, 1};

  std::array<size_t, 3> Step{1, 1, 1};
  std::array<size_t, 3> ChunkLen{1, 1, 1};

  // packed linear index -> canonical linear index
  std::vector<size_t> PackedToCanonical;

  // canonical linear index -> packed linear index
  std::vector<size_t> CanonicalToPacked;

  bool IsPermutation = true;
  bool IsSameSize = true;

  size_t physical_size() const noexcept { return PhysicalRange.size(); }
  size_t packed_size() const noexcept { return PackedToCanonical.size(); }
};

inline size_t active_dimensions_from_range3(const range<3> &R) noexcept {
  if (R[2] != 1)
    return 3;
  if (R[1] != 1)
    return 2;
  return 1;
}

inline size_t linearize3(const id<3> &Idx, const range<3> &R) noexcept {
  return (Idx[0] * R[1] + Idx[1]) * R[2] + Idx[2];
}

inline id<3> delinearize3(size_t Linear, const range<3> &R) noexcept {
  id<3> Out{0, 0, 0};
  Out[2] = Linear % R[2];
  Linear /= R[2];
  Out[1] = Linear % R[1];
  Linear /= R[1];
  Out[0] = Linear;
  return Out;
}

inline range<3>
unit_grid_from_logic(const buffer_access_logic_impl &Logic,
                     const range<3> &PhysRange) {
  return Logic.unit_count_from_physical(PhysRange);
}

inline backend_layout_plan
make_cpu_unit_first_touch_plan(const buffer_access_logic_impl &Logic,
                               const range<3> &PhysRange) {
  if (!Logic.Enabled) {
    throw std::invalid_argument(
        "make_cpu_unit_first_touch_plan: logic is disabled");
  }

  backend_layout_plan Plan;
  Plan.Backend = backend_kind::cpu;
  Plan.Layout = backend_layout_kind::unit_first_touch_permutation;
  Plan.PhysicalRange = PhysRange;
  Plan.UnitGrid = unit_grid_from_logic(Logic, PhysRange);
  Plan.Step = Logic.Step;
  Plan.ChunkLen = Logic.ChunkLen;

  const size_t PhysSize = Plan.PhysicalRange.size();
  Plan.PackedToCanonical.reserve(PhysSize);
  Plan.CanonicalToPacked.assign(PhysSize, std::numeric_limits<size_t>::max());

  std::vector<uint8_t> Seen(PhysSize, 0);

  const size_t ActiveDims = active_dimensions_from_range3(Plan.PhysicalRange);

  // 1) 按 unit 顺序遍历
  for (size_t U0 = 0; U0 < Plan.UnitGrid[0]; ++U0) {
    for (size_t U1 = 0; U1 < Plan.UnitGrid[1]; ++U1) {
      for (size_t U2 = 0; U2 < Plan.UnitGrid[2]; ++U2) {
        id<3> Base{
            U0 * Plan.Step[0],
            U1 * Plan.Step[1],
            U2 * Plan.Step[2],
        };

        // 2) unit 内按 row-major 顺序遍历
        for (size_t I0 = 0; I0 < Plan.ChunkLen[0]; ++I0) {
          for (size_t I1 = 0; I1 < Plan.ChunkLen[1]; ++I1) {
            for (size_t I2 = 0; I2 < Plan.ChunkLen[2]; ++I2) {
              id<3> Canon{
                  Base[0] + I0,
                  Base[1] + I1,
                  Base[2] + I2,
              };

              // 对低维，非活动维保持为 0
              if (ActiveDims == 1) {
                Canon[1] = 0;
                Canon[2] = 0;
              } else if (ActiveDims == 2) {
                Canon[2] = 0;
              }

              const size_t CanonLinear =
                  linearize3(Canon, Plan.PhysicalRange);

              if (!Seen[CanonLinear]) {
                Seen[CanonLinear] = 1;
                const size_t PackedLinear = Plan.PackedToCanonical.size();
                Plan.PackedToCanonical.push_back(CanonLinear);
                Plan.CanonicalToPacked[CanonLinear] = PackedLinear;
              }
            }
          }
        }
      }
    }
  }

  // 3) 补齐未被任何 unit 覆盖的元素
  for (size_t CanonLinear = 0; CanonLinear < PhysSize; ++CanonLinear) {
    if (!Seen[CanonLinear]) {
      Seen[CanonLinear] = 1;
      const size_t PackedLinear = Plan.PackedToCanonical.size();
      Plan.PackedToCanonical.push_back(CanonLinear);
      Plan.CanonicalToPacked[CanonLinear] = PackedLinear;
    }
  }

  if (Plan.PackedToCanonical.size() != PhysSize) {
    throw std::runtime_error("unit-first-touch plan is not same-size");
  }

  return Plan;
}

// 可选：保留一个非常薄的 public-wrapper，方便测试或外部过渡使用
template <int Dimensions>
inline backend_layout_plan make_cpu_unit_first_touch_plan(
    const ext::oneapi::experimental::buffer_access_logic<Dimensions> &Logic,
    const range<Dimensions> &PhysRange) {
  return make_cpu_unit_first_touch_plan(
      make_buffer_access_logic_impl(Logic), to_range3(PhysRange));
}

struct backend_layout_desc {
  bool Enabled = false;
  backend_kind Backend = backend_kind::cpu;
  backend_layout_kind Layout =
      backend_layout_kind::unit_first_touch_permutation;

  range<3> PhysicalRange{1, 1, 1};
  size_t TableSize = 0;

  // canonical linear index -> packed linear index
  const size_t *CanonicalToPacked = nullptr;

  size_t canonical_to_packed(size_t Canonical) const noexcept {
    if (!Enabled || !CanonicalToPacked || Canonical >= TableSize)
      return Canonical;
    return CanonicalToPacked[Canonical];
  }
};

struct backend_layout_cache_key {
  backend_kind Backend = backend_kind::cpu;
  backend_layout_kind Layout =
      backend_layout_kind::unit_first_touch_permutation;

  bool operator==(const backend_layout_cache_key &Rhs) const noexcept {
    return Backend == Rhs.Backend && Layout == Rhs.Layout;
  }
};

struct backend_layout_cache_key_hash {
  size_t operator()(const backend_layout_cache_key &K) const noexcept {
    return (static_cast<size_t>(K.Backend) << 8) ^
           static_cast<size_t>(K.Layout);
  }
};

struct device_layout_mapping {
  backend_layout_desc Desc{};

  // host-owned canonical->packed table
  std::shared_ptr<std::vector<size_t>> HostCanonicalToPacked;

  // CPU backend can alias host table directly.
  // GPU backend can replace this with uploaded/USM pointer later.
  const size_t *DeviceCanonicalToPacked = nullptr;

  // reserved hook for future GPU upload owner/allocation
  std::shared_ptr<void> DeviceOwner;
};

inline size_t elem_count_or_throw(size_t Bytes, size_t ElemBytes) {
  if (ElemBytes == 0 || Bytes % ElemBytes != 0) {
    throw std::runtime_error("invalid element size for backend layout packing");
  }
  return Bytes / ElemBytes;
}

// canonical host bytes -> packed host bytes
inline void pack_bytes_by_plan(const unsigned char *CanonicalSrc,
                               unsigned char *PackedDst,
                               size_t TotalBytes,
                               size_t ElemBytes,
                               const backend_layout_plan &Plan) {
  const size_t ElemCount = elem_count_or_throw(TotalBytes, ElemBytes);

  if (ElemCount != Plan.packed_size()) {
    throw std::runtime_error(
        "pack_bytes_by_plan: size mismatch, elem_count=" +
        std::to_string(ElemCount) +
        ", packed_size=" + std::to_string(Plan.packed_size()) +
        ", physical_size=" + std::to_string(Plan.physical_size()));
  }

  for (size_t P = 0; P < ElemCount; ++P) {
    const size_t Canon = Plan.PackedToCanonical[P];
    std::memcpy(PackedDst + P * ElemBytes,
                CanonicalSrc + Canon * ElemBytes,
                ElemBytes);
  }
}

// packed host bytes -> canonical host bytes
inline void unpack_bytes_by_plan(const unsigned char *PackedSrc,
                                 unsigned char *CanonicalDst,
                                 size_t TotalBytes,
                                 size_t ElemBytes,
                                 const backend_layout_plan &Plan) {
  const size_t ElemCount = elem_count_or_throw(TotalBytes, ElemBytes);

  if (ElemCount != Plan.packed_size()) {
    throw std::runtime_error("unpack_bytes_by_plan: size mismatch");
  }

  for (size_t P = 0; P < ElemCount; ++P) {
    const size_t Canon = Plan.PackedToCanonical[P];
    std::memcpy(CanonicalDst + Canon * ElemBytes,
                PackedSrc + P * ElemBytes,
                ElemBytes);
  }
}

inline backend_layout_kind
select_layout_kind_for_backend(backend_kind BK) {
  switch (BK) {
  case backend_kind::cpu:
    return backend_layout_kind::unit_first_touch_permutation;
  case backend_kind::gpu:
    // first version: keep same policy, split later
    return backend_layout_kind::unit_first_touch_permutation;
  }
  return backend_layout_kind::unit_first_touch_permutation;
}

__SYCL_EXPORT const device_layout_mapping *get_or_create_device_layout_mapping_for_accessor(
    const std::shared_ptr<buffer_impl> &Impl, context_impl *Ctx,
    backend_kind BK, backend_layout_kind LK);

} // namespace detail
} // namespace _V1
} // namespace sycl