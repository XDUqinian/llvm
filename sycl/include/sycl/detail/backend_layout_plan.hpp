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
#include <numeric>
#include <algorithm>

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
  unit_first_touch_permutation,
  blocked_first_touch_permutation,
  frequency_aware_permutation,
  same_offset_permutation,
  warp_same_offset_permutation
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

inline id<3> unit_coord_from_linear(size_t Linear, const range<3> &UnitGrid) {
  return delinearize3(Linear, UnitGrid);
}

inline void append_first_touch(size_t CanonLinear,
                               std::vector<uint8_t> &Seen,
                               backend_layout_plan &Plan) {
  if (!Seen[CanonLinear]) {
    Seen[CanonLinear] = 1;
    const size_t PackedLinear = Plan.PackedToCanonical.size();
    Plan.PackedToCanonical.push_back(CanonLinear);
    Plan.CanonicalToPacked[CanonLinear] = PackedLinear;
  }
}

template <class Fn>
inline void for_each_elem_in_unit_row_major(const backend_layout_plan &Plan,
                                            const id<3> &Base,
                                            Fn &&Visit) {
  const size_t ActiveDims = active_dimensions_from_range3(Plan.PhysicalRange);
  for (size_t I0 = 0; I0 < Plan.ChunkLen[0]; ++I0) {
    for (size_t I1 = 0; I1 < Plan.ChunkLen[1]; ++I1) {
      for (size_t I2 = 0; I2 < Plan.ChunkLen[2]; ++I2) {
        id<3> Canon{Base[0] + I0, Base[1] + I1, Base[2] + I2};
        if (ActiveDims == 1) {
          Canon[1] = 0;
          Canon[2] = 0;
        } else if (ActiveDims == 2) {
          Canon[2] = 0;
        }
        Visit(linearize3(Canon, Plan.PhysicalRange));
      }
    }
  }
}

// 布局优化策略实现
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

inline backend_layout_plan
make_cpu_blocked_first_touch_plan(const buffer_access_logic_impl &Logic,
                                  const range<3> &PhysRange,
                                  const range<3> &BlockUnits = range<3>{4, 4, 1}) {
  if (!Logic.Enabled) {
    throw std::invalid_argument(
        "make_cpu_blocked_first_touch_plan: logic is disabled");
  }

  backend_layout_plan Plan;
  Plan.Backend = backend_kind::cpu;
  Plan.Layout = backend_layout_kind::blocked_first_touch_permutation;
  Plan.PhysicalRange = PhysRange;
  Plan.UnitGrid = unit_grid_from_logic(Logic, PhysRange);
  Plan.Step = Logic.Step;
  Plan.ChunkLen = Logic.ChunkLen;

  const size_t PhysSize = Plan.PhysicalRange.size();
  Plan.PackedToCanonical.reserve(PhysSize);
  Plan.CanonicalToPacked.assign(PhysSize, std::numeric_limits<size_t>::max());
  std::vector<uint8_t> Seen(PhysSize, 0);

  const range<3> BlockGrid{
      (Plan.UnitGrid[0] + BlockUnits[0] - 1) / BlockUnits[0],
      (Plan.UnitGrid[1] + BlockUnits[1] - 1) / BlockUnits[1],
      (Plan.UnitGrid[2] + BlockUnits[2] - 1) / BlockUnits[2]};

  for (size_t B0 = 0; B0 < BlockGrid[0]; ++B0) {
    for (size_t B1 = 0; B1 < BlockGrid[1]; ++B1) {
      for (size_t B2 = 0; B2 < BlockGrid[2]; ++B2) {
        for (size_t U0 = B0 * BlockUnits[0];
             U0 < std::min((B0 + 1) * BlockUnits[0], Plan.UnitGrid[0]); ++U0) {
          for (size_t U1 = B1 * BlockUnits[1];
               U1 < std::min((B1 + 1) * BlockUnits[1], Plan.UnitGrid[1]); ++U1) {
            for (size_t U2 = B2 * BlockUnits[2];
                 U2 < std::min((B2 + 1) * BlockUnits[2], Plan.UnitGrid[2]); ++U2) {
              id<3> Base{
                  U0 * Plan.Step[0],
                  U1 * Plan.Step[1],
                  U2 * Plan.Step[2],
              };
              for_each_elem_in_unit_row_major(Plan, Base, [&](size_t CanonLinear) {
                append_first_touch(CanonLinear, Seen, Plan);
              });
            }
          }
        }
      }
    }
  }

  for (size_t CanonLinear = 0; CanonLinear < PhysSize; ++CanonLinear) {
    append_first_touch(CanonLinear, Seen, Plan);
  }

  if (Plan.PackedToCanonical.size() != PhysSize) {
    throw std::runtime_error("blocked-first-touch plan is not same-size");
  }

  return Plan;
}

inline backend_layout_plan
make_cpu_frequency_aware_plan(const buffer_access_logic_impl &Logic,
                              const range<3> &PhysRange) {
  if (!Logic.Enabled) {
    throw std::invalid_argument(
        "make_cpu_frequency_aware_plan: logic is disabled");
  }

  backend_layout_plan Plan;
  Plan.Backend = backend_kind::cpu;
  Plan.Layout = backend_layout_kind::frequency_aware_permutation;
  Plan.PhysicalRange = PhysRange;
  Plan.UnitGrid = unit_grid_from_logic(Logic, PhysRange);
  Plan.Step = Logic.Step;
  Plan.ChunkLen = Logic.ChunkLen;

  const size_t PhysSize = Plan.PhysicalRange.size();
  Plan.PackedToCanonical.reserve(PhysSize);
  Plan.CanonicalToPacked.assign(PhysSize, std::numeric_limits<size_t>::max());

  std::vector<size_t> Freq(PhysSize, 0);
  std::vector<size_t> FirstTouchRank(PhysSize, std::numeric_limits<size_t>::max());
  size_t NextRank = 0;

  // 1) 统计频次，并记录 unit-first-touch 顺序
  for (size_t U0 = 0; U0 < Plan.UnitGrid[0]; ++U0) {
    for (size_t U1 = 0; U1 < Plan.UnitGrid[1]; ++U1) {
      for (size_t U2 = 0; U2 < Plan.UnitGrid[2]; ++U2) {
        id<3> Base{
            U0 * Plan.Step[0],
            U1 * Plan.Step[1],
            U2 * Plan.Step[2],
        };
        for_each_elem_in_unit_row_major(Plan, Base, [&](size_t CanonLinear) {
          ++Freq[CanonLinear];
          if (FirstTouchRank[CanonLinear] == std::numeric_limits<size_t>::max()) {
            FirstTouchRank[CanonLinear] = NextRank++;
          }
        });
      }
    }
  }

  // 没被任何 unit 覆盖的元素，保持在后面，次序用 canonical 顺序补齐
  for (size_t CanonLinear = 0; CanonLinear < PhysSize; ++CanonLinear) {
    if (FirstTouchRank[CanonLinear] == std::numeric_limits<size_t>::max()) {
      FirstTouchRank[CanonLinear] = NextRank++;
    }
  }

  // 2) 排序生成 permutation
  std::vector<size_t> Canonicals(PhysSize);
  std::iota(Canonicals.begin(), Canonicals.end(), 0);

  std::stable_sort(Canonicals.begin(), Canonicals.end(),
                   [&](size_t A, size_t B) {
                     if (Freq[A] != Freq[B])
                       return Freq[A] > Freq[B];
                     if (FirstTouchRank[A] != FirstTouchRank[B])
                       return FirstTouchRank[A] < FirstTouchRank[B];
                     return A < B;
                   });

  // 3) 写回 plan
  for (size_t PackedLinear = 0; PackedLinear < PhysSize; ++PackedLinear) {
    const size_t CanonLinear = Canonicals[PackedLinear];
    Plan.PackedToCanonical.push_back(CanonLinear);
    Plan.CanonicalToPacked[CanonLinear] = PackedLinear;
  }

  if (Plan.PackedToCanonical.size() != PhysSize) {
    throw std::runtime_error("frequency-aware plan is not same-size");
  }

  return Plan;
}

inline backend_layout_plan
make_gpu_same_offset_plan(const buffer_access_logic_impl &Logic,
                          const range<3> &PhysRange) {
  if (!Logic.Enabled) {
    throw std::invalid_argument(
        "make_gpu_same_offset_plan: logic is disabled");
  }

  backend_layout_plan Plan;
  Plan.Backend = backend_kind::gpu;
  Plan.Layout = backend_layout_kind::same_offset_permutation;
  Plan.PhysicalRange = PhysRange;
  Plan.UnitGrid = unit_grid_from_logic(Logic, PhysRange);
  Plan.Step = Logic.Step;
  Plan.ChunkLen = Logic.ChunkLen;

  const size_t PhysSize = Plan.PhysicalRange.size();
  Plan.PackedToCanonical.reserve(PhysSize);
  Plan.CanonicalToPacked.assign(PhysSize, std::numeric_limits<size_t>::max());

  std::vector<uint8_t> Seen(PhysSize, 0);
  const size_t ActiveDims = active_dimensions_from_range3(Plan.PhysicalRange);

  // same-offset traversal:
  // 先固定 unit 内局部偏移，再遍历所有 unit
  for (size_t I0 = 0; I0 < Plan.ChunkLen[0]; ++I0) {
    for (size_t I1 = 0; I1 < Plan.ChunkLen[1]; ++I1) {
      for (size_t I2 = 0; I2 < Plan.ChunkLen[2]; ++I2) {
        for (size_t U0 = 0; U0 < Plan.UnitGrid[0]; ++U0) {
          for (size_t U1 = 0; U1 < Plan.UnitGrid[1]; ++U1) {
            for (size_t U2 = 0; U2 < Plan.UnitGrid[2]; ++U2) {
              id<3> Base{
                  U0 * Plan.Step[0],
                  U1 * Plan.Step[1],
                  U2 * Plan.Step[2],
              };
              id<3> Canon{
                  Base[0] + I0,
                  Base[1] + I1,
                  Base[2] + I2,
              };

              if (ActiveDims == 1) {
                Canon[1] = 0;
                Canon[2] = 0;
              } else if (ActiveDims == 2) {
                Canon[2] = 0;
              }

              const size_t CanonLinear = linearize3(Canon, Plan.PhysicalRange);
              append_first_touch(CanonLinear, Seen, Plan);
            }
          }
        }
      }
    }
  }

  // 补齐未覆盖元素
  for (size_t CanonLinear = 0; CanonLinear < PhysSize; ++CanonLinear) {
    append_first_touch(CanonLinear, Seen, Plan);
  }

  if (Plan.PackedToCanonical.size() != PhysSize) {
    throw std::runtime_error("gpu same-offset plan is not same-size");
  }

  return Plan;
}

inline backend_layout_plan
make_gpu_warp_same_offset_plan(const buffer_access_logic_impl &Logic,
                               const range<3> &PhysRange,
                               size_t WarpSize = 32) {
  if (!Logic.Enabled) {
    throw std::invalid_argument(
        "make_gpu_warp_same_offset_plan: logic is disabled");
  }

  backend_layout_plan Plan;
  Plan.Backend = backend_kind::gpu;
  Plan.Layout = backend_layout_kind::warp_same_offset_permutation;
  Plan.PhysicalRange = PhysRange;
  Plan.UnitGrid = unit_grid_from_logic(Logic, PhysRange);
  Plan.Step = Logic.Step;
  Plan.ChunkLen = Logic.ChunkLen;

  const size_t PhysSize = Plan.PhysicalRange.size();
  const size_t UnitCount = Plan.UnitGrid.size();
  const size_t ActiveDims = active_dimensions_from_range3(Plan.PhysicalRange);

  Plan.PackedToCanonical.reserve(PhysSize);
  Plan.CanonicalToPacked.assign(PhysSize, std::numeric_limits<size_t>::max());
  std::vector<uint8_t> Seen(PhysSize, 0);

  // warp 为 32 个 unit 一组
  for (size_t WarpBase = 0; WarpBase < UnitCount; WarpBase += WarpSize) {
    const size_t WarpEnd = std::min(WarpBase + WarpSize, UnitCount);

    // 先固定局部偏移，再扫 warp 内各 unit（线程）
    for (size_t I0 = 0; I0 < Plan.ChunkLen[0]; ++I0) {
      for (size_t I1 = 0; I1 < Plan.ChunkLen[1]; ++I1) {
        for (size_t I2 = 0; I2 < Plan.ChunkLen[2]; ++I2) {
          for (size_t U = WarpBase; U < WarpEnd; ++U) {
            id<3> UC = unit_coord_from_linear(U, Plan.UnitGrid);
            id<3> Base{
                UC[0] * Plan.Step[0],
                UC[1] * Plan.Step[1],
                UC[2] * Plan.Step[2],
            };
            id<3> Canon{
                Base[0] + I0,
                Base[1] + I1,
                Base[2] + I2,
            };

            if (ActiveDims == 1) {
              Canon[1] = 0;
              Canon[2] = 0;
            } else if (ActiveDims == 2) {
              Canon[2] = 0;
            }

            const size_t CanonLinear = linearize3(Canon, Plan.PhysicalRange);
            append_first_touch(CanonLinear, Seen, Plan);
          }
        }
      }
    }
  }

  // 补齐没有被任何 unit 覆盖到的元素
  for (size_t CanonLinear = 0; CanonLinear < PhysSize; ++CanonLinear) {
    append_first_touch(CanonLinear, Seen, Plan);
  }

  if (Plan.PackedToCanonical.size() != PhysSize) {
    throw std::runtime_error("warp-same-offset plan is not same-size");
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

inline bool try_parse_layout_kind(const char *S, backend_layout_kind &Out) {
  if (!S) return false;

  if (std::strcmp(S, "unit_first_touch") == 0 ||
      std::strcmp(S, "unit_first_touch_permutation") == 0) {
    Out = backend_layout_kind::unit_first_touch_permutation;
    return true;
  }
  if (std::strcmp(S, "blocked_first_touch") == 0 ||
      std::strcmp(S, "blocked_first_touch_permutation") == 0) {
    Out = backend_layout_kind::blocked_first_touch_permutation;
    return true;
  }
  if (std::strcmp(S, "frequency_aware") == 0 ||
      std::strcmp(S, "frequency_aware_permutation") == 0) {
    Out = backend_layout_kind::frequency_aware_permutation;
    return true;
  }
  if (std::strcmp(S, "same_offset") == 0 ||
      std::strcmp(S, "same_offset_permutation") == 0) {
    Out = backend_layout_kind::same_offset_permutation;
    return true;
  }
  if (std::strcmp(S, "warp_same_offset") == 0 ||
      std::strcmp(S, "warp_same_offset_permutation") == 0) {
    Out = backend_layout_kind::warp_same_offset_permutation;
    return true;
  }
  return false;
}

inline backend_layout_kind select_layout_kind_for_backend(backend_kind BK) {
  backend_layout_kind Override;
  if (BK == backend_kind::cpu) {
    if (try_parse_layout_kind(std::getenv("DAC_CPU_LAYOUT_KIND"), Override))
      return Override;
    return backend_layout_kind::unit_first_touch_permutation;
  }
  if (BK == backend_kind::gpu) {
    if (try_parse_layout_kind(std::getenv("DAC_GPU_LAYOUT_KIND"), Override))
      return Override;
    return backend_layout_kind::warp_same_offset_permutation;
  }
  return backend_layout_kind::unit_first_touch_permutation;
}


// inline backend_layout_kind
// select_layout_kind_for_backend(backend_kind BK) {
//   switch (BK) {
//   case backend_kind::cpu:
//     return backend_layout_kind::unit_first_touch_permutation;
//   case backend_kind::gpu:
//     // first version: keep same policy, split later
//     return backend_layout_kind::warp_same_offset_permutation;
//   }
//   return backend_layout_kind::unit_first_touch_permutation;
// }


__SYCL_EXPORT const device_layout_mapping *get_or_create_device_layout_mapping_for_accessor(
    const std::shared_ptr<buffer_impl> &Impl, context_impl *Ctx,
    backend_kind BK, backend_layout_kind LK);

} // namespace detail
} // namespace _V1
} // namespace sycl