#include <sycl/sycl.hpp>
#include <sycl/detail/backend_layout_plan.hpp>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using sycl::ext::oneapi::experimental::buffer_access_logic;
using sycl::ext::oneapi::experimental::dim_operator;

using sycl::detail::backend_kind;
using sycl::detail::backend_layout_kind;
using sycl::detail::backend_layout_plan;
using sycl::detail::buffer_access_logic_impl;
using sycl::detail::make_buffer_access_logic_impl;
using sycl::detail::select_layout_kind_for_backend;

static const char *kind_name(backend_layout_kind K) {
  switch (K) {
  case backend_layout_kind::unit_first_touch_permutation:
    return "unit_first_touch_permutation";
  case backend_layout_kind::blocked_first_touch_permutation:
    return "blocked_first_touch_permutation";
  case backend_layout_kind::frequency_aware_permutation:
    return "frequency_aware_permutation";
  case backend_layout_kind::same_offset_permutation:
    return "same_offset_permutation";
  case backend_layout_kind::warp_same_offset_permutation:
    return "warp_same_offset_permutation";
  }
  return "unknown";
}

static backend_layout_plan build_plan_by_kind(const buffer_access_logic_impl &Impl,
                                              sycl::range<3> PhysRange,
                                              backend_kind BK,
                                              backend_layout_kind LK) {
  if (BK == backend_kind::cpu) {
    switch (LK) {
    case backend_layout_kind::unit_first_touch_permutation:
      return sycl::detail::make_cpu_unit_first_touch_plan(Impl, PhysRange);
    case backend_layout_kind::blocked_first_touch_permutation:
      return sycl::detail::make_cpu_blocked_first_touch_plan(Impl, PhysRange);
    case backend_layout_kind::frequency_aware_permutation:
      return sycl::detail::make_cpu_frequency_aware_plan(Impl, PhysRange);
    default:
      throw std::runtime_error("unsupported cpu layout kind in test");
    }
  } else {
    switch (LK) {
    case backend_layout_kind::same_offset_permutation:
      return sycl::detail::make_gpu_same_offset_plan(Impl, PhysRange);
    case backend_layout_kind::warp_same_offset_permutation:
      return sycl::detail::make_gpu_warp_same_offset_plan(Impl, PhysRange, 32);
    default:
      throw std::runtime_error("unsupported gpu layout kind in test");
    }
  }
}

static void print_plan_prefix(const backend_layout_plan &Plan,
                              const std::string &Tag,
                              size_t N = 24) {
  std::cout << Tag << " -> layout=" << kind_name(Plan.Layout) << "\n";
  std::cout << "  first " << N << " PackedToCanonical: ";
  for (size_t i = 0; i < std::min(N, Plan.PackedToCanonical.size()); ++i) {
    std::cout << Plan.PackedToCanonical[i];
    if (i + 1 != std::min(N, Plan.PackedToCanonical.size()))
      std::cout << ", ";
  }
  std::cout << "\n";
}

static bool same_permutation(const backend_layout_plan &A,
                             const backend_layout_plan &B) {
  return A.PackedToCanonical == B.PackedToCanonical;
}

static void test_cpu_env_layout_select() {
  std::cout << "==== CPU env layout select test ====\n";

  // 用一个 overlap 明显、unit grid 足够大的 case
  buffer_access_logic<2> Logic;
  Logic[0] = dim_operator(1, 3);
  Logic[1] = dim_operator(1, 3);

  auto Impl = make_buffer_access_logic_impl(Logic);
  sycl::range<3> PhysRange{10, 10, 1};

  setenv("DAC_CPU_LAYOUT_KIND", "unit_first_touch", 1);
  auto LK1 = select_layout_kind_for_backend(backend_kind::cpu);
  auto P1 = build_plan_by_kind(Impl, PhysRange, backend_kind::cpu, LK1);
  print_plan_prefix(P1, "CPU env=unit_first_touch");
  assert(LK1 == backend_layout_kind::unit_first_touch_permutation);

  setenv("DAC_CPU_LAYOUT_KIND", "blocked_first_touch", 1);
  auto LK2 = select_layout_kind_for_backend(backend_kind::cpu);
  auto P2 = build_plan_by_kind(Impl, PhysRange, backend_kind::cpu, LK2);
  print_plan_prefix(P2, "CPU env=blocked_first_touch");
  assert(LK2 == backend_layout_kind::blocked_first_touch_permutation);

  setenv("DAC_CPU_LAYOUT_KIND", "frequency_aware", 1);
  auto LK3 = select_layout_kind_for_backend(backend_kind::cpu);
  auto P3 = build_plan_by_kind(Impl, PhysRange, backend_kind::cpu, LK3);
  print_plan_prefix(P3, "CPU env=frequency_aware");
  assert(LK3 == backend_layout_kind::frequency_aware_permutation);

  // 至少前两种和第三种不应全都一样
  const bool Eq12 = same_permutation(P1, P2);
  const bool Eq13 = same_permutation(P1, P3);
  const bool Eq23 = same_permutation(P2, P3);

  std::cout << "  P1 == P2 ? " << (Eq12 ? "true" : "false") << "\n";
  std::cout << "  P1 == P3 ? " << (Eq13 ? "true" : "false") << "\n";
  std::cout << "  P2 == P3 ? " << (Eq23 ? "true" : "false") << "\n";

  if (Eq12 && Eq13 && Eq23) {
    throw std::runtime_error(
        "CPU layouts selected by env are all identical; env override did not take effect");
  }

  unsetenv("DAC_CPU_LAYOUT_KIND");
  std::cout << "CPU env layout select: PASS\n\n";
}

static void test_gpu_env_layout_select() {
  std::cout << "==== GPU env layout select test ====\n";

  // unit 数要 > 32，这样 same_offset 和 warp_same_offset 才会分化
  buffer_access_logic<2> Logic;
  Logic[0] = dim_operator(2, 2);
  Logic[1] = dim_operator(2, 2);

  auto Impl = make_buffer_access_logic_impl(Logic);
  sycl::range<3> PhysRange{20, 20, 1};

  setenv("DAC_GPU_LAYOUT_KIND", "same_offset", 1);
  auto LK1 = select_layout_kind_for_backend(backend_kind::gpu);
  auto P1 = build_plan_by_kind(Impl, PhysRange, backend_kind::gpu, LK1);
  print_plan_prefix(P1, "GPU env=same_offset");
  assert(LK1 == backend_layout_kind::same_offset_permutation);

  setenv("DAC_GPU_LAYOUT_KIND", "warp_same_offset", 1);
  auto LK2 = select_layout_kind_for_backend(backend_kind::gpu);
  auto P2 = build_plan_by_kind(Impl, PhysRange, backend_kind::gpu, LK2);
  print_plan_prefix(P2, "GPU env=warp_same_offset");
  assert(LK2 == backend_layout_kind::warp_same_offset_permutation);

  const bool Eq = same_permutation(P1, P2);
  std::cout << "  P1 == P2 ? " << (Eq ? "true" : "false") << "\n";

  if (Eq) {
    throw std::runtime_error(
        "GPU same_offset and warp_same_offset are identical; env override or plan generation did not take effect");
  }

  unsetenv("DAC_GPU_LAYOUT_KIND");
  std::cout << "GPU env layout select: PASS\n\n";
}

int main() {
  try {
    test_cpu_env_layout_select();
    test_gpu_env_layout_select();
    std::cout << "test_env_layout_select PASS\n";
    return 0;
  } catch (const std::exception &E) {
    std::cerr << "test_env_layout_select FAIL: " << E.what() << "\n";
    return 1;
  }
}