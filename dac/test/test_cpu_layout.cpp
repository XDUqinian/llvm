#include <sycl/detail/backend_layout_plan.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

using sycl::detail::backend_layout_plan;
using sycl::detail::make_cpu_unit_first_touch_plan;
using sycl::detail::reorder_by_plan;
using sycl::ext::oneapi::experimental::buffer_access_logic;
using sycl::ext::oneapi::experimental::dim_operator;

static void print_vec(const char *name, const std::vector<int> &v) {
  std::cout << name << ": [";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i];
    if (i + 1 != v.size())
      std::cout << ", ";
  }
  std::cout << "]\n";
}

static bool equal_vec(const std::vector<int> &a, const std::vector<int> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

static bool verify_bijection(const backend_layout_plan &Plan) {
  const size_t N = Plan.physical_size();
  if (Plan.PackedToCanonical.size() != N || Plan.CanonicalToPacked.size() != N)
    return false;

  std::vector<int> SeenCanon(N, 0);
  std::vector<int> SeenPacked(N, 0);

  for (size_t p = 0; p < N; ++p) {
    size_t c = Plan.PackedToCanonical[p];
    if (c >= N)
      return false;
    if (Plan.CanonicalToPacked[c] != p)
      return false;
    SeenCanon[c] += 1;
    SeenPacked[p] += 1;
  }

  for (size_t i = 0; i < N; ++i) {
    if (SeenCanon[i] != 1 || SeenPacked[i] != 1)
      return false;
  }
  return true;
}

static int test_non_overlap() {
  std::cout << "\n[TEST] non-overlap 4x4, step=2, chunk=2\n";

  std::vector<int> canonical = {
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12,
     13, 14, 15, 16};

  buffer_access_logic<2> logic;
  logic[0] = dim_operator(2, 2);
  logic[1] = dim_operator(2, 2);

  auto plan = make_cpu_unit_first_touch_plan(logic, sycl::range<2>{4, 4});
  auto packed = reorder_by_plan(canonical, plan);

  std::vector<int> expected = {
      1,  2,  5,  6,
      3,  4,  7,  8,
      9, 10, 13, 14,
     11, 12, 15, 16};

  print_vec("packed  ", packed);
  print_vec("expected", expected);

  if (!verify_bijection(plan)) {
    std::cerr << "bijection check failed for non-overlap case\n";
    return 1;
  }

  if (!equal_vec(packed, expected)) {
    std::cerr << "packed order mismatch for non-overlap case\n";
    return 2;
  }

  std::cout << "PASS\n";
  return 0;
}

static int test_overlap() {
  std::cout << "\n[TEST] overlap 4x4, step=1, chunk=2\n";

  std::vector<int> canonical = {
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12,
     13, 14, 15, 16};

  buffer_access_logic<2> logic;
  logic[0] = dim_operator(1, 2);
  logic[1] = dim_operator(1, 2);

  auto plan = make_cpu_unit_first_touch_plan(logic, sycl::range<2>{4, 4});
  auto packed = reorder_by_plan(canonical, plan);

  // row-major unit order + row-major within unit + first-touch only
  std::vector<int> expected = {
      1, 2, 5, 6,
      3, 7, 4, 8,
      9, 10, 11, 12,
      13, 14, 15, 16};

  print_vec("packed  ", packed);
  print_vec("expected", expected);

  if (!verify_bijection(plan)) {
    std::cerr << "bijection check failed for overlap case\n";
    return 1;
  }

  if (!equal_vec(packed, expected)) {
    std::cerr << "packed order mismatch for overlap case\n";
    return 2;
  }

  std::cout << "PASS\n";
  return 0;
}

int main() {
  int ret = 0;
  ret |= test_non_overlap();
  ret |= test_overlap();

  if (ret != 0) {
    std::cerr << "\nSummary: FAIL\n";
    return ret;
  }

  std::cout << "\nSummary: PASS\n";
  return 0;
}