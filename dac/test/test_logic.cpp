#include <sycl/ext/oneapi/experimental/buffer_access_logic.hpp>

using sycl::ext::oneapi::experimental::dim_operator;
using sycl::ext::oneapi::experimental::buffer_access_logic;

static_assert(dim_operator(1, 2).valid());
static_assert(dim_operator(1, 2).logical_extent(4) == 6);
static_assert(dim_operator(2, 2).logical_extent(4) == 4);
static_assert(dim_operator(3, 2).logical_extent(4) == 2);
static_assert(dim_operator(5, 2).logical_extent(4) == 2);

static_assert(dim_operator(1, 2).map(0) == 0);
static_assert(dim_operator(1, 2).map(1) == 1);
static_assert(dim_operator(1, 2).map(2) == 1);
static_assert(dim_operator(1, 2).map(3) == 2);
static_assert(dim_operator(2, 2).map(0) == 0);
static_assert(dim_operator(2, 2).map(1) == 1);
static_assert(dim_operator(2, 2).map(2) == 2);
static_assert(dim_operator(2, 2).map(3) == 3);

int main() {
  buffer_access_logic<2> logic;
  logic[0] = dim_operator(1, 2);
  logic[1] = dim_operator(2, 2);

  auto logical = logic.logical_range_from_physical(sycl::range<2>(4, 5));
  return (logical[0] == 6 && logical[1] == 4) ? 0 : 1;
}