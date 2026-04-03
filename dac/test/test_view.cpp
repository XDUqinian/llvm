#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/buffer_access_logic.hpp>

#include <array>
#include <cassert>
#include <numeric>
#include <vector>

int main() {
  using namespace sycl;
  using namespace sycl::ext::oneapi::experimental;

  std::vector<int> input(16);
  std::iota(input.begin(), input.end(), 0);

  std::vector<int> output(9 * 4, -1);

  buffer<int, 2> in_buf(input.data(), range<2>{4, 4});
  buffer<int, 1> out_buf(output.data(), range<1>{output.size()});

  buffer_access_logic<2> logic(
      std::array<dim_operator, 2>{dim_operator{1, 2}, dim_operator{1, 2}});
  in_buf.set_access_logic(logic);

  queue q;
  q.submit([&](handler &h) {
    auto in = in_buf.get_access(h, read_only);
    auto out = out_buf.get_access(h, write_only, no_init);

    h.parallel_for(range<1>{9}, [=](id<1> idx) {
      const size_t unit_id = idx[0];
      auto tile = in.get_unit_view(unit_id);

      // 这里预期 tile.get_range() == {2, 2}
      for (size_t i = 0; i < tile.get_range()[0]; ++i) {
        for (size_t j = 0; j < tile.get_range()[1]; ++j) {
          out[unit_id * 4 + i * 2 + j] = tile[id<2>{i, j}];
        }
      }
    });
  });

  auto result = out_buf.get_host_access(read_only);

  const int expected[9][4] = {
      {0, 1, 4, 5},
      {1, 2, 5, 6},
      {2, 3, 6, 7},
      {4, 5, 8, 9},
      {5, 6, 9, 10},
      {6, 7, 10, 11},
      {8, 9, 12, 13},
      {9, 10, 13, 14},
      {10, 11, 14, 15},
  };

  for (size_t u = 0; u < 9; ++u) {
    for (size_t k = 0; k < 4; ++k) {
      assert(result[u * 4 + k] == expected[u][k]);
    }
  }

  return 0;
}