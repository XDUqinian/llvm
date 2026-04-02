#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/buffer_access_logic.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

using sycl::ext::oneapi::experimental::buffer_access_logic;
using sycl::ext::oneapi::experimental::dim_operator;

static void print_mat(const char *name, const std::vector<int> &v,
                      size_t rows, size_t cols) {
  std::cout << name << ":\n";
  for (size_t i = 0; i < rows; ++i) {
    std::cout << "  [";
    for (size_t j = 0; j < cols; ++j) {
      std::cout << v[i * cols + j];
      if (j + 1 != cols)
        std::cout << ", ";
    }
    std::cout << "]\n";
  }
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

int main() {
  // 4x4 physical matrix:
  // [ 1,  2,  3,  4]
  // [ 5,  6,  7,  8]
  // [ 9, 10, 11, 12]
  // [13, 14, 15, 16]
  std::vector<int> physical = {
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12,
     13, 14, 15, 16};

  sycl::buffer<int, 2> buf(physical.data(), sycl::range<2>(4, 4));

  buffer_access_logic<2> logic;
  logic[0] = dim_operator(1, 2); // row:  0,1,1,2,2,3
  logic[1] = dim_operator(1, 2); // col:  0,1,1,2,2,3

  if (!logic.valid()) {
    std::cerr << "logic is invalid\n";
    return 1;
  }

  buf.set_access_logic(logic);

  auto logical_range = logic.logical_range_from_physical(buf.get_range());
  size_t logical_rows = logical_range[0];
  size_t logical_cols = logical_range[1];

  std::cout << "physical range = (" << buf.get_range()[0] << ", "
            << buf.get_range()[1] << ")\n";
  std::cout << "logical range  = (" << logical_rows << ", "
            << logical_cols << ")\n";

  if (logical_rows != 6 || logical_cols != 6) {
    std::cerr << "unexpected logical range\n";
    return 1;
  }

  // Expected logical 6x6 view:
  // row map = [0,1,1,2,2,3]
  // col map = [0,1,1,2,2,3]
  std::vector<int> expected = {
      1,  2,  2,  3,  3,  4,
      5,  6,  6,  7,  7,  8,
      5,  6,  6,  7,  7,  8,
      9, 10, 10, 11, 11, 12,
      9, 10, 10, 11, 11, 12,
     13, 14, 14, 15, 15, 16};

  // ------------------------------------------------------------
  // 1) Host accessor test
  // ------------------------------------------------------------
  std::vector<int> host_out(logical_rows * logical_cols, -1);
  {
    sycl::host_accessor<int, 2, sycl::access::mode::read> acc(buf);
    for (size_t i = 0; i < logical_rows; ++i) {
      for (size_t j = 0; j < logical_cols; ++j) {
        host_out[i * logical_cols + j] = acc[sycl::id<2>(i, j)];
      }
    }
  }

  print_mat("expected ", expected, logical_rows, logical_cols);
  print_mat("host_out ", host_out, logical_rows, logical_cols);

  bool host_ok = equal_vec(host_out, expected);
  std::cout << "host accessor check: " << (host_ok ? "PASS" : "FAIL") << "\n";

  // ------------------------------------------------------------
  // 2) Device/kernel accessor test
  // ------------------------------------------------------------
  std::vector<int> device_out(logical_rows * logical_cols, -1);

  {
    sycl::buffer<int, 2> out_buf(device_out.data(),
                                 sycl::range<2>(logical_rows, logical_cols));

    sycl::queue q;
    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    q.submit([&](sycl::handler &h) {
      auto in_acc = buf.get_access<sycl::access::mode::read>(h);
      auto out_acc = out_buf.get_access<sycl::access::mode::write>(h);

      h.parallel_for(sycl::range<2>(logical_rows, logical_cols),
                     [=](sycl::id<2> idx) {
        out_acc[idx] = in_acc[idx];
      });
    });

    q.wait();
  }

  print_mat("device_out", device_out, logical_rows, logical_cols);

  bool device_ok = equal_vec(device_out, expected);
  std::cout << "device accessor check: " << (device_ok ? "PASS" : "FAIL")
            << "\n";

  std::cout << "\nSummary:\n";
  std::cout << "  host   : " << (host_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "  device : " << (device_ok ? "PASS" : "FAIL") << "\n";

  if (!host_ok) {
    std::cerr << "Host accessor path is not using logic correctly.\n";
    return 2;
  }

  if (!device_ok) {
    std::cerr << "Device accessor path is not using logic correctly.\n";
    std::cerr << "If host PASS but device FAIL, your runtime changes are in place,\n";
    std::cerr << "but compiler-side __init(...) argument passing is not finished yet.\n";
    return 3;
  }

  return 0;
}