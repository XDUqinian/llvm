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
  constexpr size_t M = 3;
  constexpr size_t K = 4;
  constexpr size_t N = 2;

  // A: 3x4
  // [ 1,  2,  3,  4]
  // [ 5,  6,  7,  8]
  // [ 9, 10, 11, 12]
  std::vector<int> A_host = {
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12};

  // B: 4x2
  // [1, 2]
  // [3, 4]
  // [5, 6]
  // [7, 8]
  std::vector<int> B_host = {
      1, 2,
      3, 4,
      5, 6,
      7, 8};

  std::vector<int> C_host(M * N, -1);

  {
    sycl::buffer<int, 2> A_buf(A_host.data(), sycl::range<2>(M, K));
    sycl::buffer<int, 2> B_buf(B_host.data(), sycl::range<2>(K, N));
    sycl::buffer<int, 2> C_buf(C_host.data(), sycl::range<2>(M, N));

    // A 按“行单元”划分：
    // 行维 step=1, chunk=1
    // 列维不划分 => 整列宽度 K 作为一个 chunk
    buffer_access_logic<2> A_logic;
    A_logic[0] = dim_operator(1, 1);
    A_logic[1] = dim_operator(1, K);

    // B 按“列单元”划分：
    // 行维不划分 => 整行高度 K 作为一个 chunk
    // 列维 step=1, chunk=1
    buffer_access_logic<2> B_logic;
    B_logic[0] = dim_operator(1, K);
    B_logic[1] = dim_operator(1, 1);

    if (!A_logic.valid() || !B_logic.valid()) {
        std::cerr << "logic is invalid\n";
        return 1;
    }

    A_buf.set_access_logic(A_logic);
    B_buf.set_access_logic(B_logic);

    sycl::queue q;
    std::cout << "Running on: "
                << q.get_device().get_info<sycl::info::device::name>() << "\n";

    q.submit([&](sycl::handler &h) {
        auto A = A_buf.get_access<sycl::access::mode::read>(h);
        auto B = B_buf.get_access<sycl::access::mode::read>(h);
        auto C = C_buf.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> idx) {
        const size_t i = idx[0];
        const size_t j = idx[1];

        auto A_row = A.get_unit_view(i); // 1 x K
        auto B_col = B.get_unit_view(j); // K x 1

        int sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += A_row[sycl::id<2>(0, k)] * B_col[sycl::id<2>(k, 0)];
        }

        C[idx] = sum;
        });
    });

    q.wait();
  }

  std::vector<int> expected = {
      50,  60,
      114, 140,
      178, 220};

  print_mat("A", A_host, M, K);
  print_mat("B", B_host, K, N);
  print_mat("C", C_host, M, N);
  print_mat("expected", expected, M, N);

  bool ok = equal_vec(C_host, expected);
  std::cout << "matmul check: " << (ok ? "PASS" : "FAIL") << "\n";

  if (!ok) {
    std::cerr << "Matrix multiplication result is incorrect.\n";
    return 2;
  }

  return 0;
}