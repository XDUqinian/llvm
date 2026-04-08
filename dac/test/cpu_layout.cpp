#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/buffer_access_logic.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using sycl::ext::oneapi::experimental::buffer_access_logic;
using sycl::ext::oneapi::experimental::dim_operator;

using T = float;
using Clock = std::chrono::steady_clock;

static double avg_ms(const std::vector<double> &v) {
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double min_ms(const std::vector<double> &v) {
  return *std::min_element(v.begin(), v.end());
}

static double max_abs_diff(const std::vector<T> &a, const std::vector<T> &b) {
  double mx = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    mx = std::max(mx, std::abs(static_cast<double>(a[i] - b[i])));
  return mx;
}

static std::vector<T> make_matrix(size_t rows, size_t cols) {
  std::vector<T> v(rows * cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      v[i * cols + j] = static_cast<T>(((i * 17 + j * 13) % 7) + 1);
    }
  }
  return v;
}

static std::vector<T> transpose_pack(const std::vector<T> &B, size_t K,
                                     size_t N) {
  // input:  B[k][j], shape K x N, row-major
  // output: Bt[j][k], shape N x K, row-major
  std::vector<T> Bt(N * K);
  for (size_t k = 0; k < K; ++k) {
    for (size_t j = 0; j < N; ++j) {
      Bt[j * K + k] = B[k * N + j];
    }
  }
  return Bt;
}

static std::vector<T> host_gemm(const std::vector<T> &A, const std::vector<T> &B,
                                size_t M, size_t K, size_t N) {
  std::vector<T> C(M * N, 0);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (size_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
  return C;
}

static double run_baseline_once(sycl::queue &q,
                                sycl::buffer<T, 2> &A_buf,
                                sycl::buffer<T, 2> &B_buf,
                                sycl::buffer<T, 2> &C_buf,
                                size_t M, size_t K, size_t N) {
  auto t0 = Clock::now();

  q.submit([&](sycl::handler &h) {
    auto A = A_buf.template get_access<sycl::access::mode::read>(h);
    auto B = B_buf.template get_access<sycl::access::mode::read>(h);
    auto C = C_buf.template get_access<sycl::access::mode::write>(h);

    h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> idx) {
      const size_t i = idx[0];
      const size_t j = idx[1];

      // B 的第 j 个列单元，逻辑尺寸应为 K x 1
      auto B_col = B.get_unit_view(j);

      T sum = 0;
      for (size_t k = 0; k < K; ++k) {
        sum += A[sycl::id<2>(i, k)] * B_col[sycl::id<2>(k, 0)];
      }
      C[idx] = sum;
    });
  });

  q.wait_and_throw();
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_optimized_once(sycl::queue &q,
                                 sycl::buffer<T, 2> &A_buf,
                                 sycl::buffer<T, 2> &Bt_buf,
                                 sycl::buffer<T, 2> &C_buf,
                                 size_t M, size_t K, size_t N) {
  auto t0 = Clock::now();

  q.submit([&](sycl::handler &h) {
    auto A = A_buf.template get_access<sycl::access::mode::read>(h);
    auto Bt = Bt_buf.template get_access<sycl::access::mode::read>(h);
    auto C = C_buf.template get_access<sycl::access::mode::write>(h);

    h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> idx) {
      const size_t i = idx[0];
      const size_t j = idx[1];

      // Bt 的第 j 个“行单元”，逻辑尺寸应为 1 x K
      auto B_row = Bt.get_unit_view(j);

      T sum = 0;
      for (size_t k = 0; k < K; ++k) {
        sum += A[sycl::id<2>(i, k)] * B_row[sycl::id<2>(0, k)];
      }
      C[idx] = sum;
    });
  });

  q.wait_and_throw();
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
  constexpr size_t M = 512;
  constexpr size_t K = 512;
  constexpr size_t N = 512;

  constexpr int Warmup = 2;
  constexpr int Repeat = 6;

  auto A_host = make_matrix(M, K);
  auto B_host = make_matrix(K, N);
  auto Bt_host = transpose_pack(B_host, K, N);

  auto C_ref = host_gemm(A_host, B_host, M, K, N);

  std::vector<T> C_base_host(M * N, 0);
  std::vector<T> C_opt_host(M * N, 0);

  sycl::queue q{sycl::cpu_selector{}};
  std::cout << "Running on: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  {
    sycl::buffer<T, 2> A_buf(A_host.data(), sycl::range<2>(M, K));
    sycl::buffer<T, 2> B_buf(B_host.data(), sycl::range<2>(K, N));
    sycl::buffer<T, 2> Bt_buf(Bt_host.data(), sycl::range<2>(N, K));
    sycl::buffer<T, 2> C_base_buf(C_base_host.data(), sycl::range<2>(M, N));
    sycl::buffer<T, 2> C_opt_buf(C_opt_host.data(), sycl::range<2>(M, N));

    // baseline: B 按列单元访问
    // dim0 不划分 => 整个 K 作为一个 chunk
    // dim1 划分成单列
    buffer_access_logic<2> B_col_logic;
    B_col_logic[0] = dim_operator(1, K);
    B_col_logic[1] = dim_operator(1, 1);

    // optimized: Bt 按行单元访问
    // dim0 划分成单行
    // dim1 不划分 => 整个 K 作为一个 chunk
    buffer_access_logic<2> Bt_row_logic;
    Bt_row_logic[0] = dim_operator(1, 1);
    Bt_row_logic[1] = dim_operator(1, K);

    if (!B_col_logic.valid() || !Bt_row_logic.valid()) {
      std::cerr << "logic is invalid\n";
      return 1;
    }

    B_buf.set_access_logic(B_col_logic);
    Bt_buf.set_access_logic(Bt_row_logic);

    // warmup
    for (int i = 0; i < Warmup; ++i) {
      run_baseline_once(q, A_buf, B_buf, C_base_buf, M, K, N);
      run_optimized_once(q, A_buf, Bt_buf, C_opt_buf, M, K, N);
    }

    std::vector<double> base_ms;
    std::vector<double> opt_ms;

    for (int i = 0; i < Repeat; ++i) {
      base_ms.push_back(run_baseline_once(q, A_buf, B_buf, C_base_buf, M, K, N));
      opt_ms.push_back(run_optimized_once(q, A_buf, Bt_buf, C_opt_buf, M, K, N));
    }

    // 显式同步 host 读取
    {
      sycl::host_accessor Cb(C_base_buf, sycl::read_only);
      sycl::host_accessor Co(C_opt_buf, sycl::read_only);
      for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
          C_base_host[i * N + j] = Cb[sycl::id<2>(i, j)];
          C_opt_host[i * N + j] = Co[sycl::id<2>(i, j)];
        }
      }
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nBaseline  (row-major B + column logic):\n";
    std::cout << "  avg = " << avg_ms(base_ms) << " ms\n";
    std::cout << "  min = " << min_ms(base_ms) << " ms\n";

    std::cout << "\nOptimized (packed/transposed B + row logic):\n";
    std::cout << "  avg = " << avg_ms(opt_ms) << " ms\n";
    std::cout << "  min = " << min_ms(opt_ms) << " ms\n";

    std::cout << "\nSpeedup:\n";
    std::cout << "  avg speedup = " << (avg_ms(base_ms) / avg_ms(opt_ms)) << "x\n";
    std::cout << "  min speedup = " << (min_ms(base_ms) / min_ms(opt_ms)) << "x\n";
  }

  double diff_base = max_abs_diff(C_base_host, C_ref);
  double diff_opt = max_abs_diff(C_opt_host, C_ref);

  std::cout << "\nCorrectness:\n";
  std::cout << "  baseline max abs diff  = " << diff_base << "\n";
  std::cout << "  optimized max abs diff = " << diff_opt << "\n";

  if (diff_base > 1e-4 || diff_opt > 1e-4) {
    std::cerr << "Result check failed.\n";
    return 2;
  }

  return 0;
}