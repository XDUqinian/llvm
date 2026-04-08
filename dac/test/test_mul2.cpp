#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/buffer_access_logic.hpp>

#include <iostream>
#include <vector>

using sycl::ext::oneapi::experimental::buffer_access_logic;
using sycl::ext::oneapi::experimental::dim_operator;

int main() {
  using T = float;

  constexpr size_t M = 64;
  constexpr size_t K = 64;
  constexpr size_t N = 64;

  constexpr size_t BM = 8;
  constexpr size_t BK = 8;
  constexpr size_t BN = 8;

  static_assert(M % BM == 0);
  static_assert(K % BK == 0);
  static_assert(N % BN == 0);

  constexpr size_t MT = M / BM;
  constexpr size_t KT = K / BK;
  constexpr size_t NT = N / BN;

  std::vector<T> A_host(M * K);
  std::vector<T> B_host(K * N);
  std::vector<T> C_host(M * N, 0);

  for (size_t i = 0; i < M; ++i)
    for (size_t k = 0; k < K; ++k)
      A_host[i * K + k] = static_cast<T>((i + k) % 7 + 1);

  for (size_t k = 0; k < K; ++k)
    for (size_t j = 0; j < N; ++j)
      B_host[k * N + j] = static_cast<T>((k + j) % 5 + 1);

  {
    sycl::buffer<T, 2> A_buf(A_host.data(), sycl::range<2>(M, K));
    sycl::buffer<T, 2> B_buf(B_host.data(), sycl::range<2>(K, N));
    sycl::buffer<T, 2> C_buf(C_host.data(), sycl::range<2>(M, N));

    // A 划成 BM x BK 的 tile
    buffer_access_logic<2> A_logic;
    A_logic[0] = dim_operator(BM, BM);
    A_logic[1] = dim_operator(BK, BK);

    // B 划成 BK x BN 的 tile
    buffer_access_logic<2> B_logic;
    B_logic[0] = dim_operator(BK, BK);
    B_logic[1] = dim_operator(BN, BN);

    A_buf.set_access_logic(A_logic);
    B_buf.set_access_logic(B_logic);

    sycl::queue q;

    q.submit([&](sycl::handler &h) {
      auto A = A_buf.get_access<sycl::access::mode::read>(h);
      auto B = B_buf.get_access<sycl::access::mode::read>(h);
      auto C = C_buf.get_access<sycl::access::mode::write>(h);

      h.parallel_for(sycl::range<2>(MT, NT), [=](sycl::id<2> tile_idx) {
        const size_t mt = tile_idx[0];
        const size_t nt = tile_idx[1];

        T acc[BM][BN] = {};

        for (size_t kt = 0; kt < KT; ++kt) {
          auto A_tile = A.get_unit_view(mt * KT + kt); // BM x BK
          auto B_tile = B.get_unit_view(kt * NT + nt); // BK x BN

          for (size_t i = 0; i < BM; ++i) {
            for (size_t j = 0; j < BN; ++j) {
              for (size_t k = 0; k < BK; ++k) {
                acc[i][j] += A_tile[sycl::id<2>(i, k)] *
                             B_tile[sycl::id<2>(k, j)];
              }
            }
          }
        }

        for (size_t i = 0; i < BM; ++i) {
          for (size_t j = 0; j < BN; ++j) {
            C[sycl::id<2>(mt * BM + i, nt * BN + j)] = acc[i][j];
          }
        }
      });
    });

    q.wait_and_throw();
  }

  // 简单校验
  bool ok = true;
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      T ref = 0;
      for (size_t k = 0; k < K; ++k)
        ref += A_host[i * K + k] * B_host[k * N + j];
      if (C_host[i * N + j] != ref) {
        std::cout << "mismatch at (" << i << ", " << j
                  << "), got " << C_host[i * N + j]
                  << ", expected " << ref << "\n";
        ok = false;
        goto done;
      }
    }
  }

done:
  std::cout << (ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}