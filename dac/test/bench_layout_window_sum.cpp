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
  constexpr size_t PHYS_ROWS = 6;
  constexpr size_t PHYS_COLS = 6;
  constexpr size_t WIN = 3;

  // 6x6 physical matrix:
  // [ 1,  2,  3,  4,  5,  6]
  // [ 7,  8,  9, 10, 11, 12]
  // [13, 14, 15, 16, 17, 18]
  // [19, 20, 21, 22, 23, 24]
  // [25, 26, 27, 28, 29, 30]
  // [31, 32, 33, 34, 35, 36]
  std::vector<int> physical = {
       1,  2,  3,  4,  5,  6,
       7,  8,  9, 10, 11, 12,
      13, 14, 15, 16, 17, 18,
      19, 20, 21, 22, 23, 24,
      25, 26, 27, 28, 29, 30,
      31, 32, 33, 34, 35, 36};

  sycl::buffer<int, 2> buf(physical.data(), sycl::range<2>(PHYS_ROWS, PHYS_COLS));

  // 涓庡師绋嬪簭涓€鑷达細step=1, chunk_len=3
  // 琛岄€昏緫鏄犲皠: [0,1,2, 1,2,3, 2,3,4, 3,4,5]
  // 鍒楅€昏緫鏄犲皠: [0,1,2, 1,2,3, 2,3,4, 3,4,5]
  // 鍥犳 logical view 涓瘡涓?3x3 block锛屾濂藉搴旂墿鐞嗙煩闃典腑鐨勪竴涓粦鍔ㄧ獥鍙ｃ€?
  buffer_access_logic<2> logic;
  logic[0] = dim_operator(1, 3);
  logic[1] = dim_operator(1, 3);

  if (!logic.valid()) {
    std::cerr << "logic is invalid\n";
    return 1;
  }

  buf.set_access_logic(logic);

  auto logical_range = logic.logical_range_from_physical(buf.get_range());
  size_t logical_rows = logical_range[0];
  size_t logical_cols = logical_range[1];

  std::cout << "physical range = (" << PHYS_ROWS << ", " << PHYS_COLS << ")\n";
  std::cout << "logical range  = (" << logical_rows << ", " << logical_cols << ")\n";

  if (logical_rows != 12 || logical_cols != 12) {
    std::cerr << "unexpected logical range\n";
    return 2;
  }

  const size_t out_rows = logical_rows / WIN; // 4
  const size_t out_cols = logical_cols / WIN; // 4

  // ------------------------------------------------------------
  // 0) CPU reference: 鐩存帴鍦ㄧ墿鐞嗙煩闃典笂鍋?3x3 婊戝姩绐楀彛姹傚拰
  // ------------------------------------------------------------
  std::vector<int> expected(out_rows * out_cols, 0);
  for (size_t wr = 0; wr < out_rows; ++wr) {
    for (size_t wc = 0; wc < out_cols; ++wc) {
      int sum = 0;
      for (size_t i = 0; i < WIN; ++i) {
        for (size_t j = 0; j < WIN; ++j) {
          sum += physical[(wr + i) * PHYS_COLS + (wc + j)];
        }
      }
      expected[wr * out_cols + wc] = sum;
    }
  }

  // ------------------------------------------------------------
  // 1) Host accessor test
  //    鍦?logical view 涓紝姣忎釜 window 瀵瑰簲涓€涓?3x3 block锛?
  //    window(wr, wc) <=> logical rows [wr*3, wr*3+1, wr*3+2]
  //                      logical cols [wc*3, wc*3+1, wc*3+2]
  // ------------------------------------------------------------
  std::vector<int> host_out(out_rows * out_cols, -1);
  {
    sycl::host_accessor<int, 2, sycl::access::mode::read> acc(buf);
    for (size_t wr = 0; wr < out_rows; ++wr) {
      for (size_t wc = 0; wc < out_cols; ++wc) {
        int sum = 0;
        for (size_t i = 0; i < WIN; ++i) {
          for (size_t j = 0; j < WIN; ++j) {
            size_t lr = wr * WIN + i;
            size_t lc = wc * WIN + j;
            sum += acc[sycl::id<2>(lr, lc)];
          }
        }
        host_out[wr * out_cols + wc] = sum;
      }
    }
  }

  print_mat("expected ", expected, out_rows, out_cols);
  print_mat("host_out ", host_out, out_rows, out_cols);

  bool host_ok = equal_vec(host_out, expected);
  std::cout << "host sliding-window sum check: "
            << (host_ok ? "PASS" : "FAIL") << "\n";

  // ------------------------------------------------------------
  // 2) Device/kernel accessor test
  // ------------------------------------------------------------
  std::vector<int> device_out(out_rows * out_cols, -1);
  {
    sycl::buffer<int, 2> out_buf(device_out.data(), sycl::range<2>(out_rows, out_cols));

    sycl::queue q;
    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    q.submit([&](sycl::handler &h) {
      auto in_acc = buf.get_access<sycl::access::mode::read>(h);
      auto out_acc = out_buf.get_access<sycl::access::mode::write>(h);

      h.parallel_for(sycl::range<2>(out_rows, out_cols), [=](sycl::id<2> idx) {
        const size_t wr = idx[0];
        const size_t wc = idx[1];

        int sum = 0;
        for (size_t i = 0; i < WIN; ++i) {
          for (size_t j = 0; j < WIN; ++j) {
            size_t lr = wr * WIN + i;
            size_t lc = wc * WIN + j;
            sum += in_acc[sycl::id<2>(lr, lc)];
          }
        }

        out_acc[idx] = sum;
      });
    });

    q.wait();
  }

  print_mat("device_out", device_out, out_rows, out_cols);

  bool device_ok = equal_vec(device_out, expected);
  std::cout << "device sliding-window sum check: "
            << (device_ok ? "PASS" : "FAIL") << "\n";

  std::cout << "\nSummary:\n";
  std::cout << "  host   : " << (host_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "  device : " << (device_ok ? "PASS" : "FAIL") << "\n";

  if (!host_ok) {
    std::cerr << "Host accessor path is not using logic correctly.\n";
    return 3;
  }

  if (!device_ok) {
    std::cerr << "Device accessor path is not using logic correctly.\n";
    std::cerr << "If host PASS but device FAIL, your runtime changes are in place,\n";
    std::cerr << "but compiler-side __init(...) argument passing is not finished yet.\n";
    return 4;
  }

  return 0;
}