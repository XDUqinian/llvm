#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/buffer_access_logic.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

namespace exp = sycl::ext::oneapi::experimental;

static void print_vec(const char *name, const std::vector<int> &v) {
  std::cout << name << ": ";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i] << (i + 1 == v.size() ? '\n' : ' ');
  }
}

int main() {
  constexpr size_t N = 16;

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 0);

  // 这个 vector 用来接收“设备端按物理顺序读出来”的结果
  std::vector<int> raw_from_device(N, -1);

  try {
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    {
      sycl::buffer<int, 1> logic_buf(input.data(), sycl::range<1>(N));
      sycl::buffer<int, 1> raw_buf(raw_from_device.data(), sycl::range<1>(N));

      // 关键点：
      // 1) 让 has_access_logic() == true，从而触发你在 runtime 里加的 reorder
      // 2) 但逻辑映射本身保持 identity，不额外改 index
      exp::buffer_access_logic<1> identity_logic(
          std::array<exp::dim_operator, 1>{exp::dim_operator{1, 1}});
      logic_buf.set_access_logic(identity_logic);

      if (!logic_buf.has_access_logic()) {
        std::cerr << "logic_buf.has_access_logic() is false\n";
        return 2;
      }

      // Kernel: 直接把 logic_buf[i] 拷到一个“普通 buffer”里。
      //
      // 因为这里 access logic 是 identity：
      //   accessor[i] 读到的就是物理槽位 i 的内容。
      // 如果你的 H2D reorder 确实把设备侧物理内存倒序了，
      // 那 raw_buf 回传到 host 后应该正好是 N-1 ... 0。
      q.submit([&](sycl::handler &cgh) {
         auto in =
             logic_buf.template get_access<sycl::access::mode::read>(cgh);
         auto out =
             raw_buf.template get_access<sycl::access::mode::write>(cgh);

         cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
           const size_t i = idx[0];
           out[idx] = in[idx];
         });
       }).wait();

      // 读 raw_buf，触发普通 D2H，把“设备物理内容”带回 host
      {
        auto raw_acc =
            raw_buf.template get_access<sycl::access::mode::read>();
        for (size_t i = 0; i < N; ++i) {
          raw_from_device[i] = raw_acc[i];
        }
      }

      // 读 logic_buf，触发你实现的 D2H restore/unpack。
      // 如果 restore 正确，这里应该重新看到 0,1,2,...,N-1
      {
        auto host_acc =
            logic_buf.template get_access<sycl::access::mode::read>();

        bool restored_ok = true;
        for (size_t i = 0; i < N; ++i) {
          if (host_acc[i] != static_cast<int>(i)) {
            restored_ok = false;
            std::cerr << "Restore check failed at i=" << i
                      << ", got=" << host_acc[i]
                      << ", expected=" << i << "\n";
            break;
          }
        }

        if (!restored_ok) {
          print_vec("raw_from_device", raw_from_device);
          return 3;
        }
      }
    }

    std::vector<int> expect_reversed(N);
    for (size_t i = 0; i < N; ++i) {
      expect_reversed[i] = static_cast<int>(N - 1 - i);
    }

    bool raw_reversed_ok = (raw_from_device == expect_reversed);

    print_vec("input", input);
    print_vec("raw_from_device", raw_from_device);
    print_vec("expect_reversed", expect_reversed);

    if (!raw_reversed_ok) {
      std::cerr << "FAIL: device-side raw content is not reversed.\n";
      return 4;
    }

    std::cout << "PASS:\n";
    std::cout << "  1) device-side physical content is reversed\n";
    std::cout << "  2) host-side restored view is original order\n";
    return 0;
  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return 1;
  }
}