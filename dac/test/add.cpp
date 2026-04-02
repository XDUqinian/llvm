#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
  constexpr size_t N = 1024;

  std::vector<float> a(N), b(N), c(N, 0.0f);
  for (size_t i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }

  try {
    sycl::queue q;

    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    {
      sycl::buffer<float, 1> bufA(a.data(), sycl::range<1>(N));
      sycl::buffer<float, 1> bufB(b.data(), sycl::range<1>(N));
      sycl::buffer<float, 1> bufC(c.data(), sycl::range<1>(N));

      q.submit([&](sycl::handler &h) {
        auto A = bufA.get_access<sycl::access::mode::read>(h);
        auto B = bufB.get_access<sycl::access::mode::read>(h);
        auto C = bufC.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
          C[i] = A[i] + B[i];
        });
      });

      q.wait();
    } // buffer 析构时把结果同步回 c

    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
      float expect = a[i] + b[i];
      if (c[i] != expect) {
        std::cout << "Mismatch at " << i
                  << ": got " << c[i]
                  << ", expect " << expect << "\n";
        ok = false;
        break;
      }
    }

    if (ok) {
      std::cout << "Success\n";
    } else {
      std::cout << "Failed\n";
      return 1;
    }

  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return 1;
  }

  return 0;
}