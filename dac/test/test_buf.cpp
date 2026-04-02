#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/buffer_access_logic.hpp>
#include <iostream>

int main() {
  int data[16] = {};

  sycl::buffer<int, 2> buf(data, sycl::range<2>(4, 4));

  sycl::ext::oneapi::experimental::buffer_access_logic<2> logic;
  logic[0] = sycl::ext::oneapi::experimental::dim_operator(1, 2);
  logic[1] = sycl::ext::oneapi::experimental::dim_operator(2, 2);

  std::cout << "before set, has_access_logic = "
            << buf.has_access_logic() << std::endl;

  if (buf.has_access_logic()) {
    std::cerr << "Error: buffer should not have logic before set_access_logic\n";
    return 1;
  }

  buf.set_access_logic(logic);

  std::cout << "after set, has_access_logic = "
            << buf.has_access_logic() << std::endl;

  if (!buf.has_access_logic()) {
    std::cerr << "Error: buffer should have logic after set_access_logic\n";
    return 2;
  }

  // 只验证 get_access_logic() 能正常调用和返回
  auto got_logic = buf.get_access_logic();
  (void)got_logic;

  std::cout << "get_access_logic() returned successfully\n";
  std::cout << "PASS\n";
  return 0;
}