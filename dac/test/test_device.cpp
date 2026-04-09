#include <sycl/sycl.hpp>
#include <iostream>
#include <string>

int main() {
  try {
    sycl::queue q{sycl::gpu_selector_v};

    const sycl::device q_dev = q.get_device();
    sycl::device h_dev;
    bool same_device = false;
    bool callback_ran = false;

    q.submit([&](sycl::handler &h) {
      // 这里测试的就是你仓库里的 __SYCL_EXPORT device getDeviceFromHandler(handler &)
      h_dev = sycl::detail::getDeviceFromHandler(h);
      same_device = (h_dev == q_dev);
      callback_ran = true;

      // 给 command group 一个实际 action，避免空提交
      h.single_task<class test_get_device_from_handler_dummy>([]() {});
    });

    q.wait_and_throw();

    std::cout << "callback_ran: " << (callback_ran ? "true" : "false") << "\n";

    std::cout << "queue device name   : "
              << q_dev.get_info<sycl::info::device::name>() << "\n";
    std::cout << "handler device name : "
              << h_dev.get_info<sycl::info::device::name>() << "\n";

    std::cout << "queue vendor        : "
              << q_dev.get_info<sycl::info::device::vendor>() << "\n";
    std::cout << "handler vendor      : "
              << h_dev.get_info<sycl::info::device::vendor>() << "\n";

    std::cout << "queue is_gpu        : " << q_dev.is_gpu() << "\n";
    std::cout << "handler is_gpu      : " << h_dev.is_gpu() << "\n";

    std::cout << "same_device         : " << (same_device ? "true" : "false")
              << "\n";

    if (!callback_ran) {
      std::cerr << "FAIL: submit callback did not run\n";
      return 2;
    }

    if (!same_device) {
      std::cerr << "FAIL: getDeviceFromHandler(handler&) != q.get_device()\n";
      return 1;
    }

    std::cout << "PASS\n";
    return 0;
  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return 3;
  } catch (const std::exception &e) {
    std::cerr << "std::exception: " << e.what() << "\n";
    return 4;
  }
}