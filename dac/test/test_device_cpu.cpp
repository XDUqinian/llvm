#include <sycl/sycl.hpp>
#include <iostream>
#include <string>

int main() {
  try {
    sycl::queue q{sycl::cpu_selector_v};

    const sycl::device q_dev = q.get_device();
    sycl::device h_dev;
    bool same_device = false;
    bool callback_ran = false;

    q.submit([&](sycl::handler &h) {
      h_dev = sycl::detail::getDeviceFromHandler(h);
      same_device = (h_dev == q_dev);
      callback_ran = true;

      h.single_task<class test_get_device_from_handler_cpu_dummy>([]() {});
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

    std::cout << "queue is_cpu        : " << q_dev.is_cpu() << "\n";
    std::cout << "handler is_cpu      : " << h_dev.is_cpu() << "\n";

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

    if (!h_dev.is_cpu()) {
      std::cerr << "FAIL: handler device is not CPU\n";
      return 3;
    }

    std::cout << "PASS\n";
    return 0;
  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return 4;
  } catch (const std::exception &e) {
    std::cerr << "std::exception: " << e.what() << "\n";
    return 5;
  }
}