//==----------------- buffer_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/buffer_access_logic_impl.hpp>
#include <sycl/detail/backend_layout_plan.hpp>
#include <detail/sycl_mem_obj_t.hpp>
#include <sycl/access/access.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/stl_type_traits.hpp> // for iterator_to_const_type_t
#include <sycl/detail/ur.hpp>
#include <sycl/property_list.hpp>

#include <optional>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <mutex>
#include <unordered_map>

namespace sycl {
inline namespace _V1 {
// Forward declarations
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
class accessor;
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;
template <typename DataT, int Dimensions, access::mode AccessMode>
class host_accessor;

namespace detail {

class buffer_impl final : public SYCLMemObjT {
  using BaseT = SYCLMemObjT;
  using typename BaseT::MemObjType;
private:
  std::optional<buffer_access_logic_impl> MAccessLogic;

  range<3> MPhysicalRange{1, 1, 1};

  struct cached_layout_entry {
    std::shared_ptr<const backend_layout_plan> HostPlan;
    std::unordered_map<context_impl *,
                       std::shared_ptr<device_layout_mapping>>
        PerContextMappings;
  };

  mutable std::mutex MLayoutCacheMutex;
  mutable std::unordered_map<backend_layout_cache_key,
                             cached_layout_entry,
                             backend_layout_cache_key_hash>
      MLayoutCache;

  void invalidateBackendLayoutCache() const {
    std::lock_guard<std::mutex> Lock(MLayoutCacheMutex);
    MLayoutCache.clear();
  }

public:
  buffer_impl(size_t SizeInBytes, size_t, const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    verifyProps(Props);
    if (Props.has_property<sycl::property::buffer::use_host_ptr>())
      throw sycl::exception(
          make_error_code(errc::invalid),
          "The use_host_ptr property requires host pointer to be provided");
  }

  buffer_impl(void *HostData, size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    verifyProps(Props);
    if (Props.has_property<
            sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
      throw sycl::exception(
          make_error_code(errc::invalid),
          "The use_pinned_host_memory cannot be used with host pointer");

    BaseT::handleHostData(HostData, RequiredAlign);
  }

  buffer_impl(const void *HostData, size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    verifyProps(Props);
    if (Props.has_property<
            sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
      throw sycl::exception(
          make_error_code(errc::invalid),
          "The use_pinned_host_memory cannot be used with host pointer");

    BaseT::handleHostData(HostData, RequiredAlign);
  }

  buffer_impl(const std::shared_ptr<const void> &HostData,
              const size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator, bool IsConstPtr)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    verifyProps(Props);
    if (Props.has_property<
            sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
      throw sycl::exception(
          make_error_code(errc::invalid),
          "The use_pinned_host_memory cannot be used with host pointer");

    BaseT::handleHostData(std::const_pointer_cast<void>(HostData),
                          RequiredAlign, IsConstPtr);
  }

  buffer_impl(const std::function<void(void *)> &CopyFromInput,
              const size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              std::unique_ptr<detail::SYCLMemObjAllocator> Allocator,
              bool IsConstPtr)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    verifyProps(Props);
    if (Props.has_property<
            sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
      throw sycl::exception(
          make_error_code(errc::invalid),
          "The use_pinned_host_memory cannot be used with host pointer");

    BaseT::handleHostData(CopyFromInput, RequiredAlign, IsConstPtr);
  }

  template <typename T>
  using EnableIfNotConstIterator =
      std::enable_if_t<!iterator_to_const_type_t<T>::value, T>;

  buffer_impl(cl_mem MemObject, const context &SyclContext,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              event AvailableEvent)
      : buffer_impl(ur::cast<ur_native_handle_t>(MemObject), SyclContext,
                    std::move(Allocator), /*OwnNativeHandle*/ true,
                    std::move(AvailableEvent)) {}

  buffer_impl(ur_native_handle_t MemObject, const context &SyclContext,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              bool OwnNativeHandle, event AvailableEvent)
      : BaseT(MemObject, SyclContext, OwnNativeHandle,
              std::move(AvailableEvent), std::move(Allocator)) {}

  void *allocateMem(context_impl *Context, bool InitFromUserData, void *HostPtr,
                    ur_event_handle_t &OutEventToWait) override;
  void constructorNotification(const detail::code_location &CodeLoc,
                               void *UserObj, const void *HostObj,
                               const void *Type, uint32_t Dim,
                               uint32_t ElemType, size_t Range[3]);
  void destructorNotification(void *UserObj);

  MemObjType getType() const override { return MemObjType::Buffer; }

  ~buffer_impl() {
    try {
      BaseT::updateHostMemory();
    } catch (...) {
    }
    destructorNotification(this);
  }

  void resize(size_t size) {
    BaseT::MSizeInBytes = size;
    invalidateBackendLayoutCache();
  }

  void setAccessLogic(const buffer_access_logic_impl &Logic) {
    MAccessLogic = Logic;
    invalidateBackendLayoutCache();
  }

  void setPhysicalRange(range<3> R) {
    const size_t R0 = R[0] == 0 ? 1 : R[0];
    const size_t R1 = R[1] == 0 ? 1 : R[1];
    const size_t R2 = R[2] == 0 ? 1 : R[2];
    MPhysicalRange = range<3>{R0, R1, R2};
    invalidateBackendLayoutCache();
  }

  range<3> getPhysicalRange() const noexcept { return MPhysicalRange; }

  bool hasAccessLogic() const noexcept {
    return MAccessLogic.has_value();
  }

  const buffer_access_logic_impl *getAccessLogic() const noexcept {
    return MAccessLogic ? &*MAccessLogic : nullptr;
  }

  size_t getAccessLogicElemSize() const noexcept override {
    return BaseT::get_allocator_internal()->getValueSize();
  }

  bool hasBackendLayoutPolicy() const noexcept override {
    return hasAccessLogic();
  }

  const backend_layout_plan *
  getOrCreateBackendLayoutPlan(backend_kind BK,
                               backend_layout_kind LK) const override;

  const device_layout_mapping *
  getOrCreateDeviceLayoutMapping(context_impl *Ctx, backend_kind BK,
                                 backend_layout_kind LK) const override;

  void addInteropObject(std::vector<ur_native_handle_t> &Handles) const;

  std::vector<ur_native_handle_t> getNativeVector(backend BackendName) const;

  void verifyProps(const property_list &Props) const;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
