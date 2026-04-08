//==----------------- buffer_impl.cpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/buffer_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/memory_manager.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/xpti_registry.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/properties/buffer_properties.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

static backend_layout_plan
build_layout_plan_from_impl(const buffer_access_logic_impl &LogicImpl,
                            range<3> PhysicalRange,
                            backend_kind BK,
                            backend_layout_kind LK) {
  (void)BK;
  (void)LK;
  return make_cpu_unit_first_touch_plan(LogicImpl, PhysicalRange);
}

const backend_layout_plan *
buffer_impl::getOrCreateBackendLayoutPlan(backend_kind BK,
                                          backend_layout_kind LK) const {
  if (!hasAccessLogic())
    return nullptr;

  if (MPhysicalRange[0] == 0 || MPhysicalRange[1] == 0 || MPhysicalRange[2] == 0) {
    throw std::runtime_error(
        "buffer_impl::getOrCreateBackendLayoutPlan: invalid physical range");
  }

  const backend_layout_cache_key Key{BK, LK};

  {
    std::lock_guard<std::mutex> Lock(MLayoutCacheMutex);
    auto It = MLayoutCache.find(Key);
    if (It != MLayoutCache.end() && It->second.HostPlan)
      return It->second.HostPlan.get();
  }

  auto Plan = std::make_shared<backend_layout_plan>(
      build_layout_plan_from_impl(*getAccessLogic(), MPhysicalRange, BK, LK));

  std::lock_guard<std::mutex> Lock(MLayoutCacheMutex);
  auto &Entry = MLayoutCache[Key];
  if (!Entry.HostPlan)
    Entry.HostPlan = std::move(Plan);
  return Entry.HostPlan.get();
}

const device_layout_mapping *
buffer_impl::getOrCreateDeviceLayoutMapping(context_impl *Ctx,
                                            backend_kind BK,
                                            backend_layout_kind LK) const {
  const backend_layout_plan *Plan = getOrCreateBackendLayoutPlan(BK, LK);
  if (!Plan)
    return nullptr;

  const backend_layout_cache_key Key{BK, LK};

  {
    std::lock_guard<std::mutex> Lock(MLayoutCacheMutex);
    auto It = MLayoutCache.find(Key);
    if (It != MLayoutCache.end()) {
      auto ItMap = It->second.PerContextMappings.find(Ctx);
      if (ItMap != It->second.PerContextMappings.end())
        return ItMap->second.get();
    }
  }

  auto Mapping = std::make_shared<device_layout_mapping>();
  Mapping->HostCanonicalToPacked =
      std::make_shared<std::vector<size_t>>(Plan->CanonicalToPacked);

  Mapping->Desc.Enabled = true;
  Mapping->Desc.Backend = BK;
  Mapping->Desc.Layout = LK;
  Mapping->Desc.PhysicalRange = Plan->PhysicalRange;
  Mapping->Desc.TableSize = Mapping->HostCanonicalToPacked->size();

  if (BK == backend_kind::cpu) {
    Mapping->DeviceCanonicalToPacked =
        Mapping->HostCanonicalToPacked->data();
    Mapping->Desc.CanonicalToPacked = Mapping->DeviceCanonicalToPacked;
  } else {
    // GPU 先占位：后续在这里把表上传到 device/USM allocation
    Mapping->Desc.Enabled = false;
  }

  std::lock_guard<std::mutex> Lock(MLayoutCacheMutex);
  auto &Entry = MLayoutCache[Key];
  auto &Slot = Entry.PerContextMappings[Ctx];
  if (!Slot)
    Slot = std::move(Mapping);
  return Slot.get();
}

const device_layout_mapping *
get_or_create_device_layout_mapping_for_accessor(
    const std::shared_ptr<buffer_impl> &Impl, context_impl *Ctx,
    backend_kind BK, backend_layout_kind LK) {
  if (!Impl || !Impl->hasBackendLayoutPolicy())
    return nullptr;
  return Impl->getOrCreateDeviceLayoutMapping(Ctx, BK, LK);
}

void *buffer_impl::allocateMem(context_impl *Context, bool InitFromUserData,
                               void *HostPtr,
                               ur_event_handle_t &OutEventToWait) {
  bool HostPtrReadOnly = false;
  BaseT::determineHostPtr(Context, InitFromUserData, HostPtr, HostPtrReadOnly);
  assert(!(nullptr == HostPtr && BaseT::useHostPtr() && !Context) &&
         "Internal error. Allocating memory on the host "
         "while having use_host_ptr property");
  return MemoryManager::allocateMemBuffer(
      Context, this, HostPtr, HostPtrReadOnly, BaseT::getSizeInBytes(),
      BaseT::MInteropEvent, BaseT::MInteropContext.get(), MProps,
      OutEventToWait);
}
void buffer_impl::constructorNotification(const detail::code_location &CodeLoc,
                                          void *UserObj, const void *HostObj,
                                          const void *Type, uint32_t Dim,
                                          uint32_t ElemSize, size_t Range[3]) {
  
  const size_t R0 = Range[0];
  const size_t R1 = Dim >= 2 ? Range[1] : 1;
  const size_t R2 = Dim >= 3 ? Range[2] : 1;
  setPhysicalRange(range<3>{Range[0], Range[1], Range[2]});
  XPTIRegistry::bufferConstructorNotification(UserObj, CodeLoc, HostObj, Type,
                                              Dim, ElemSize, Range);
}

void buffer_impl::destructorNotification(void *UserObj) {
  XPTIRegistry::bufferDestructorNotification(UserObj);
}

void buffer_impl::addInteropObject(
    std::vector<ur_native_handle_t> &Handles) const {
  if (MOpenCLInterop) {
    if (std::find(Handles.begin(), Handles.end(),
                  ur::cast<ur_native_handle_t>(MInteropMemObject)) ==
        Handles.end()) {
      adapter_impl &Adapter = getAdapter();
      Adapter.call<UrApiKind::urMemRetain>(
          ur::cast<ur_mem_handle_t>(MInteropMemObject));
      ur_native_handle_t NativeHandle = 0;
      Adapter.call<UrApiKind::urMemGetNativeHandle>(MInteropMemObject, nullptr,
                                                    &NativeHandle);
      Handles.push_back(NativeHandle);
    }
  }
}

std::vector<ur_native_handle_t>
buffer_impl::getNativeVector(backend BackendName) const {
  std::vector<ur_native_handle_t> Handles{};
  if (!MRecord) {
    addInteropObject(Handles);
    return Handles;
  }

  for (auto &Cmd : MRecord->MAllocaCommands) {
    ur_mem_handle_t NativeMem =
        ur::cast<ur_mem_handle_t>(Cmd->getMemAllocation());
    auto Ctx = Cmd->getWorkerContext();
    // If Host Shared Memory is not supported then there is alloca for host that
    // doesn't have context and platform
    if (!Ctx)
      continue;
    const platform_impl &Platform = Ctx->getPlatformImpl();
    if (Platform.getBackend() != BackendName)
      continue;

    adapter_impl &Adapter = Platform.getAdapter();
    ur_native_handle_t Handle = 0;
    // When doing buffer interop we don't know what device the memory should be
    // resident on, so pass nullptr for Device param. Buffer interop may not be
    // supported by all backends.
    Adapter.call<UrApiKind::urMemGetNativeHandle>(NativeMem, /*Dev*/ nullptr,
                                                  &Handle);
    Handles.push_back(Handle);

    if (Platform.getBackend() == backend::opencl) {
      __SYCL_OCL_CALL(clRetainMemObject, ur::cast<cl_mem>(Handle));
    }
  }

  addInteropObject(Handles);
  return Handles;
}

void buffer_impl::verifyProps(const property_list &Props) const {
  auto CheckDataLessProperties = [](int PropertyKind) {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)
    switch (PropertyKind) {
#include <sycl/properties/buffer_properties.def>
    default:
      return false;
    }
  };
  auto CheckPropertiesWithData = [](int PropertyKind) {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
    switch (PropertyKind) {
#include <sycl/properties/buffer_properties.def>
    default:
      return false;
    }
  };
  detail::PropertyValidator::checkPropsAndThrow(Props, CheckDataLessProperties,
                                                CheckPropertiesWithData);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
