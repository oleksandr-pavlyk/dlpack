#include <CL/sycl.hpp>
#include "dlpack/dlpack.h"


#if __SYCL_COMPILER_VERSION < 20210925
using filter_selector_t = sycl::ONEAPI::filter_selector;
#else
using filter_selector_t = sycl::ext::oneapi::filter_selector;
#endif

/* Routines to construct SYCL device from DLPack's device_id */
sycl::device
get_cpu_device(size_t dev_id) {
    filter_selector_t fs("cpu:" + std::to_string(dev_id));
    return sycl::device{fs};
}

sycl::device
get_gpu_device(size_t dev_id) {
    filter_selector_t fs("gpu:" + std::to_string(dev_id));
    return sycl::device{fs};
}

sycl::device
get_accelerator_device(size_t dev_id) {
    filter_selector_t fs("accelerator:" + std::to_string(dev_id));
    return sycl::device{fs};
}

sycl::context
get_default_context(const sycl::device &dev) {
#if __SYCL_COMPILER_VERSION < 20210925
    sycl::queue q(dev);
    return q.get_context();
#else
#   if defined(SYCL_EXT_ONEAPI_DEFAULT_CONTEXT)
    // FIXME: must check that this is a root device
    auto p = dev.get_platform();
    return p.ext_oneapi_get_default_context();
#   else
    #error "Required default platform context extension is not available, \
           see https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/PlatformContext/PlatformContext.adoc"
#   endif
#endif
}

DLDeviceType
get_dlpack_device(const sycl::device &d, size_t &dev_id) {
    DLDeviceType dev_type;
    sycl::info::device_type sycl_dev_type =
	d.get_info<sycl::info::device::device_type>();

    switch(sycl_dev_type) {
    case sycl::info::device_type::cpu:
	dev_type = kDLONEAPI_CPU;
	break;
    case sycl::info::device_type::gpu:
	dev_type = kDLONEAPI_GPU;
	break;
    case sycl::info::device_type::accelerator:
	dev_type = kDLONEAPI_ACCELERATOR;
	break;
    default:
	throw std::runtime_error(
	    "Custom SYCL devices are not supported by DLPack protocol"
	    );
    }
    constexpr int not_found = -1;
    const auto &root_devices = sycl::device::get_devices();
    sycl::default_selector mRanker;
    int index = not_found;
    for (const auto &root_device : root_devices) {
	if (mRanker(root_device) < 0)
	    continue;
	if (sycl_dev_type == root_device.get_info<sycl::info::device::device_type>()) {
	    ++index;
	    if (root_device == d)
		break;
	}
    }
    dev_id = index;
    return dev_type;
}
