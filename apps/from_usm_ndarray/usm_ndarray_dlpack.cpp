#include <CL/sycl.hpp>
#include <cstdint>
#include <iostream>
#include "dlpack/dlpack.h"

extern "C" {
void Give(DLManagedTensor dl_managed_tensor);
void Finalize();
void FreeHandle();
}

namespace {

DLManagedTensor *given = nullptr;

using filter_selector_t = sycl::ext::oneapi::filter_selector;

void display(DLManagedTensor mt) {
    DLTensor tensor = mt.dl_tensor;
    DLDevice dldev = tensor.device;
    DLDataType dtype = tensor.dtype;
    int i, ndim = tensor.ndim;
    std::cout << "On C++ side" << std::endl;
    std::cout << "Data: " << std::hex << tensor.data << std::dec << std::endl;
    std::cout << "Device (" << dldev.device_type << ", " << dldev.device_id << ")" << std::endl;
    std::cout << "Type description: { 'code': " << static_cast<int>(dtype.code)
	      << ", 'bits': " << static_cast<int>(dtype.bits)
	      << ", 'lanes': " << static_cast<int>(dtype.lanes) << " }"  << std::endl;
    std::cout << "ndim: " << ndim << std::endl;
    std::cout << "shape: (";
    {
	int64_t* sh = tensor.shape;
	for(int i=0; i < ndim; ++i) {
	    std::cout << sh[i];
	    std::cout << ((i+1) < ndim ? ", " : "");
	}
    }
    std::cout << ")" << std::endl;

    std::cout << "strides: (";
    {
	int64_t* st = tensor.strides;
	for(int i=0; i < ndim; ++i) {
	    std::cout << st[i];
	    std::cout << ((i+1) < ndim ? ", " : "");
	}
    }
    std::cout << ")" << std::endl;

    sycl::device *d_ptr = nullptr;

    switch(dldev.device_type) {
    case kDLONEAPI_GPU:
	d_ptr = new sycl::device(filter_selector_t("gpu:" + std::to_string(dldev.device_id)));
	break;
    case kDLONEAPI_CPU:
	d_ptr = new sycl::device(filter_selector_t("cpu:" + std::to_string(dldev.device_id)));
	break;
    case kDLONEAPI_ACCELERATOR:
	d_ptr = new sycl::device(filter_selector_t("accelerator:" + std::to_string(dldev.device_id)));
	break;
    default:
	throw std::runtime_error("Can not handler these types");
    }

    sycl::platform p = d_ptr->get_platform();
    sycl::context ctxt = p.ext_oneapi_get_default_context();
    delete d_ptr;

    sycl::usm::alloc kind = sycl::get_pointer_type(tensor.data, ctxt);
    switch (kind) {
    case sycl::usm::alloc::shared:
	std::cout << "USM-shared allocation-based array" << std::endl;
	break;
    case sycl::usm::alloc::device:
	std::cout << "USM-device allocation-based array" << std::endl;
	break;
    case sycl::usm::alloc::host:
	std::cout << "USM-host allocation-based array" << std::endl;
	break;
    default:
	std::cout << "USM allocation is unknown" << std::endl;
    }
}

} //  end of anonymous namespace

void Give(DLManagedTensor dl_managed_tensor) {
    display(dl_managed_tensor);
    given = (DLManagedTensor *) malloc(sizeof(DLManagedTensor));
    *given = dl_managed_tensor;
}

void Finalize() {
    given->deleter(given);
}

void FreeHandle() {
    free(given);
    given = NULL;
}
