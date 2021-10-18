import numpy as np
import dpctl
import dpctl.tensor as dpt
import gc
import ctypes

libmain = ctypes.cdll.LoadLibrary("./libmain.so")

kDLONEAPI_GPU = 14
kDLONEAPI_CPU = 15
kDLONEAPI_ACCELERATOR = 16

class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]

class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16)
    ]
    TYPE_MAP = {
        "bool": (1, 1, 1),
        "int8": (0, 8, 1),
        "uint8": (1, 8, 1),
        "int16": (0, 8, 1),
        "uint16": (1, 8, 1),
        "int32": (0, 32, 1),
        "uint32": (1, 32, 1),
        "int64": (0, 64, 1),
        "uint64": (1, 64, 1),
        "float16": (2, 16, 1),
        "float32": (2, 32, 1),
        "float64": (2, 64, 1),
        "complex64": (5, 64, 1),
        "complex128": (5, 128, 1),
    }

class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64)
    ]

class DLManagedTensor(ctypes.Structure):
    pass

DLManagedTensorHandle = ctypes.POINTER(DLManagedTensor)

DeleterFunc = ctypes.CFUNCTYPE(None, DLManagedTensorHandle)

DLManagedTensor._fields_ = [
     ("dl_tensor", DLTensor),
     ("manager_ctx", ctypes.c_void_p),
     ("deleter", DeleterFunc)
]


def display(array):
    print("data =", hex(array._pointer))
    print("dtype =", array.dtype)
    print("ndim =", array.ndim)
    print("shape =", array.shape)
    print("strides =", array.strides)
    sycl_dev = array.sycl_device
    print("sycl_device_name", sycl_dev.name)
    print("sycl_dlpack_id", sycl_dev.get_filter_string(include_backend=False))


def make_manager_ctx(obj):
    pyobj = ctypes.py_object(obj)
    void_p = ctypes.c_void_p.from_buffer(pyobj)  # like id(obj), but wrapped in c_void_p
    ctypes.pythonapi.Py_IncRef(pyobj)
    return void_p


# N.B.: In practice, one should ensure that this function
# is not destructed before the numpy array is destructed.
@DeleterFunc
def dl_managed_tensor_deleter(dl_managed_tensor_handle):
    void_p = dl_managed_tensor_handle.contents.manager_ctx
    pyobj = ctypes.cast(void_p, ctypes.py_object)
    print("Deleting manager_ctx:")
    display(pyobj.value)
    ctypes.pythonapi.Py_DecRef(pyobj)
    print("Deleter self...")
    libmain.FreeHandle()
    print("Done")


def vec_as(v, ctype):
    len_v = len(v)
    if len_v == 0:
        return None
    return (ctype * len_v)(*v)


def make_dl_tensor_usm_ndarray(array):
    if not isinstance(array, dpt.usm_ndarray):
        raise TypeError
    # You may check array.flags here, e.g. array.flags['C_CONTIGUOUS']
    dl_tensor = DLTensor()
    dl_tensor.data = ctypes.cast(array.usm_data._pointer, ctypes.c_void_p)
    sycl_dev = array.sycl_device
    dev_id = int(sycl_dev.get_filter_string(include_backend=False).split(":")[-1])
    if sycl_dev.device_type is dpctl.device_type.cpu:
        dl_tensor.device = DLDevice(kDLONEAPI_CPU, dev_id)
    elif sycl_dev.device_type is dpctl.device_type.gpu:
        dl_tensor.device = DLDevice(kDLONEAPI_GPU, dev_id)
    elif sycl_dev.device_type is dpctl.device_type.accelerator:
        dl_tensor.device = DLDevice(kDLONEAPI_ACCELERATOR, dev_id)
    else:
        raise RuntimeError
    dl_tensor.ndim = array.ndim
    dl_tensor.dtype = DLDataType.TYPE_MAP[str(array.dtype)]
    # For 0-dim ndarrays, strides and shape will be NULL
    dl_tensor.shape = vec_as(array.shape, ctypes.c_int64)
    dl_tensor.strides = vec_as(array.strides, ctypes.c_int64)
    dl_tensor.byte_offset = (array._pointer - array.usm_data._pointer)
    return dl_tensor


def main():
    usm_ary = dpt.usm_ndarray((3,1,30), dtype="f4", buffer="shared", buffer_ctor_kwargs={"queue": dpctl.SyclQueue("cpu")});
    usm_ary[:] = np.random.rand(3, 1, 30)
    print("Created:")
    display(usm_ary)
    c_obj = DLManagedTensor()
    c_obj.dl_tensor = make_dl_tensor_usm_ndarray(usm_ary)
    c_obj.manager_ctx = make_manager_ctx(usm_ary)
    c_obj.deleter = dl_managed_tensor_deleter
    print("-------------------------")
    del usm_ary
    gc.collect()
    libmain.Give(c_obj)
    print("-------------------------")
    del c_obj
    gc.collect()
    libmain.Finalize()
    print("-------------------------")


if __name__ == "__main__":
    main()
