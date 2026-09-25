"""Select one GPU consistently for CUDA and EGL before importing the simulator."""
import ctypes as C
import os
import subprocess


def configure_gpu():
    selected = os.environ.get("TABLETOP_GPU", "0")
    if not selected.isdigit():
        raise ValueError("TABLETOP_GPU must be a single physical GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # EGL indices are independent of CUDA indices. Query the visible CUDA ordinal.
    lib = C.CDLL("libEGL.so.1")
    lib.eglGetProcAddress.argtypes = [C.c_char_p]
    lib.eglGetProcAddress.restype = C.c_void_p
    enum_ptr = lib.eglGetProcAddress(b"eglQueryDevicesEXT")
    attr_ptr = lib.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not enum_ptr or not attr_ptr:
        raise RuntimeError("EGL device query extensions unavailable")
    enum = C.CFUNCTYPE(C.c_uint, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int))(enum_ptr)
    attr = C.CFUNCTYPE(C.c_uint, C.c_void_p, C.c_int, C.POINTER(C.c_ssize_t))(attr_ptr)
    devices, count = (C.c_void_p * 64)(), C.c_int()
    if not enum(64, devices, C.byref(count)):
        raise RuntimeError("EGL enumeration failed")
    for index in range(count.value):
        ordinal = C.c_ssize_t(-1)
        if attr(devices[index], 0x323A, C.byref(ordinal)) and ordinal.value == 0:
            # robosuite's import-time check confuses CUDA and EGL namespaces.
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
            import robosuite  # noqa: F401
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(index)
            os.environ["EGL_DEVICE_ID"] = str(index)
            return index
    raise RuntimeError("Cannot map selected GPU to EGL; refusing an unbound renderer")


def check_gpu_idle():
    selected = os.environ.get("TABLETOP_GPU", "0")
    used = subprocess.check_output([
        "nvidia-smi", f"--id={selected}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    if int(used) > 512:
        raise RuntimeError(f"GPU {selected} already uses {used} MiB; select an idle GPU")
