import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

ENGINE_PATH = "models/retinaface_model_fp16_640.engine"

def check(ret):
    # cuda-python cudart functions may return:
    #   - cudaError_t
    #   - (cudaError_t,)
    #   - (cudaError_t, value)
    if isinstance(ret, tuple):
        err = ret[0]
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Runtime error: {err}")
        return ret[1] if len(ret) > 1 else None
    else:
        err = ret
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Runtime error: {err}")
        return None


logger = trt.Logger(trt.Logger.INFO)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# --- Allocate buffers for each I/O tensor (static shapes in your engine) ---
io = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    shape = tuple(engine.get_tensor_shape(name))
    dtype = engine.get_tensor_dtype(name)

    if dtype != trt.DataType.FLOAT:
        raise RuntimeError(f"{name}: expected FLOAT, got {dtype}")

    host = np.empty(shape, dtype=np.float32)

    nbytes = host.nbytes
    dptr = check(cudart.cudaMalloc(nbytes))

    io.append((name, mode, host, dptr, nbytes))

# Stream
stream = check(cudart.cudaStreamCreate())

# Fill input with random data (just to prove execution works)
for (name, mode, host, dptr, nbytes) in io:
    if mode == trt.TensorIOMode.INPUT:
        host[:] = np.random.random_sample(host.shape).astype(np.float32)

# Copy inputs H2D + set addresses
for (name, mode, host, dptr, nbytes) in io:
    context.set_tensor_address(name, int(dptr))
    if mode == trt.TensorIOMode.INPUT:
        check(cudart.cudaMemcpyAsync(int(dptr), host.ctypes.data, nbytes,
                                     cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))

# Run
ok = context.execute_async_v3(stream)
if not ok:
    raise RuntimeError("execute_async_v3 returned False")

# Copy outputs D2H
for (name, mode, host, dptr, nbytes) in io:
    if mode == trt.TensorIOMode.OUTPUT:
        check(cudart.cudaMemcpyAsync(host.ctypes.data, int(dptr), nbytes,
                                     cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))

check(cudart.cudaStreamSynchronize(stream))

print("OK: ran", ENGINE_PATH)
for (name, mode, host, dptr, nbytes) in io:
    if mode == trt.TensorIOMode.OUTPUT:
        print(f"{name}: shape={host.shape}  dtype={host.dtype}  sample={host.reshape(-1)[:5]}")

# Cleanup
for (_, _, _, dptr, _) in io:
    check(cudart.cudaFree(int(dptr)))
check(cudart.cudaStreamDestroy(stream))
