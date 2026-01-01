#!/usr/bin/env python3
"""
TensorRT ArcFace smoketest (Python 3.10, Jetson)

- Loads: models/arcface_model_fp16.engine
- Allocates GPU buffers with cuda-python (cudart)
- Runs one inference with a dummy (random) NCHW float32 input
- Prints embedding shape + first values + L2 norm

Run:
  python3.10 trt_arcface_smoketest.py
"""

import os
import sys
import numpy as np
import tensorrt as trt
from cuda import cudart


ENGINE_PATH = "models/arcface_model_fp16.engine"
INPUT_NAME_GUESS = None  # keep None (we auto-detect first INPUT tensor)


def check(err, msg=""):
    """
    cuda-python returns tuples like: (cudaError_t.cudaSuccess, <result...>)
    or sometimes only (cudaError_t.xxx,) depending on the call.
    This wrapper handles both safely.
    """
    if isinstance(err, tuple):
        status = err[0]
        rest = err[1:]  # may be empty
    else:
        status = err
        rest = ()

    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA failed: {status} {msg}")

    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return rest


def trt_dtype_to_np(dtype: trt.DataType):
    if dtype == trt.DataType.FLOAT:
        return np.float32
    if dtype == trt.DataType.HALF:
        return np.float16
    if dtype == trt.DataType.INT8:
        return np.int8
    if dtype == trt.DataType.INT32:
        return np.int32
    if dtype == trt.DataType.BOOL:
        return np.bool_
    raise TypeError(f"Unsupported TRT dtype: {dtype}")


def volume(shape):
    v = 1
    for d in shape:
        v *= int(d)
    return int(v)


def main():
    if not os.path.exists(ENGINE_PATH):
        print(f"ERROR: engine not found: {ENGINE_PATH}")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.INFO)
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize engine")

    # Create execution context
    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError("Failed to create execution context")

    # Auto-detect input/output tensor names (TensorRT 10 API)
    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    if not input_names:
        raise RuntimeError("No INPUT tensors found in engine")
    if not output_names:
        raise RuntimeError("No OUTPUT tensors found in engine")

    input_name = INPUT_NAME_GUESS or input_names[0]
    out_name = output_names[0]  # arcface usually has a single output

    in_shape = tuple(ctx.get_tensor_shape(input_name))
    out_shape = tuple(ctx.get_tensor_shape(out_name))

    in_dtype = engine.get_tensor_dtype(input_name)
    out_dtype = engine.get_tensor_dtype(out_name)

    np_in = trt_dtype_to_np(in_dtype)
    np_out = trt_dtype_to_np(out_dtype)

    # Sanity: expect fixed (1,3,112,112)
    if any(d <= 0 for d in in_shape):
        raise RuntimeError(f"Dynamic/invalid input shape in engine: {in_shape}")

    # Host buffers
    host_in = (np.random.rand(*in_shape).astype(np_in)).copy()
    host_out = np.empty(out_shape, dtype=np_out)

    # Device buffers
    in_bytes = host_in.nbytes
    out_bytes = host_out.nbytes

    d_in = check(cudart.cudaMalloc(in_bytes), "cudaMalloc input")
    d_out = check(cudart.cudaMalloc(out_bytes), "cudaMalloc output")

    stream = check(cudart.cudaStreamCreate(), "cudaStreamCreate")

    try:
        # H2D
        check(
            cudart.cudaMemcpyAsync(
                d_in,
                host_in.ctypes.data,
                in_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync H2D",
        )

        # Bind tensor addresses
        ctx.set_tensor_address(input_name, int(d_in))
        ctx.set_tensor_address(out_name, int(d_out))

        # Execute
        ok = ctx.execute_async_v3(stream_handle=stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")

        # D2H
        check(
            cudart.cudaMemcpyAsync(
                host_out.ctypes.data,
                d_out,
                out_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync D2H",
        )

        check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    finally:
        # Cleanup
        if stream is not None:
            try:
                check(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
            except Exception:
                pass
        if d_in is not None:
            try:
                check(cudart.cudaFree(d_in), "cudaFree input")
            except Exception:
                pass
        if d_out is not None:
            try:
                check(cudart.cudaFree(d_out), "cudaFree output")
            except Exception:
                pass

    emb = host_out.astype(np.float32, copy=False)  # for norm/printing
    l2 = float(np.linalg.norm(emb))

    print(f"OK: ran {ENGINE_PATH}")
    print(f"input tensor:  {input_name}  shape={in_shape}  dtype={np_in.__name__}")
    print(f"output tensor: {out_name}  shape={out_shape}  dtype={np_out.__name__}")
    flat = emb.reshape(-1)
    print("embedding sample:", flat[:8])
    print("L2 norm:", l2)


if __name__ == "__main__":
    main()

