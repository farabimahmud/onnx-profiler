import functools
from utils import get_shape, get_dtype_size

def analyze_memory_usage(node, initializers, tensor_shapes):
    mem_size = 0
    for out in node.output:
        shape = tensor_shapes.get(out, [])
        if shape:
            size = 4 * max(1, functools.reduce(lambda a,b: a*b, shape, 1))  # Assume float32 (4 bytes)
            mem_size += size
    return mem_size

def analyze_parameters(node, initializers):
    param_count = 0
    for inp in node.input:
        if inp in initializers:
            tensor = initializers[inp]
            size = 1
            for d in tensor.dims:
                size *= d
            param_count += size
    return param_count 