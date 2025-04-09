import functools
from utils import get_shape, get_dtype_size

def analyze_memory_usage(node, initializers, value_info_map, input_shapes, value_info_shapes):
    mem_size = 0
    for out in node.output:
        shape = eval(get_shape(out, input_shapes, value_info_shapes)) if get_shape(out, input_shapes, value_info_shapes) != "unknown" else []
        if shape:
            size = get_dtype_size(out, value_info_map) * max(1, functools.reduce(lambda a,b: a*b, shape, 1))
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