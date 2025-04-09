import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_shape(name, input_shapes, value_info_shapes):
    shape = input_shapes.get(name) or value_info_shapes.get(name, [])
    if shape:
        return str(shape)
    return "unknown"

def get_dtype_size(name, value_info_map):
    vi = value_info_map.get(name)
    if vi:
        dtype = vi.type.tensor_type.elem_type
        return {
            1: 4,   # float32
            2: 1,   # uint8
            3: 1,   # int8
            5: 2,   # float16
            6: 2,   # int16
            7: 4,   # int32
            8: 8,   # int64
            10: 1,  # bool
        }.get(dtype, 4)
    return 4

def set_logging_level(verbose):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO) 