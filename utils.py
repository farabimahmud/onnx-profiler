import logging
import sys
from pathlib import Path

def configure_logging(verbose: bool = False, log_file: Path = None):
    """Configure logging with both console and file handlers.
    
    Args:
        verbose: Whether to enable verbose (DEBUG) logging
        log_file: Path to the log file (optional)
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")

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

# Keep the old function for backward compatibility
def set_logging_level(verbose):
    configure_logging(verbose) 