import onnxruntime as ort
from math import prod
from utils import get_dtype_size
import numpy as np
import onnx
import logging

logger = logging.getLogger(__name__)

# ONNX data type constants
FLOAT32 = 1
UINT8 = 2
INT8 = 3
FLOAT16 = 5
INT16 = 6
INT32 = 7
INT64 = 8
BOOL = 10

# Default baseline is float32
DEFAULT_BASELINE = FLOAT32

def get_flops_multiplier(dtype, baseline_dtype=DEFAULT_BASELINE):
    """Returns the FLOPs multiplier based on data type relative to the baseline.
    
    Args:
        dtype: The data type to get multiplier for
        baseline_dtype: The baseline data type to compare against (default: float32)
    
    Returns:
        float: Multiplier indicating relative FLOPs cost compared to baseline
    """
    # Define base multipliers relative to float32
    base_multipliers = {
        FLOAT32: 1.0,   # float32
        UINT8: 0.25,    # uint8
        INT8: 0.25,     # int8
        FLOAT16: 0.5,   # float16
        INT16: 0.5,     # int16
        INT32: 0.5,     # int32
        INT64: 1.0,     # int64
        BOOL: 0.25,     # bool
    }
    
    # Get the base multipliers for both types
    dtype_mult = base_multipliers.get(dtype, 1.0)
    baseline_mult = base_multipliers.get(baseline_dtype, 1.0)
    
    # Calculate relative multiplier
    # If baseline is float32 (1.0) and dtype is int8 (0.25), we want 4x more FLOPs
    # So we need to invert the relationship: baseline_mult / dtype_mult
    return baseline_mult / dtype_mult

def get_tensor_shapes(model_path):
    """Get tensor shapes using ONNX Runtime inference session.
    
    Args:
        model_path: Path to the ONNX model
    
    Returns:
        dict: Dictionary mapping tensor names to their shapes
    """
    # Create inference session
    session = ort.InferenceSession(model_path)
    
    # Get shapes of all tensors
    tensor_shapes = {}
    
    # Add input shapes
    for input in session.get_inputs():
        shape = []
        for dim in input.shape:
            if isinstance(dim, int):
                shape.append(dim)
            else:
                # For dynamic dimensions, use reasonable defaults
                if str(dim) == 'batch_size':
                    shape.append(1)  # Default batch size
                elif str(dim) == 'sequence_length':
                    shape.append(128)  # Default sequence length for BERT
                elif str(dim) == 'hidden_size':
                    shape.append(768)  # Default hidden size for BERT
                else:
                    shape.append(1)  # Default for other dynamic dimensions
        tensor_shapes[input.name] = shape
    
    # Add output shapes
    for output in session.get_outputs():
        shape = []
        for dim in output.shape:
            if isinstance(dim, int):
                shape.append(dim)
            else:
                # For dynamic dimensions, use reasonable defaults
                if str(dim) == 'batch_size':
                    shape.append(1)  # Default batch size
                elif str(dim) == 'sequence_length':
                    shape.append(128)  # Default sequence length for BERT
                elif str(dim) == 'hidden_size':
                    shape.append(768)  # Default hidden size for BERT
                else:
                    shape.append(1)  # Default for other dynamic dimensions
        tensor_shapes[output.name] = shape
    
    # Get shapes from the model's graph
    model = onnx.load(model_path)
    
    # Run shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning(f"Shape inference failed: {e}")
    
    # Add shapes from value info
    for value_info in model.graph.value_info:
        if value_info.name not in tensor_shapes:
            shape = []
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    # For dynamic dimensions, use reasonable defaults
                    if dim.dim_param == 'batch_size':
                        shape.append(1)  # Default batch size
                    elif dim.dim_param == 'sequence_length':
                        shape.append(128)  # Default sequence length for BERT
                    elif dim.dim_param == 'hidden_size':
                        shape.append(768)  # Default hidden size for BERT
                    else:
                        shape.append(1)  # Default for other dynamic dimensions
                else:
                    shape.append(1)  # Default for unknown dimensions
            tensor_shapes[value_info.name] = shape
    
    # Add shapes from initializers
    for init in model.graph.initializer:
        if init.name not in tensor_shapes:
            tensor_shapes[init.name] = list(init.dims)
    
    # Add shapes for weight tensors with onnx:: prefix
    for node in model.graph.node:
        if node.op_type == "MatMul":
            for inp in node.input:
                if inp.startswith("onnx::") and inp not in tensor_shapes:
                    # For weight tensors, we need to infer the shape from the operation
                    # For BERT/DistilBERT, the weights are typically [hidden_size, hidden_size]
                    # or [hidden_size, intermediate_size] for FFN layers
                    tensor_shapes[inp] = [768, 768]  # Default to [hidden_size, hidden_size]
                    logger.debug(f"Added shape for weight tensor {inp}: {tensor_shapes[inp]}")
    
    # Debug logging for MatMul operations
    for node in model.graph.node:
        if node.op_type == "MatMul":
            logger.debug(f"MatMul node: {node.name}")
            logger.debug(f"Inputs: {node.input}")
            logger.debug(f"Outputs: {node.output}")
            for inp in node.input:
                if inp in tensor_shapes:
                    logger.debug(f"Input shape for {inp}: {tensor_shapes[inp]}")
            for out in node.output:
                if out in tensor_shapes:
                    logger.debug(f"Output shape for {out}: {tensor_shapes[out]}")
    
    return tensor_shapes

def estimate_flops(node, tensor_shapes, baseline_dtype=DEFAULT_BASELINE):
    """Estimate FLOPs for a node, relative to the specified baseline data type.
    
    Args:
        node: ONNX node to analyze
        tensor_shapes: Dictionary mapping tensor names to their shapes from ONNX Runtime
        baseline_dtype: The data type to use as baseline (default: float32)
    
    Returns:
        tuple: (estimated FLOPs, whether the operation is supported)
    """
    op = node.op_type
    flops = 0
    supported = True

    try:
        # Get shapes from ONNX Runtime
        input_shapes_list = []
        for inp in node.input:
            if inp in tensor_shapes:
                input_shapes_list.append(tensor_shapes[inp])
            else:
                logger.debug(f"Missing shape for input: {inp}")
                input_shapes_list.append([])
        
        output_shapes_list = []
        for out in node.output:
            if out in tensor_shapes:
                output_shapes_list.append(tensor_shapes[out])
            else:
                logger.debug(f"Missing shape for output: {out}")
                output_shapes_list.append([])
        
        first_out = output_shapes_list[0] if output_shapes_list else []

        # Debug logging for input shapes
        logger.debug(f"Operation: {op}")
        logger.debug(f"Input shapes: {input_shapes_list}")
        logger.debug(f"Output shapes: {output_shapes_list}")
        logger.debug(f"Input tensors: {node.input}")
        logger.debug(f"Output tensors: {node.output}")

        if op == "Gather":
            # Gather operation: FLOPs = N * D where N is the number of indices and D is the dimension size
            if len(input_shapes_list) >= 2:
                indices_shape = input_shapes_list[1]
                data_shape = input_shapes_list[0]
                if len(indices_shape) > 0 and len(data_shape) > 0:
                    # For BERT/DistilBERT, the indices are typically [batch_size, sequence_length]
                    # and the data is [vocab_size, hidden_size]
                    if len(indices_shape) == 2 and len(data_shape) == 2:
                        batch_size, seq_len = indices_shape
                        vocab_size, hidden_size = data_shape
                        base_flops = batch_size * seq_len * hidden_size
                        logger.debug(f"Gather - batch_size: {batch_size}, seq_len: {seq_len}, hidden_size: {hidden_size}")
                        logger.debug(f"Base FLOPs: {base_flops}")
                        flops = base_flops
                    else:
                        # Generic case: multiply all dimensions
                        base_flops = prod(indices_shape) * data_shape[-1]
                        logger.debug(f"Generic Gather - indices_shape: {indices_shape}, data_shape: {data_shape}")
                        logger.debug(f"Base FLOPs: {base_flops}")
                        flops = base_flops
                else:
                    supported = False
                    logger.debug(f"Gather not supported: invalid input shapes - {input_shapes_list}")

        elif op == "Erf":
            # Error function (used in GELU): FLOPs = N where N is the number of elements
            if len(input_shapes_list) > 0:
                input_shape = input_shapes_list[0]
                if len(input_shape) > 0:
                    base_flops = prod(input_shape)
                    logger.debug(f"Erf - input_shape: {input_shape}")
                    logger.debug(f"Base FLOPs: {base_flops}")
                    flops = base_flops
                else:
                    supported = False
                    logger.debug(f"Erf not supported: invalid input shape - {input_shapes_list}")

        elif op == "Where":
            # Where operation (used in attention masking): FLOPs = N where N is the number of elements
            if len(input_shapes_list) >= 3:
                condition_shape = input_shapes_list[0]
                if len(condition_shape) > 0:
                    base_flops = prod(condition_shape)
                    logger.debug(f"Where - condition_shape: {condition_shape}")
                    logger.debug(f"Base FLOPs: {base_flops}")
                    flops = base_flops
                else:
                    supported = False
                    logger.debug(f"Where not supported: invalid input shapes - {input_shapes_list}")

        elif op == "Shape":
            # Shape operation: FLOPs = 0 (just metadata operation)
            flops = 0
            supported = True

        elif op == "Equal":
            # Equal operation (used in attention masking): FLOPs = N where N is the number of elements
            if len(input_shapes_list) >= 2:
                input_shape = input_shapes_list[0]
                if len(input_shape) > 0:
                    base_flops = prod(input_shape)
                    logger.debug(f"Equal - input_shape: {input_shape}")
                    logger.debug(f"Base FLOPs: {base_flops}")
                    flops = base_flops
                else:
                    supported = False
                    logger.debug(f"Equal not supported: invalid input shapes - {input_shapes_list}")

        elif op == "MatMul":
            if len(input_shapes_list) >= 2:
                a, b = input_shapes_list[:2]
                
                # Debug logging for shapes
                logger.debug(f"MatMul node: {node.name}")
                logger.debug(f"MatMul shapes - A: {a}, B: {b}")
                
                # Handle weight tensors (typically the second input)
                if len(b) == 2 and b[0] == 768 and b[1] == 768:  # Typical BERT weight shape
                    logger.debug(f"Detected BERT weight tensor: {b}")
                    # For BERT/DistilBERT, the weights are typically [hidden_size, hidden_size]
                    # or [hidden_size, intermediate_size] for FFN layers
                    if "ffn" in node.name.lower() and "lin1" in node.name.lower():
                        # First FFN layer: [hidden_size, intermediate_size]
                        b = [768, 3072]
                    elif "ffn" in node.name.lower() and "lin2" in node.name.lower():
                        # Second FFN layer: [intermediate_size, hidden_size]
                        b = [3072, 768]
                
                # Handle different input shapes
                if len(a) == 2 and len(b) == 2:
                    # 2D matrix multiplication: A[m,k] * B[k,n] = C[m,n]
                    # FLOPs = 2 * m * k * n
                    m, k = a
                    _, n = b
                    base_flops = 2 * m * k * n
                    logger.debug(f"2D MatMul - m: {m}, k: {k}, n: {n}")
                    logger.debug(f"Base FLOPs: {base_flops}")
                elif len(a) == 3 and len(b) == 3:
                    # Batched matrix multiplication: A[b,m,k] * B[b,k,n] = C[b,m,n]
                    # FLOPs = 2 * b * m * k * n
                    b_dim, m, k = a
                    _, _, n = b
                    base_flops = 2 * b_dim * m * k * n
                    logger.debug(f"Batched MatMul - b: {b_dim}, m: {m}, k: {k}, n: {n}")
                    logger.debug(f"Base FLOPs: {base_flops}")
                elif len(a) == 3 and len(b) == 2:
                    # Batched matrix multiplication with 2D weight: A[b,m,k] * B[k,n] = C[b,m,n]
                    # FLOPs = 2 * b * m * k * n
                    b_dim, m, k = a
                    _, n = b
                    base_flops = 2 * b_dim * m * k * n
                    logger.debug(f"Batched MatMul with 2D weight - b: {b_dim}, m: {m}, k: {k}, n: {n}")
                    logger.debug(f"Base FLOPs: {base_flops}")
                elif len(a) == 2 and len(b) == 3:
                    # 2D matrix with batched input: A[m,k] * B[b,k,n] = C[b,m,n]
                    # FLOPs = 2 * b * m * k * n
                    m, k = a
                    b_dim, _, n = b
                    base_flops = 2 * b_dim * m * k * n
                    logger.debug(f"2D MatMul with batched input - b: {b_dim}, m: {m}, k: {k}, n: {n}")
                    logger.debug(f"Base FLOPs: {base_flops}")
                else:
                    # Generic case: multiply all dimensions except the last
                    # For A[...,m,k] * B[...,k,n] = C[...,m,n]
                    # FLOPs = 2 * prod(leading_dims) * m * k * n
                    leading_dims = a[:-2] if len(a) > 2 else [1]
                    m, k = a[-2:]
                    _, n = b[-2:]
                    base_flops = 2 * prod(leading_dims) * m * k * n
                    logger.debug(f"Generic MatMul - leading_dims: {leading_dims}, m: {m}, k: {k}, n: {n}")
                    logger.debug(f"Base FLOPs: {base_flops}")
                
                flops = base_flops
                logger.debug(f"Final MatMul FLOPs: {flops}")
            else:
                supported = False
                logger.debug(f"MatMul not supported: invalid input shapes - {input_shapes_list}")

        elif op == "Conv":
            if len(input_shapes_list) >= 2:
                x, w = input_shapes_list[:2]
                if len(x) == 4 and len(w) == 4 and first_out:
                    # For Conv, we need to consider both input and weight data types
                    input_dtype = None
                    weight_dtype = None
                    if node.input[0] in tensor_shapes:
                        input_dtype = tensor_shapes[node.input[0]][3]
                    if node.input[1] in tensor_shapes:
                        weight_dtype = tensor_shapes[node.input[1]][3]
                    
                    dtype_multiplier = max(
                        get_flops_multiplier(input_dtype, baseline_dtype) if input_dtype else 1.0,
                        get_flops_multiplier(weight_dtype, baseline_dtype) if weight_dtype else 1.0
                    )
                    
                    # Base FLOPs for convolution: 2 * (kernel_size) * (output_size)
                    base_flops = 2 * w[0] * w[1] * w[2] * w[3] * first_out[2] * first_out[3]
                    flops = base_flops * dtype_multiplier
                    logger.debug(f"Conv FLOPs: {flops} (base: {base_flops}, multiplier: {dtype_multiplier})")
                else:
                    supported = False
                    logger.debug(f"Conv not supported: invalid shapes - x: {x}, w: {w}, out: {first_out}")
            else:
                supported = False
                logger.debug(f"Conv not supported: insufficient inputs")

        elif op == "ConvTranspose":
            x, w = input_shapes_list[:2]
            if len(x) == 4 and len(w) == 4 and first_out:
                # Similar to Conv, consider both input and weight types
                input_dtype = None
                weight_dtype = None
                if node.input[0] in tensor_shapes:
                    input_dtype = tensor_shapes[node.input[0]][3]
                if node.input[1] in tensor_shapes:
                    weight_dtype = tensor_shapes[node.input[1]][3]
                
                dtype_multiplier = max(
                    get_flops_multiplier(input_dtype, baseline_dtype) if input_dtype else 1.0,
                    get_flops_multiplier(weight_dtype, baseline_dtype) if weight_dtype else 1.0
                )
                
                # Base FLOPs for transposed convolution
                base_flops = 2 * w[1] * w[0] * w[2] * w[3] * first_out[2] * first_out[3]
                flops = base_flops * dtype_multiplier
            else:
                supported = False

        elif op == "Gemm":
            a, b = input_shapes_list[:2]
            c = input_shapes_list[2] if len(input_shapes_list) > 2 else None
            if len(a) == 2 and len(b) == 2:
                # For matrix multiplication, consider all input types
                input_dtypes = []
                for inp in node.input[:2]:
                    if inp in tensor_shapes:
                        input_dtypes.append(tensor_shapes[inp][3])
                
                # Use the maximum multiplier to account for the most expensive operation
                dtype_multiplier = max(
                    [get_flops_multiplier(dtype, baseline_dtype) for dtype in input_dtypes] + [1.0]
                )
                
                m, k = a
                _, n = b
                # Base FLOPs for matrix multiplication: 2 * m * k * n
                base_flops = 2 * m * k * n
                if c: base_flops += m * n  # Add bias term if present
                flops = base_flops * dtype_multiplier
            else:
                supported = False

        elif op == "LayerNormalization":
            if input_shapes_list and len(input_shapes_list[0]) >= 2:
                # Get input shape
                input_shape = input_shapes_list[0]
                
                # Debug logging
                logger.debug(f"LayerNorm input shape: {input_shape}")
                
                # Calculate total elements
                total_elements = prod(input_shape)
                
                # LayerNorm operations per element:
                # 1. Mean calculation: N operations
                # 2. Variance calculation: 2N operations (subtract mean and square)
                # 3. Normalization: 3N operations (subtract mean, divide by std, scale/shift)
                base_flops = 6 * total_elements
                flops = base_flops
                
                logger.debug(f"LayerNorm FLOPs: {flops} (total_elements: {total_elements})")
            else:
                supported = False
                logger.debug(f"LayerNorm not supported: insufficient input shapes - {input_shapes_list}")

        elif op == "Constant":
            # Constant nodes are just data, no computation
            flops = 0

        elif op in ["Add", "Sub", "Mul", "Div", "Pow"]:
            # Base FLOPs for element-wise operations: number of elements
            if first_out:
                base_flops = prod(first_out)
                flops = base_flops
            else:
                supported = False
                logger.debug(f"Element-wise op not supported: no output shape")

        elif op in ["Relu", "Sigmoid", "Tanh"]:
            # Base FLOPs for activation functions: number of elements
            if first_out:
                base_flops = prod(first_out)
                flops = base_flops
            else:
                supported = False
                logger.debug(f"Activation not supported: no output shape")

        elif op.startswith("Reduce"):
            # Base FLOPs for reduction: number of input elements
            if input_shapes_list:
                base_flops = prod(input_shapes_list[0])
                flops = base_flops
            else:
                supported = False
                logger.debug(f"Reduce not supported: no input shape")

        elif op in ["Softmax"]:
            # Base FLOPs for softmax: 5 * number of elements
            if first_out:
                base_flops = 5 * prod(first_out)
                flops = base_flops
            else:
                supported = False
                logger.debug(f"Softmax not supported: no output shape")

        elif op in ["AveragePool", "MaxPool"]:
            if input_shapes_list and first_out:
                # Base FLOPs for pooling: kernel_area * output_size
                kernel_area = int(prod(input_shapes_list[0]) / prod(first_out))
                base_flops = kernel_area * prod(first_out)
                flops = base_flops
            else:
                supported = False
                logger.debug(f"Pool not supported: missing shapes - inputs: {input_shapes_list}, output: {first_out}")

        elif op in ["Reshape", "Flatten", "Transpose", "Identity", "Cast", "Slice", "Pad", "Squeeze", "Unsqueeze", "Concat", "Expand"]:
            # These operations are just data movement, no computation
            flops = 0

        else:
            supported = False

    except Exception as e:
        logger.debug(f"FLOP calc error in {node.op_type}: {e}")
        supported = False

    return flops, supported 