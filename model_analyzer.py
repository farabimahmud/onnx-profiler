import onnx
import onnx.shape_inference
from utils import get_shape
from flops_calculator import estimate_flops, DEFAULT_BASELINE, get_tensor_shapes
from memory_analyzer import analyze_memory_usage, analyze_parameters
import logging

logger = logging.getLogger(__name__)

def analyze_model(model_path, baseline_dtype=DEFAULT_BASELINE):
    """Analyze an ONNX model and compute various metrics.
    
    Args:
        model_path: Path to the ONNX model file
        baseline_dtype: The data type to use as baseline for FLOPs calculation
                       (default: float32)
    
    Returns:
        tuple: (summary, total_flops, total_memory, total_params, ir_version)
    """
    model = onnx.load(model_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
        logger.info("Shape inference completed successfully.")
    except Exception as e:
        logger.warning(f"Shape inference failed: {e}")

    # Get tensor shapes from ONNX Runtime
    tensor_shapes = get_tensor_shapes(model_path)
    logger.info("Got tensor shapes from ONNX Runtime")

    graph = model.graph
    nodes = graph.node
    initializers = {init.name: init for init in graph.initializer}
    
    summary = []
    total_flops = 0
    total_memory = 0
    total_params = 0
    unsupported_count = 0

    for node in nodes:
        is_quant = any(qop in node.op_type for qop in ["QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"])
        flops, supported = estimate_flops(node, tensor_shapes, baseline_dtype)
        if not supported:
            unsupported_count += 1
        total_flops += flops

        mem_size = analyze_memory_usage(node, initializers, tensor_shapes)
        total_memory += mem_size

        param_count = analyze_parameters(node, initializers)
        total_params += param_count

        # Get input shapes from ONNX Runtime
        input_shapes = []
        for inp in node.input:
            if inp in tensor_shapes:
                input_shapes.append(tensor_shapes[inp])
        
        # Get output shapes from ONNX Runtime
        output_shapes = []
        for out in node.output:
            if out in tensor_shapes:
                output_shapes.append(tensor_shapes[out])

        # For input shape, use the first non-empty shape if available
        input_shape = str(input_shapes[0]) if input_shapes else "unknown"
        
        # For output shape, use the first non-empty shape if available
        output_shape = str(output_shapes[0]) if output_shapes else "unknown"

        flops_pct = (flops / total_flops * 100) if total_flops > 0 else 0
        mem_pct = (mem_size / total_memory * 100) if total_memory > 0 else 0
        param_pct = (param_count / total_params * 100) if total_params > 0 else 0

        summary.append({
            "flops_pct": round(flops_pct, 2),
            "memory_pct": round(mem_pct, 2),
            "param_pct": round(param_pct, 2),
            "name": node.name,
            "op_type": node.op_type,
            "inputs": ",".join(node.input),
            "outputs": ",".join(node.output),
            "quantized": "Yes" if is_quant else "No",
            "flops": flops,
            "flops_supported": "Yes" if supported else "No",
            "memory_bytes": mem_size,
            "parameters": param_count,
            "input_shape": input_shape,
            "output_shape": output_shape
        })

    return summary, total_flops, total_memory, total_params, model.ir_version 