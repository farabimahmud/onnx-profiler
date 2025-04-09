import onnx
import onnx.shape_inference
from utils import logger, get_shape
from flops_calculator import estimate_flops
from memory_analyzer import analyze_memory_usage, analyze_parameters

def analyze_model(model_path):
    model = onnx.load(model_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
        logger.info("Shape inference completed successfully.")
    except Exception as e:
        logger.warning(f"Shape inference failed: {e}")

    graph = model.graph
    nodes = graph.node
    initializers = {init.name: init for init in graph.initializer}
    
    input_shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in graph.input}
    value_info_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.output) + list(graph.input)}
    value_info_shapes = {
        vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in value_info_map.values()
    }

    summary = []
    total_flops = 0
    total_memory = 0
    total_params = 0
    unsupported_count = 0

    for node in nodes:
        is_quant = any(qop in node.op_type for qop in ["QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"])
        flops, supported = estimate_flops(node, input_shapes, value_info_shapes)
        if not supported:
            unsupported_count += 1
        total_flops += flops

        mem_size = analyze_memory_usage(node, initializers, value_info_map, input_shapes, value_info_shapes)
        total_memory += mem_size

        param_count = analyze_parameters(node, initializers)
        total_params += param_count

        input_dims = [get_shape(inp, input_shapes, value_info_shapes) for inp in node.input if get_shape(inp, input_shapes, value_info_shapes) != "unknown"]
        output_dims = [get_shape(out, input_shapes, value_info_shapes) for out in node.output if get_shape(out, input_shapes, value_info_shapes) != "unknown"]
        input_shape = input_dims[0] if input_dims and all(s == input_dims[0] for s in input_dims) else "varied"
        output_shape = output_dims[0] if output_dims and all(s == output_dims[0] for s in output_dims) else "varied"

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