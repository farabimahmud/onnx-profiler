import typer
from pathlib import Path
import onnx
import onnx.shape_inference
import csv
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(help="A modern CLI tool to inspect, profile, and modify ONNX models.")

@app.command()
def profile(
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
    aggr: bool = typer.Option(False, "--aggr", help="Aggregate statistics by op type"),
    model_path: Path = typer.Argument(..., exists=True, help="Path to the ONNX model file"),
    output_csv: Path = typer.Option(None, "--csv", help="Export profile as CSV to the given file path")
):
    """Profile an ONNX model: opset, node summary, quantization info, FLOPs, memory usage, parameter count, and tensor dimensions."""
    import functools

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    model = onnx.load(model_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
        logger.info("Shape inference completed successfully.")
    except Exception as e:
        logger.warning(f"Shape inference failed: {e}")

    graph = model.graph
    nodes = graph.node
    initializers = {init.name: init for init in graph.initializer}
    summary = []
    total_flops = 0
    total_memory = 0
    total_params = 0
    unsupported_count = 0

    input_shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in graph.input}
    value_info_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.output) + list(graph.input)}
    value_info_shapes = {
        vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in value_info_map.values()
    }

    def get_shape(name):
        shape = input_shapes.get(name) or value_info_shapes.get(name, [])
        if shape:
            return str(shape)
        return "unknown"

    def get_dtype_size(name):
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

    def estimate_flops(node):
        from math import prod
        op = node.op_type
        flops = 0
        supported = True

        def shape(name):
            try:
                return eval(get_shape(name))
            except:
                return []

        try:
            input_shapes = [shape(inp) for inp in node.input if shape(inp)]
            output_shapes = [shape(out) for out in node.output if shape(out)]
            first_out = output_shapes[0] if output_shapes else []

            if op == "Conv":
                x, w = input_shapes[:2]
                if len(x) == 4 and len(w) == 4 and first_out:
                    flops = 2 * w[0] * w[1] * w[2] * w[3] * first_out[2] * first_out[3]
                else:
                    supported = False

            elif op == "ConvTranspose":
                x, w = input_shapes[:2]
                if len(x) == 4 and len(w) == 4 and first_out:
                    flops = 2 * w[1] * w[0] * w[2] * w[3] * first_out[2] * first_out[3]
                else:
                    supported = False

            elif op == "Gemm":
                a, b = input_shapes[:2]
                c = input_shapes[2] if len(input_shapes) > 2 else None
                if len(a) == 2 and len(b) == 2:
                    m, k = a
                    _, n = b
                    flops = 2 * m * k * n
                    if c: flops += m * n
                else:
                    supported = False

            elif op == "MatMul":
                a, b = input_shapes[:2]
                if len(a) == 2 and len(b) == 2:
                    flops = 2 * a[0] * a[1] * b[1]
                elif len(a) == 3 and len(b) == 3:
                    flops = 2 * a[0] * a[1] * b[2]
                else:
                    supported = False

            elif op in ["Add", "Sub", "Mul", "Div", "Pow", "Relu", "Sigmoid", "Tanh"]:
                flops = prod(first_out)

            elif op.startswith("Reduce"):
                flops = prod(input_shapes[0])

            elif op in ["Softmax"]:
                flops = 5 * prod(first_out)

            elif op in ["AveragePool", "MaxPool"]:
                if input_shapes and first_out:
                    kernel_area = int(prod(input_shapes[0]) / prod(first_out))
                    flops = kernel_area * prod(first_out)

            elif op in ["Reshape", "Flatten", "Transpose", "Identity", "Cast", "Slice", "Pad", "Squeeze", "Unsqueeze", "Concat", "Expand"]:
                flops = 0

            else:
                supported = False

        except Exception as e:
            logger.debug(f"FLOP calc error in {node.op_type}: {e}")
            supported = False

        return flops, supported

    for node in nodes:
        is_quant = any(qop in node.op_type for qop in ["QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"])
        flops, supported = estimate_flops(node)
        if not supported:
            unsupported_count += 1
        total_flops += flops

        mem_size = 0
        for out in node.output:
            shape = eval(get_shape(out)) if get_shape(out) != "unknown" else []
            if shape:
                size = get_dtype_size(out) * max(1, functools.reduce(lambda a,b: a*b, shape, 1))
                mem_size += size
                total_memory += size

        param_count = 0
        for inp in node.input:
            if inp in initializers:
                tensor = initializers[inp]
                size = 1
                for d in tensor.dims:
                    size *= d
                param_count += size
        total_params += param_count

        input_dims = [get_shape(inp) for inp in node.input if get_shape(inp) != "unknown"]
        output_dims = [get_shape(out) for out in node.output if get_shape(out) != "unknown"]
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

    fieldnames = ["name", "op_type", "inputs", "outputs", "quantized", "flops", "flops_supported", "memory_bytes", "parameters", "input_shape", "output_shape", "flops_pct", "memory_pct", "param_pct"]

    if aggr:
            from collections import defaultdict
            agg = defaultdict(lambda: {"flops": 0, "memory_bytes": 0, "parameters": 0, "count": 0})
            for row in summary:
                op = row["op_type"]
                agg[op]["flops"] += row["flops"]
                agg[op]["memory_bytes"] += row["memory_bytes"]
                agg[op]["parameters"] += row["parameters"]
                agg[op]["count"] += 1

            aggr_data = []  # list of aggregated rows per op_type
            for op, stats in agg.items():
                aggr_data.append({
                    "op_type": op,
                    "count": stats["count"],
                    "flops": stats["flops"],
                    "memory_bytes": stats["memory_bytes"],
                    "parameters": stats["parameters"],
                    "flops_pct": round(stats["flops"] / total_flops * 100, 2) if total_flops > 0 else 0,
                    "memory_pct": round(stats["memory_bytes"] / total_memory * 100, 2) if total_memory > 0 else 0,
                    "param_pct": round(stats["parameters"] / total_params * 100, 2) if total_params > 0 else 0,
                })

            writer = csv.DictWriter(f, fieldnames=["op_type", "count", "flops", "flops_pct", "memory_bytes", "memory_pct", "parameters", "param_pct"])
            writer.writeheader()
            writer.writerows(aggr_data)
            writer.writerow({
                "op_type": "TOTAL",
                "count": sum(d["count"] for d in aggr_data),
                "flops": total_flops,
                "memory_bytes": total_memory,
                "parameters": total_params,
                "flops_pct": 100.0,
                "memory_pct": 100.0,
                "param_pct": 100.0,
            })
            
    else:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
        logger.info(f"Profile saved to {output_csv}")
        logger.info(f"Total FLOPs: {total_flops:,}")
        logger.info(f"Estimated memory usage: {total_memory / 1024 / 1024:.2f} MB")
        logger.info(f"Total parameters: {total_params:,}")


    logger.info(f"Model IR version: {model.ir_version}")
    logger.info(f"Opset imports: {model.opset_import}")
    logger.info(f"Inputs: {[i.name for i in graph.input]}")
    logger.info(f"Outputs: {[o.name for o in graph.output]}")
    logger.info(f"Number of nodes: {len(nodes)}")
    logger.info(f"Total FLOPs: {total_flops:,}")
    logger.info(f"Estimated memory usage: {total_memory / 1024 / 1024:.2f} MB")
    logger.info(f"Total parameters: {total_params:,}")
    for item in summary[:10]:
        logger.info(item)

    if unsupported_count > 0:
        logger.warning(f"Warning: {unsupported_count} node(s) could not be profiled for FLOPs.")

if __name__ == "__main__":
    app()
