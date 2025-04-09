from math import prod
from utils import logger

def estimate_flops(node, input_shapes, output_shapes):
    op = node.op_type
    flops = 0
    supported = True

    def shape(name):
        try:
            return eval(name)
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