import typer
from pathlib import Path
from utils import set_logging_level, configure_logging
from model_analyzer import analyze_model
from csv_exporter import export_to_csv
from flops_calculator import DEFAULT_BASELINE, FLOAT32, INT8, FLOAT16
import logging

logger = logging.getLogger(__name__)

app = typer.Typer(help="A modern CLI tool to inspect, profile, and modify ONNX models.")

def parse_baseline_dtype(dtype_str: str) -> int:
    """Parse baseline data type from string."""
    dtype_map = {
        "float32": FLOAT32,
        "int8": INT8,
        "float16": FLOAT16,
    }
    if dtype_str.lower() not in dtype_map:
        raise typer.BadParameter(f"Baseline dtype must be one of: {', '.join(dtype_map.keys())}")
    return dtype_map[dtype_str.lower()]

@app.command()
def profile(
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
    aggr: bool = typer.Option(False, "--aggr", help="Aggregate statistics by op type"),
    baseline: str = typer.Option("float32", "--baseline", help="Baseline data type for FLOPs calculation (float32, int8, float16)"),
    model_path: Path = typer.Argument(..., exists=True, help="Path to the ONNX model file"),
    output_csv: Path = typer.Option(None, "--csv", help="Export profile as CSV to the given file path"),
    log_file: Path = typer.Option(None, "--log", help="Path to the log file")
):
    """Profile an ONNX model: opset, node summary, quantization info, FLOPs, memory usage, parameter count, and tensor dimensions."""
    # Configure logging with the specified log file
    configure_logging(verbose, log_file)
    
    try:
        baseline_dtype = parse_baseline_dtype(baseline)
    except typer.BadParameter as e:
        logger.error(f"Error: {e}")
        raise typer.Exit(1)
    
    summary, total_flops, total_memory, total_params, ir_version = analyze_model(model_path, baseline_dtype)
    
    if output_csv:
        export_to_csv(summary, output_csv, aggr, total_flops, total_memory, total_params)
    
    logger.info(f"Model IR version: {ir_version}")
    logger.info(f"Total FLOPs: {total_flops}")
    logger.info(f"Total memory usage: {total_memory} bytes")
    logger.info(f"Total parameters: {total_params}")
    
    # Print summary table
    if not output_csv:
        logger.info("\nNode Summary:")
        logger.info("=" * 80)
        logger.info(f"{'Op Type':<15} {'Name':<30} {'FLOPs':<15} {'Memory':<15} {'Params':<15}")
        logger.info("-" * 80)
        for node in summary:
            logger.info(f"{node['op_type']:<15} {node['name']:<30} {node['flops']:<15} {node['memory_bytes']:<15} {node['parameters']:<15}")
    
if __name__ == "__main__":
    app() 