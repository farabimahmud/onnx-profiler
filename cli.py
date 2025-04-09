import typer
from pathlib import Path
from utils import set_logging_level
from model_analyzer import analyze_model
from csv_exporter import export_to_csv

app = typer.Typer(help="A modern CLI tool to inspect, profile, and modify ONNX models.")

@app.command()
def profile(
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
    aggr: bool = typer.Option(False, "--aggr", help="Aggregate statistics by op type"),
    model_path: Path = typer.Argument(..., exists=True, help="Path to the ONNX model file"),
    output_csv: Path = typer.Option(None, "--csv", help="Export profile as CSV to the given file path")
):
    """Profile an ONNX model: opset, node summary, quantization info, FLOPs, memory usage, parameter count, and tensor dimensions."""
    set_logging_level(verbose)
    
    summary, total_flops, total_memory, total_params, ir_version = analyze_model(model_path)
    
    if output_csv:
        export_to_csv(summary, output_csv, aggr, total_flops, total_memory, total_params)
    
    print(f"Model IR version: {ir_version}")

if __name__ == "__main__":
    app() 