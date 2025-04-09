import csv
from collections import defaultdict
from utils import logger

def export_to_csv(summary, output_csv, aggr=False, total_flops=0, total_memory=0, total_params=0):
    with open(output_csv, 'w', newline='') as f:
        if aggr:
            agg = defaultdict(lambda: {"flops": 0, "memory_bytes": 0, "parameters": 0, "count": 0})
            for row in summary:
                op = row["op_type"]
                agg[op]["flops"] += row["flops"]
                agg[op]["memory_bytes"] += row["memory_bytes"]
                agg[op]["parameters"] += row["parameters"]
                agg[op]["count"] += 1

            aggr_data = []
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
            fieldnames = ["name", "op_type", "inputs", "outputs", "quantized", "flops", "flops_supported", 
                         "memory_bytes", "parameters", "input_shape", "output_shape", "flops_pct", "memory_pct", "param_pct"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)
            
        logger.info(f"Profile saved to {output_csv}")
        logger.info(f"Total FLOPs: {total_flops:,}")
        logger.info(f"Estimated memory usage: {total_memory / 1024 / 1024:.2f} MB")
        logger.info(f"Total parameters: {total_params:,}") 