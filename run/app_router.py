import torch
import logging
import time
import typer

def run_app(builder):
    app = typer.Typer()

    @app.command()
    def run_test():
        """
        Example subcommand for a test run.
        Usage:
            python custom.py run-test
        """
        from .pipelines.run_test import main as main_run_test
        # torch.cuda.memory._record_memory_history()
        
        main_run_test(builder)
        
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)

    @app.command()
    def run_grid_search(t: str, d: str, k: str, max_samples: int = None):
        """
        Example subcommand for grid search.
        Usage:
            python custom.py run-grid-search --t=0.3,0.4 --d=4,8,16,32 --k=8 --max-samples=10
        """
        from .pipelines.run_grid_search import main as main_run_grid_search
        main_run_grid_search(builder, temperature_values=t, max_depth_values=d, topk_len_values=k, max_samples=max_samples)
        
    @app.command()
    def run_benchmark(benchmarks: str = None, max_samples: int = None):
        """
        Example subcommand for benchmarking.
        Usage: 
            python custom.py run-benchmark --bench-name=mt-bench
        """
        from .pipelines.run_benchmark import main as main_run_benchmark
        main_run_benchmark(builder, benchmarks=benchmarks, max_samples=max_samples)

    @app.command()
    def run_gradio():
        """
        Example subcommand for launching a Gradio demo.
        Usage:
            python custom.py run-gradio
        """
        print(f"Running Gradio")

    app()