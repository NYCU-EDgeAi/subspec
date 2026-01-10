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
        from run.pipelines.run_test import main as main_run_test
        # torch.cuda.memory._record_memory_history()
        
        main_run_test(builder)
        
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
    
    @app.command()
    def run_agent_test():
        """
        Example subcommand for a test run.
        Usage:
            python custom.py run-test
        """
        from run.pipelines.run_agent_test import main as run_agent_test
        run_agent_test(builder)

    @app.command()
    def run_grid_search(
        t: str,
        d: str,
        k: str,
        e: str = typer.Option(None, help="Comma-separated lossy_threshold values (e.g. 0.1,0.2,0.3)"),
        w: str = typer.Option(None, help="Comma-separated lossy_window_size values (e.g. 4,6,8)"),
        max_samples: int = None,
    ):
        """
        Example subcommand for grid search.
        Usage:
            python custom.py run-grid-search --t=0.3,0.4 --d=4,8,16,32 --k=8 --max-samples=10
        """
        from run.pipelines.run_grid_search import main as main_run_grid_search
        main_run_grid_search(
            builder,
            temperature_values=t,
            max_depth_values=d,
            topk_len_values=k,
            lossy_threshold_values=e,
            lossy_window_size_values=w,
            max_samples=max_samples,
        )
        
    @app.command()
    def run_benchmark(benchmarks: str = None, max_samples: int = None):
        """
        Example subcommand for benchmarking.
        Usage: 
            python custom.py run-benchmark --bench-name=mt-bench
        """
        from run.pipelines.run_benchmark import main as main_run_benchmark
        main_run_benchmark(builder, benchmarks=benchmarks, max_samples=max_samples)
        
    @app.command()
    def run_benchmark_acc(benchmarks: str = None, max_samples: int = None):
        """
        Example subcommand for benchmarking.
        Usage: 
            python custom.py run-benchmark --bench-name=mt-bench
        """
        from run.pipelines.run_benchmark_acc import main as main_run_benchmark_acc
        main_run_benchmark_acc(builder, benchmarks=benchmarks, max_samples=max_samples)

    @app.command()
    def run_benchmark_agent(benchmarks: str = None, max_samples: int = None):
        """
        Example subcommand for benchmarking.
        Usage: 
            python custom.py run-benchmark --bench-name=mt-bench
        """
        from run.pipelines.run_benchmark_agent import main as main_run_benchmark_agent
        main_run_benchmark_agent(builder, benchmarks=benchmarks, max_samples=max_samples)

    @app.command()
    def run_gradio(
        host: str = typer.Option("127.0.0.1", help="Host/interface to bind the Gradio server"),
        port: int = typer.Option(7860, help="Port to bind the Gradio server"),
        share: bool = typer.Option(False, help="Create a public Gradio share link"),
    ):
        """Launch a Gradio chat playground.

        Usage:
            python -m run.main --method subspec_sd run-gradio --host 0.0.0.0 --port 7860
        """
        from run.pipelines.run_gradio import main as run_gradio_main
        run_gradio_main(builder, host=host, port=port, share=share)

    @app.command()
    def run_api(
        host: str = typer.Option("0.0.0.0", help="Host to bind the API server"),
        port: int = typer.Option(8000, help="Port to bind the API server"),
        workers: int = typer.Option(1, help="Number of Uvicorn workers (use 1 for GPU models)"),
        log_level: str = typer.Option("info", help="Uvicorn log level"),
    ):
        """Launch an OpenAI-compatible HTTP server.

        Usage:
            python -m run.main --method subspec_sd run-api --host 0.0.0.0 --port 8000
        """
        from run.pipelines.run_api import main as run_api_main
        run_api_main(builder, host=host, port=port, workers=workers, log_level=log_level)

    app()