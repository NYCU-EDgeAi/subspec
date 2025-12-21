
import sys
import os
import subprocess
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run.core.registry import ModelRegistry
from run.core.presets import register_presets

def run_pipeline_test(method, device="cuda:0"):
    print(f"Testing pipeline: {method} on {device}...")
    cmd = [
        sys.executable, "-m", "run.main",
        "--method", method,
        "--warmup-iter", "0",
        "--device", device,
        "run-test"
    ]
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"[PASS] {method}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {method}")
        print(f"Error output:\n{e.stderr}")
        return False, e.stderr

def main():
    parser = argparse.ArgumentParser(description="Test all registered pipelines.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run tests on")
    parser.add_argument("--methods", type=str, nargs="+", help="Specific methods to test (default: all)")
    args = parser.parse_args()

    # Ensure presets are registered to get the list
    register_presets()
    
    if args.methods:
        methods_to_test = args.methods
    else:
        # Get all registered methods
        methods_to_test = list(ModelRegistry._registry.keys())
        # Filter out some that might need specific complex setup if desired, 
        # but for now we try all.
    
    print(f"Starting tests for methods: {methods_to_test}")
    
    results = {}
    for method in methods_to_test:
        success, output = run_pipeline_test(method, args.device)
        results[method] = success
        
    print("\n" + "="*30)
    print("Test Summary")
    print("="*30)
    all_passed = True
    for method, success in results.items():
        status = "PASS" if success else "FAIL"
        if not success:
            all_passed = False
        print(f"{method}: {status}")
        
    if not all_passed:
        sys.exit(1)
    else:
        print("\nAll pipelines passed!")

if __name__ == "__main__":
    main()
