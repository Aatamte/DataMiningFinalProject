"""vLLM judge server management.

Commands:
    uv run python scripts/run_vllm.py --list                          # List available models
    uv run python scripts/run_vllm.py --download qwen3-8b-awq         # Download model
    uv run python scripts/run_vllm.py --server                        # Start server
    uv run python scripts/run_vllm.py --stop                          # Stop server

Models are downloaded to models/ directory and loaded from there.
Set JUDGE_MODEL in .env to configure which model to use.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Server Configuration - EDIT THESE
# =============================================================================

GPU_MEMORY_UTILIZATION = 0.3   # GPU memory % (~7.2 GB, leaves room for training)
MAX_MODEL_LEN = 8192           # Context window (full context for long prompts)
MAX_NUM_SEQS = 10              # Max concurrent sequences (handles parallel judge calls)
CHUNKED_PREFILL = True         # Better for long prompts
TENSOR_PARALLEL_SIZE = 1       # Multi-GPU support
DTYPE = "auto"                 # float16/bfloat16/auto
KV_CACHE_DTYPE = "auto"        # auto (FP8 not supported on RTX 3090 with FlashInfer)
CPU_OFFLOAD_GB = 0             # No CPU offload
DISABLE_LOG_REQUESTS = True    # Less overhead


# =============================================================================
# Supported Models - short name -> HuggingFace path
# =============================================================================

MODELS = {
    # Qwen3 8B variants
    "qwen3-8b":       "Qwen/Qwen3-8B",
    "qwen3-8b-awq":   "Qwen/Qwen3-8B-AWQ",
    "qwen3-8b-fp8":   "Qwen/Qwen3-8B-FP8",

    # Qwen3 4B variants
    "qwen3-4b":       "Qwen/Qwen3-4B",
    "qwen3-4b-awq":   "Qwen/Qwen3-4B-AWQ",

    # Qwen3 smaller
    "qwen3-1.7b":     "Qwen/Qwen3-1.7B",
    "qwen3-0.6b":     "Qwen/Qwen3-0.6B",
}


# =============================================================================
# Paths
# =============================================================================

MODELS_DIR = Path("models")    # Local model storage directory
PID_FILE = Path("tools/vllm/server.pid")


# =============================================================================
# Helpers
# =============================================================================

def get_port_from_env() -> int:
    """Extract port from JUDGE_BASE_URL in .env."""
    base_url = os.environ.get("JUDGE_BASE_URL", "http://localhost:1234/v1")
    try:
        from urllib.parse import urlparse
        return urlparse(base_url).port or 1234
    except:
        return 1234


def get_judge_model() -> str:
    """Get JUDGE_MODEL from .env."""
    return os.environ.get("JUDGE_MODEL", "")


def resolve_model(name: str, check_local: bool = True) -> tuple[str, bool]:
    """Resolve model name to path.

    Args:
        name: Short name (qwen3-8b) or HF path (Qwen/Qwen3-8B)
        check_local: Whether to check for local copy first

    Returns:
        (model_path, is_local) - path to use and whether it's local
    """
    # Determine HF model ID
    if "/" in name:
        hf_id = name
    else:
        key = name.lower()
        if key not in MODELS:
            print(f"ERROR: Unknown model '{name}'")
            print(f"\nSupported short names:")
            for short, full in MODELS.items():
                print(f"  {short:15} -> {full}")
            print(f"\nOr use full HuggingFace path (e.g., Qwen/Qwen3-8B-AWQ)")
            return "", False
        hf_id = MODELS[key]

    # Check for local copy
    if check_local:
        # Local dir uses model name (e.g., models/Qwen3-8B-AWQ)
        local_name = hf_id.split("/")[-1]
        local_path = MODELS_DIR / local_name
        if local_path.exists() and (local_path / "config.json").exists():
            return str(local_path), True

    return hf_id, False


def check_server(port: int) -> bool:
    """Check if server is responding."""
    try:
        response = urlopen(f"http://localhost:{port}/health", timeout=2)
        return response.status == 200
    except:
        return False


# =============================================================================
# Commands
# =============================================================================

def cmd_download(model_name: str):
    """Download a model to local models/ directory."""
    hf_id, is_local = resolve_model(model_name, check_local=False)
    if not hf_id:
        return False

    # Destination directory
    local_name = hf_id.split("/")[-1]
    local_path = MODELS_DIR / local_name

    if local_path.exists() and (local_path / "config.json").exists():
        print(f"Model already exists: {local_path}")
        return True

    print(f"Downloading: {hf_id}")
    print(f"Destination: {local_path}\n")

    # Create models dir
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Use huggingface-cli to download to local dir
    result = subprocess.run([
        "huggingface-cli", "download", hf_id,
        "--local-dir", str(local_path),
        "--local-dir-use-symlinks", "False",
    ])

    if result.returncode == 0:
        print(f"\nModel downloaded to: {local_path}")
        print(f"Set in .env: JUDGE_MODEL={model_name}")
    else:
        print(f"\nFailed to download. Install huggingface-cli:")
        print("  pip install huggingface-hub")

    return result.returncode == 0


def cmd_list():
    """List available models and their status."""
    print("=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)
    print(f"{'Short Name':<15} {'HuggingFace ID':<25} {'Status':<10}")
    print("-" * 60)

    for short, hf_id in MODELS.items():
        local_name = hf_id.split("/")[-1]
        local_path = MODELS_DIR / local_name
        if local_path.exists() and (local_path / "config.json").exists():
            status = "LOCAL"
        else:
            status = "remote"
        print(f"{short:<15} {hf_id:<25} {status:<10}")

    print("-" * 60)
    print(f"Local models dir: {MODELS_DIR.absolute()}")


def cmd_stop():
    """Stop the running server."""
    if not PID_FILE.exists():
        print("No server PID file found.")
        return False

    try:
        pid = int(PID_FILE.read_text().strip())
        print(f"Stopping server (PID: {pid})...")
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        PID_FILE.unlink()
        print("Server stopped.")
        return True
    except Exception as e:
        print(f"Failed to stop: {e}")
        PID_FILE.unlink(missing_ok=True)
        return False


def cmd_server(model_name: str, port: int, foreground: bool):
    """Start the vLLM server."""
    model_path, is_local = resolve_model(model_name)
    if not model_path:
        return False

    # Check vllm installed
    try:
        import vllm
    except ImportError:
        print("ERROR: vllm not installed.")
        print("Install it separately (to avoid dependency conflicts):")
        print("  pip install vllm")
        return False

    # Check if already running
    if check_server(port):
        print(f"Server already running on port {port}")
        print("Use --stop first")
        return False

    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--dtype", DTYPE,
        "--kv-cache-dtype", KV_CACHE_DTYPE,
        "--trust-remote-code",
    ]

    if CPU_OFFLOAD_GB > 0:
        cmd.extend(["--cpu-offload-gb", str(CPU_OFFLOAD_GB)])
    if CHUNKED_PREFILL:
        cmd.append("--enable-chunked-prefill")
    if TENSOR_PARALLEL_SIZE > 1:
        cmd.extend(["--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE)])
    if DISABLE_LOG_REQUESTS:
        cmd.append("--disable-log-requests")

    print("=" * 60)
    print("VLLM SERVER")
    print("=" * 60)
    source = "LOCAL" if is_local else "HuggingFace"
    print(f"Model: {model_path} ({source})")
    print(f"Port: {port}")
    print(f"GPU Memory: {GPU_MEMORY_UTILIZATION * 100:.0f}%")
    print(f"Max Context: {MAX_MODEL_LEN}")
    print(f"Max Seqs: {MAX_NUM_SEQS}")
    print(f"KV Cache: {KV_CACHE_DTYPE}")
    print("=" * 60)
    print()

    # Start process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Save PID
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(process.pid))

    print("Loading model...\n")

    import threading
    server_ready = False
    start_time = time.time()
    timeout = 300

    def read_output():
        nonlocal server_ready
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"  {line.rstrip()}")
                if "Uvicorn running" in line or "Application startup complete" in line:
                    server_ready = True

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    while time.time() - start_time < timeout:
        if server_ready or check_server(port):
            print(f"\nServer ready! http://localhost:{port}/v1")

            if foreground:
                print("\nPress Ctrl+C to stop.\n")
                try:
                    reader.join()
                except KeyboardInterrupt:
                    print("\nStopping...")
                    process.terminate()
                    PID_FILE.unlink(missing_ok=True)
            else:
                print(f"PID: {process.pid}")
                print(f"Stop: uv run python scripts/run_vllm.py --stop")
            return True

        if process.poll() is not None:
            print("\nERROR: Server died.")
            return False

        time.sleep(0.5)

    print(f"\nERROR: Timeout ({timeout}s)")
    process.terminate()
    return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="vLLM judge server management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a model first
  %(prog)s --download Qwen/Qwen3-8B-AWQ

  # Start server (uses JUDGE_MODEL from .env)
  %(prog)s --server

  # Start with specific model
  %(prog)s --server --model Qwen/Qwen3-8B-AWQ

  # Stop server
  %(prog)s --stop
""",
    )

    # Commands
    parser.add_argument("--list", action="store_true",
        help="List available models")
    parser.add_argument("--download", metavar="MODEL",
        help="Download model to local models/ directory")
    parser.add_argument("--server", action="store_true",
        help="Start the vLLM server")
    parser.add_argument("--stop", action="store_true",
        help="Stop the running server")

    # Options
    parser.add_argument("--model", "-m",
        help="Model ID (default: JUDGE_MODEL from .env)")
    parser.add_argument("--port", "-p", type=int,
        help="Port (default: from JUDGE_BASE_URL in .env)")
    parser.add_argument("--foreground", "-f", action="store_true",
        help="Run in foreground")

    args = parser.parse_args()

    # Must specify a command
    if not (args.list or args.download or args.server or args.stop):
        parser.print_help()
        print("\nError: Specify --list, --download, --server, or --stop")
        sys.exit(1)

    # Handle commands
    if args.list:
        cmd_list()
        return

    if args.stop:
        cmd_stop()
        return

    if args.download:
        cmd_download(args.download)
        return

    if args.server:
        model_id = args.model or get_judge_model()
        if not model_id:
            print("ERROR: No model specified.")
            print("Use --model or set JUDGE_MODEL in .env")
            sys.exit(1)

        port = args.port or get_port_from_env()
        success = cmd_server(model_id, port, args.foreground)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
