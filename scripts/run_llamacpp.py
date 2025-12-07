"""Run the llama.cpp judge server.

This script handles everything:
1. Downloads llama.cpp if not present
2. Downloads the GGUF model (from JUDGE_MODEL in .env) if not present
3. Starts the server

Usage:
    uv run python scripts/run_llamacpp.py           # Start server
    uv run python scripts/run_llamacpp.py --stop    # Stop server
    uv run python scripts/run_llamacpp.py -f        # Run in foreground
"""

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError

# Load .env
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Judge Server Configuration - EDIT THESE
# =============================================================================

# Number of parallel requests to handle
# Higher = better throughput for batch judging, but more memory
# Recommended: 4-8 for judge workload
PARALLEL_REQUESTS = 4

# Context window size (in tokens)
# Judge prompts with full trajectories can be 2-4k tokens
# Recommended: 4096-8192
CONTEXT_SIZE = 4096

# Batch size for prompt processing
# Higher = faster prompt processing, but more memory
# Recommended: 512-2048
BATCH_SIZE = 512

# GPU layers to offload (99 = all layers on GPU)
# Lower this if you get OOM errors
GPU_LAYERS = 99

# Enable continuous batching (key for throughput)
CONTINUOUS_BATCHING = True

# Enable flash attention (faster, less memory)
FLASH_ATTENTION = True

# Lock model in memory (prevent swapping, recommended)
MLOCK = True


# =============================================================================
# Download Configuration - Usually don't need to change
# =============================================================================

LLAMACPP_VERSION = "b4722"
LLAMACPP_RELEASE_URL = f"https://github.com/ggerganov/llama.cpp/releases/download/{LLAMACPP_VERSION}"

# Windows releases
LLAMACPP_WINDOWS_CUDA = f"llama-{LLAMACPP_VERSION}-bin-win-cuda-cu12.2.0-x64.zip"
LLAMACPP_WINDOWS_CPU = f"llama-{LLAMACPP_VERSION}-bin-win-avx2-x64.zip"

# Linux releases
LLAMACPP_LINUX_CUDA = f"llama-{LLAMACPP_VERSION}-bin-ubuntu-x64-cuda-cu12.2.0.tar.gz"
LLAMACPP_LINUX_CPU = f"llama-{LLAMACPP_VERSION}-bin-ubuntu-x64.tar.gz"

# Model mappings - JUDGE_MODEL name -> GGUF URL
MODEL_GGUF_URLS = {
    "qwen3-8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/qwen3-8b-q4_k_m.gguf",
    "qwen3-4b": "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/qwen3-4b-q4_k_m.gguf",
    "qwen3-1.7b": "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/qwen3-1.7b-q4_k_m.gguf",
    "qwen2.5-7b": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
    "qwen2.5-3b": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
}

DEFAULT_INSTALL_DIR = Path("tools/llamacpp")
PID_FILE = DEFAULT_INSTALL_DIR / "server.pid"


# =============================================================================
# Environment helpers
# =============================================================================

def get_port_from_env() -> int:
    """Extract port from JUDGE_BASE_URL in .env."""
    base_url = os.environ.get("JUDGE_BASE_URL", "http://localhost:1234/v1")
    try:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        return parsed.port or 1234
    except:
        return 1234


def get_judge_model() -> str:
    """Get JUDGE_MODEL from .env."""
    return os.environ.get("JUDGE_MODEL", "qwen3-8b").lower()


def detect_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# =============================================================================
# Download helpers
# =============================================================================

def download_with_progress(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress indicator."""
    print(f"  {desc}...")
    print(f"    URL: {url}")
    print(f"    Dest: {dest}")
    sys.stdout.flush()

    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r    Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        elif block_num % 100 == 0:
            # Unknown size - show blocks downloaded
            mb = (block_num * block_size) / (1024 * 1024)
            print(f"\r    Downloaded: {mb:.1f} MB", end="", flush=True)

    try:
        urlretrieve(url, dest, reporthook=progress_hook)
        print()  # Newline after progress
        return True
    except Exception as e:
        print(f"\n    ERROR: Download failed: {e}")
        return False


def ensure_llamacpp(install_dir: Path, use_cuda: bool) -> Path | None:
    """Download llama.cpp if not present. Returns path to server exe."""
    is_windows = platform.system() == "Windows"
    server_name = "llama-server.exe" if is_windows else "llama-server"
    server_exe = install_dir / server_name

    if server_exe.exists():
        return server_exe

    print("\n[Setup] llama.cpp not found, downloading...")
    install_dir.mkdir(parents=True, exist_ok=True)

    # Select release based on platform
    if is_windows:
        release_file = LLAMACPP_WINDOWS_CUDA if use_cuda else LLAMACPP_WINDOWS_CPU
    else:
        release_file = LLAMACPP_LINUX_CUDA if use_cuda else LLAMACPP_LINUX_CPU

    archive_path = install_dir / release_file
    url = f"{LLAMACPP_RELEASE_URL}/{release_file}"

    # Download
    if not archive_path.exists():
        if not download_with_progress(url, archive_path, f"Downloading llama.cpp ({'CUDA' if use_cuda else 'CPU'})"):
            return None

    # Extract
    print("  Extracting...")
    sys.stdout.flush()
    try:
        if release_file.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(install_dir)
        elif release_file.endswith(".tar.gz"):
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(install_dir)
        else:
            print(f"  ERROR: Unknown archive format: {release_file}")
            return None
    except Exception as e:
        print(f"  ERROR: Failed to extract: {e}")
        return None

    # List what was extracted
    print(f"  Extracted to: {install_dir}")
    print(f"  Looking for: {server_name}")

    # Find server exe (may be nested in subdirectory)
    matches = list(install_dir.rglob(server_name))
    print(f"  Found {len(matches)} matches: {[str(m) for m in matches]}")

    for match in matches:
        if match.parent != install_dir:
            print(f"  Moving {match} -> {server_exe}")
            shutil.move(str(match), str(server_exe))
        # Make executable on Linux
        if not is_windows:
            server_exe.chmod(0o755)
            print(f"  Made executable: {server_exe}")
        return server_exe

    print(f"  ERROR: {server_name} not found in archive")
    print(f"  Contents of {install_dir}:")
    for f in install_dir.iterdir():
        print(f"    {f.name}")
    return None


def ensure_model(install_dir: Path, judge_model: str) -> Path | None:
    """Download model if not present. Returns path to model file."""
    models_dir = install_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if model already exists
    for gguf in models_dir.glob("*.gguf"):
        if judge_model.replace("-", "").replace(".", "") in gguf.name.replace("-", "").replace(".", "").lower():
            print(f"  Model found: {gguf.name}")
            return gguf

    # Need to download
    if judge_model not in MODEL_GGUF_URLS:
        print(f"\n  WARNING: JUDGE_MODEL '{judge_model}' not in known models.")
        print(f"  Known: {', '.join(MODEL_GGUF_URLS.keys())}")
        print(f"  Using default: qwen3-8b")
        judge_model = "qwen3-8b"

    url = MODEL_GGUF_URLS[judge_model]
    model_name = url.split("/")[-1]
    model_path = models_dir / model_name

    print(f"\n[Setup] Model not found, downloading {judge_model}...")
    if not download_with_progress(url, model_path, f"Downloading {model_name}"):
        return None

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")
    return model_path


# =============================================================================
# Server management
# =============================================================================

def check_server(port: int) -> bool:
    """Check if server is responding."""
    try:
        response = urlopen(f"http://localhost:{port}/health", timeout=2)
        return response.status == 200
    except (URLError, TimeoutError):
        return False


def stop_server() -> bool:
    """Stop the running server."""
    if not PID_FILE.exists():
        print("No server PID file found.")
        return False

    try:
        pid = int(PID_FILE.read_text().strip())
        print(f"Stopping server (PID: {pid})...")

        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)

        PID_FILE.unlink()
        print("Server stopped.")
        return True
    except (ValueError, ProcessLookupError, PermissionError) as e:
        print(f"Failed to stop server: {e}")
        PID_FILE.unlink(missing_ok=True)
        return False


def start_server(
    server_exe: Path,
    model_path: Path,
    port: int,
    use_cuda: bool,
    foreground: bool,
) -> bool:
    """Start the llama.cpp server."""

    cmd = [
        str(server_exe),
        "-m", str(model_path),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--parallel", str(PARALLEL_REQUESTS),
        "--ctx-size", str(CONTEXT_SIZE),
        "--batch-size", str(BATCH_SIZE),
    ]

    if CONTINUOUS_BATCHING:
        cmd.append("--cont-batching")

    if MLOCK:
        cmd.append("--mlock")

    if use_cuda:
        cmd.extend(["--n-gpu-layers", str(GPU_LAYERS)])
        if FLASH_ATTENTION:
            cmd.append("--flash-attn")

    print(f"\nStarting llama.cpp server...")
    print(f"  Model: {model_path.name}")
    print(f"  Port: {port}")
    print(f"  Parallel: {PARALLEL_REQUESTS}")
    print(f"  Context: {CONTEXT_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Cont Batching: {CONTINUOUS_BATCHING}")
    print(f"  Mlock: {MLOCK}")
    print(f"  GPU: {'Yes' if use_cuda else 'No'}")
    if use_cuda:
        print(f"  GPU Layers: {GPU_LAYERS}")
        print(f"  Flash Attn: {FLASH_ATTENTION}")
    print()

    # Start server - stream output during loading
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

    print("Loading model (streaming output)...\n")
    print("-" * 60)

    import threading

    # Track if server is ready
    server_ready = False
    start_time = time.time()
    timeout = 120  # 2 minute timeout

    def read_output():
        """Read and print output from process."""
        nonlocal server_ready
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"  {line.rstrip()}")
                # Check for ready indicators
                if "HTTP server listening" in line or "model loaded" in line.lower():
                    server_ready = True

    # Start output reader thread
    reader_thread = threading.Thread(target=read_output, daemon=True)
    reader_thread.start()

    # Wait for server to be ready
    while time.time() - start_time < timeout:
        if server_ready or check_server(port):
            print("-" * 60)
            print(f"\nServer ready! Endpoint: http://localhost:{port}/v1")

            if foreground:
                print("\nRunning in foreground. Press Ctrl+C to stop.\n")
                try:
                    reader_thread.join()
                except KeyboardInterrupt:
                    print("\nStopping server...")
                    process.terminate()
                    PID_FILE.unlink(missing_ok=True)
            else:
                print(f"Server running in background (PID: {process.pid})")
                print(f"To stop: uv run python scripts/run_llamacpp.py --stop")

            return True

        # Check if process died
        if process.poll() is not None:
            print("-" * 60)
            print("\nERROR: Server process died.")
            return False

        time.sleep(0.5)

    print("-" * 60)
    print(f"\nERROR: Server failed to start within {timeout}s")
    process.terminate()
    return False


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the llama.cpp judge server (auto-downloads if needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Configuration (edit at top of script):
  PARALLEL_REQUESTS = {PARALLEL_REQUESTS}
  CONTEXT_SIZE = {CONTEXT_SIZE}
  BATCH_SIZE = {BATCH_SIZE}
  GPU_LAYERS = {GPU_LAYERS}
  CONTINUOUS_BATCHING = {CONTINUOUS_BATCHING}
  FLASH_ATTENTION = {FLASH_ATTENTION}
  MLOCK = {MLOCK}
""",
    )
    parser.add_argument(
        "--install-dir", "-d",
        type=Path,
        default=DEFAULT_INSTALL_DIR,
        help=f"Installation directory (default: {DEFAULT_INSTALL_DIR})",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Server port (default: from JUDGE_BASE_URL in .env)",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the running server",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU acceleration",
    )
    parser.add_argument(
        "--foreground", "-f",
        action="store_true",
        help="Run in foreground (show server output)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    install_dir = args.install_dir.resolve()
    port = args.port or get_port_from_env()

    # Handle stop command
    if args.stop:
        stop_server()
        return

    # Check if already running
    if check_server(port):
        print(f"Server already running on port {port}")
        print(f"Use --stop to stop it first")
        return

    print("=" * 60)
    print("LLAMA.CPP JUDGE SERVER")
    print("=" * 60)

    # Read config from .env
    judge_model = get_judge_model()
    use_cuda = not args.cpu_only and detect_cuda()

    print(f"JUDGE_MODEL: {judge_model}")
    print(f"Port: {port}")
    print(f"CUDA: {'Yes' if use_cuda else 'No'}")

    # Ensure llama.cpp is installed
    server_exe = ensure_llamacpp(install_dir, use_cuda)
    if not server_exe:
        print("\nERROR: Failed to set up llama.cpp")
        sys.exit(1)

    # Ensure model is downloaded
    model_path = ensure_model(install_dir, judge_model)
    if not model_path:
        print("\nERROR: Failed to download model")
        sys.exit(1)

    # Start server
    success = start_server(
        server_exe=server_exe,
        model_path=model_path,
        port=port,
        use_cuda=use_cuda,
        foreground=args.foreground,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
