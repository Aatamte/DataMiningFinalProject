"""vLLM server management.

Commands:
    uv run python scripts/run_vllm.py --list                    # List all profiles
    uv run python scripts/run_vllm.py --server                  # Start ALL profiles
    uv run python scripts/run_vllm.py --server --profile sft    # Start one profile
    uv run python scripts/run_vllm.py --stop                    # Stop ALL profiles
    uv run python scripts/run_vllm.py --stop --profile sft      # Stop one profile

Configuration in configs/vllm.yaml
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

import yaml
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Load config from YAML
# =============================================================================

CONFIG_PATH = Path("configs/vllm.yaml")

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()


def get_profile(name: str = None) -> tuple[dict, str]:
    """Get a profile from config."""
    profiles = CONFIG.get("profiles", {})
    if name is None:
        # Return first profile as default
        if profiles:
            name = list(profiles.keys())[0]
            return profiles[name], name
        return {}, ""
    return profiles.get(name, {}), name

def get_server_config() -> dict:
    """Get server settings from config."""
    return CONFIG.get("server", {})

def get_lora_config() -> dict:
    """Get LoRA settings from config."""
    return CONFIG.get("lora", {})


# =============================================================================
# Paths
# =============================================================================

MODELS_DIR = Path("models")    # Local model storage directory
PID_DIR = Path("tools/vllm")   # PID files directory

def get_pid_file(profile_name: str) -> Path:
    """Get PID file path for a profile."""
    return PID_DIR / f"{profile_name}.pid"


# =============================================================================
# Helpers
# =============================================================================

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

def cmd_list():
    """List available profiles from config."""
    profiles = CONFIG.get("profiles", {})
    active = CONFIG.get("active", "")

    print("=" * 60)
    print("VLLM PROFILES (configs/vllm.yaml)")
    print("=" * 60)

    for name, profile in profiles.items():
        model = profile.get("model", "")
        port = profile.get("port", 1234)
        lora = profile.get("lora")

        # Check if running
        pid_file = get_pid_file(name)
        running = pid_file.exists() and check_server(port)

        marker = "*" if name == active else " "
        status = "RUNNING" if running else ""

        print(f"{marker} {name:<12} port:{port:<5} {status}")
        print(f"    model: {model}")
        if lora:
            lora_exists = Path(lora).exists()
            lora_status = "" if lora_exists else "(missing)"
            print(f"    lora:  {lora} {lora_status}")
        print()

    print("-" * 60)
    print(f"* = active profile")
    print(f"Config: {CONFIG_PATH.absolute()}")


def cmd_stop(profile_name: str = None):
    """Stop a running server."""
    if profile_name is None:
        _, profile_name = get_profile()

    pid_file = get_pid_file(profile_name)

    if not pid_file.exists():
        print(f"No PID file for profile '{profile_name}'")
        return False

    try:
        pid = int(pid_file.read_text().strip())
        print(f"Stopping {profile_name} (PGID: {pid})...")
        # Kill entire process group (vLLM spawns child processes via Ray)
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process group doesn't exist, try single process
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(2)
        # Force kill if still running
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pid_file.unlink()
        print("Server stopped.")
        return True
    except Exception as e:
        print(f"Failed to stop: {e}")
        pid_file.unlink(missing_ok=True)
        return False


def cmd_server_all(foreground: bool = False):
    """Start vLLM servers for ALL profiles."""
    profiles = CONFIG.get("profiles", {})
    if not profiles:
        print("ERROR: No profiles defined in config")
        return False

    print("=" * 60)
    print("STARTING ALL VLLM PROFILES")
    print("=" * 60)

    started = []
    for name in profiles:
        print(f"\nStarting profile: {name}")
        if cmd_server_single(name, foreground=False):
            started.append(name)
        else:
            print(f"  Failed to start {name}")

    print("\n" + "=" * 60)
    print(f"Started {len(started)}/{len(profiles)} profiles: {', '.join(started)}")
    print("=" * 60)

    if foreground and started:
        print("\nPress Ctrl+C to stop all servers.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping all servers...")
            for name in started:
                cmd_stop(name)

    return len(started) > 0


def cmd_server_single(profile_name: str, foreground: bool = False):
    """Start the vLLM server for a single profile."""
    profile, profile_name = get_profile(profile_name)
    if not profile:
        print(f"ERROR: Profile '{profile_name}' not found in config")
        return False

    model_path = profile.get("model", "")
    port = profile.get("port", 1234)
    lora_path = profile.get("lora")

    if not model_path:
        print(f"ERROR: No model specified in profile '{profile_name}'")
        return False

    # Check vllm installed
    try:
        import vllm
    except ImportError:
        print("ERROR: vllm not installed.")
        print("  pip install vllm")
        return False

    # Check if already running
    if check_server(port):
        print(f"Port {port} already in use")
        print(f"Use: --stop --profile {profile_name}")
        return False

    # Get server settings from config
    server_cfg = get_server_config()
    lora_cfg = get_lora_config()

    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", server_cfg.get("host", "0.0.0.0"),
        "--gpu-memory-utilization", str(profile.get("gpu_memory", 0.5)),
        "--max-model-len", str(profile.get("max_model_len", server_cfg.get("max_model_len", 6500))),
        "--max-num-seqs", str(server_cfg.get("max_num_seqs", 10)),
        "--dtype", server_cfg.get("dtype", "auto"),
        "--kv-cache-dtype", server_cfg.get("kv_cache_dtype", "auto"),
        "--trust-remote-code",
    ]

    if server_cfg.get("enable_chunked_prefill", True):
        cmd.append("--enable-chunked-prefill")
    if server_cfg.get("disable_log_requests", True):
        cmd.append("--disable-log-requests")

    # Add LoRA if specified
    if lora_path:
        cmd.extend([
            "--enable-lora",
            "--lora-modules", f"{profile_name}={lora_path}",
            "--max-lora-rank", str(lora_cfg.get("max_rank", 64)),
            "--max-loras", str(lora_cfg.get("max_loras", 4)),
        ])

    print("=" * 60)
    print(f"VLLM SERVER: {profile_name}")
    print("=" * 60)
    print(f"Model: {model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"Port: {port}")
    print(f"GPU Memory: {profile.get('gpu_memory', 0.5) * 100:.0f}%")
    print(f"Max Model Len: {profile.get('max_model_len', server_cfg.get('max_model_len', 6500))}")
    print("=" * 60)
    print()

    # Start process in new process group (so we can kill all child processes)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # Creates new process group
    )

    # Save PGID (process group ID) - same as PID for session leader
    pgid = os.getpgid(process.pid)
    pid_file = get_pid_file(profile_name)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pgid))

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
                    pid_file.unlink(missing_ok=True)
            else:
                print(f"PGID: {pgid}")
                print(f"Stop: uv run python scripts/run_vllm.py --stop --profile {profile_name}")
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
        description="vLLM server management (config: configs/vllm.yaml)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List profiles
  %(prog)s --list

  # Start active profile (set in config)
  %(prog)s --server

  # Start specific profile
  %(prog)s --server --profile sft

  # Stop a profile
  %(prog)s --stop --profile sft
""",
    )

    # Commands
    parser.add_argument("--list", action="store_true",
        help="List available profiles")
    parser.add_argument("--server", action="store_true",
        help="Start server for a profile")
    parser.add_argument("--stop", action="store_true",
        help="Stop a running server")

    # Options
    parser.add_argument("--profile", "-p",
        help="Profile name (default: active profile in config)")
    parser.add_argument("--foreground", "-f", action="store_true",
        help="Run in foreground")

    args = parser.parse_args()

    # Must specify a command
    if not (args.list or args.server or args.stop):
        parser.print_help()
        print("\nError: Specify --list, --server, or --stop")
        sys.exit(1)

    # Handle commands
    if args.list:
        cmd_list()
        return

    if args.stop:
        if args.profile:
            cmd_stop(args.profile)
        else:
            # Stop ALL profiles
            profiles = CONFIG.get("profiles", {})
            for name in profiles:
                cmd_stop(name)
        return

    if args.server:
        if args.profile:
            # Start single profile
            success = cmd_server_single(args.profile, args.foreground)
        else:
            # Start ALL profiles
            success = cmd_server_all(args.foreground)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
