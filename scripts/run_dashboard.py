from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "model_results_dashboard.py"


def local_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def port_is_busy(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the SOCO Streamlit dashboard.")
    parser.add_argument("--port", type=int, default=8502, help="Local Streamlit port. Default: 8502")
    parser.add_argument("--host", default="localhost", help="Host shown in the local URL. Default: localhost")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser tab.")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    network_ip = local_ip()

    if port_is_busy(args.port):
        print(f"\nPort {args.port} is already in use.", flush=True)
        print("Opening the existing app or service at:", flush=True)
        print(f"Local URL: {url}", flush=True)
        print("If this is an old dashboard session, stop it from the terminal where it is running.\n", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    print("\nSOCO Model Results Dashboard is starting...", flush=True)
    print("Open the app in your browser:", flush=True)
    print(f"Local URL: {url}", flush=True)
    if network_ip and args.host in {"localhost", "127.0.0.1"}:
        print(f"Network URL: http://{network_ip}:{args.port}", flush=True)
    if not args.no_browser:
        print("Opening the browser automatically now...", flush=True)
        webbrowser.open(url)
    print("If the browser does not open automatically, copy/paste the URL above.", flush=True)
    print("To close the app, return to this terminal and press Ctrl+C.\n", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(args.port),
    ]
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
