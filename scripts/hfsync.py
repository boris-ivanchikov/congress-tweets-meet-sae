import argparse
import subprocess, sys, os
from pathlib import Path
from urllib.parse import urlsplit
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO = "ivanchikovboris/congress-tweets-meet-sae"
FOLDERS = ["data", "sae/runs", "interpretation/runs"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--proxy-file", type=Path, help="File containing a proxy URL")
    args = parser.parse_args()
    env = os.environ.copy()
    if args.proxy_file:
        proxy = args.proxy_file.expanduser().read_text().strip()
        try:
            parsed = urlsplit(proxy)
            valid = (parsed.scheme in ("http", "https", "socks5", "socks5h")
                     and parsed.hostname and parsed.port
                     and not any(c.isspace() for c in proxy))
        except ValueError:
            valid = False
        if not valid:
            parser.error("Proxy file must contain one valid HTTP(S) or SOCKS5 URL with a port")
        # Both the Python Hub client and Rust Xet uploader inherit these.
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env[name] = env[name.lower()] = proxy
        env["NO_PROXY"] = env["no_proxy"] = ""
        
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    action = args.action
    for folder in FOLDERS:
        if action == "push":
            if not os.path.isdir(folder):
                print(f"Skipping {folder} (not found)")
                continue
            print(f"Pushing {folder} ...")
            subprocess.run([sys.executable, "-m", "huggingface_hub.cli.hf", "upload", REPO, folder, folder], check=True, env=env)
        else:
            print(f"Pulling {folder} ...")
            subprocess.run([sys.executable, "-m", "huggingface_hub.cli.hf", "download", REPO, "--include", f"{folder}/*", "--local-dir", "."], check=True, env=env)

    print("Done.")


if __name__ == "__main__":
    main()
