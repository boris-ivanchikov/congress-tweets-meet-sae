import subprocess, sys, os

REPO = "ivanchikovboris/congress-tweets-meet-sae"
FOLDERS = ["data", "sae/runs"]

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("push", "pull"):
        print("Usage: sync [push|pull]")
        sys.exit(1)
        
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    action = sys.argv[1]
    for folder in FOLDERS:
        if action == "push":
            if not os.path.isdir(folder):
                print(f"Skipping {folder} (not found)")
                continue
            print(f"Pushing {folder} ...")
            subprocess.run(["hf", "upload", REPO, folder, folder], check=True)
        else:
            print(f"Pulling {folder} ...")
            subprocess.run(["hf", "download", REPO, "--include", f"{folder}/*", "--local-dir", "."], check=True)

    print("Done.")