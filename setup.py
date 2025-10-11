import subprocess
import sys

def run(cmd):
    """Run a shell command and raise on failure."""
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    # Create virtual environment
    run("uv venv")

    # Install basic tools
    run("uv pip install -U pip setuptools")

    # Install chumpy (legacy package)
    run("uv pip install chumpy --no-build-isolation")

    # Install all requirements
    run("uv pip install -r requirements_no_cuda.txt --no-build-isolation")

    print("✅ Environment setup complete.")
