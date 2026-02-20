import subprocess

def run_cmd(cmd):
    print(f"> {cmd}")
    # Run command in PowerShell
    result = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {cmd}")

def main():
    commands = [
        r'& "C:\Users\piotr.styrkowiec\AppData\Local\Programs\Python\Python38\python.exe" -m venv .venv',
        r'python -m pip install --upgrade pip',
        r'uv run pip install psychopy --no-deps',
        r'uv run pip install pandas arabic_reshaper python-bidi pyyaml pillow astunparse cryptography esprima freetype-py future gevent gitpython javascripthon jedi markdown-it-py MeshPy msgpack msgpack-numpy opencv-python psutil psychtoolbox pyarrow pyparallel pypiwin32 pyqt6 pyserial python-gitlab python-vlc pywinhook pyzmq questplus ujson websockets wxPython xmlschema setuptools==70.3.0 pyglet==1.4.11'
    ]

    for cmd in commands:
        run_cmd(cmd)

if __name__ == "__main__":
    main()




# & "C:\Users\piotr.styrkowiec\AppData\Local\Programs\Python\Python38\python.exe" -m venv .venv
# .\.venv\Scripts\activate
## python -m pip install --upgrade pip
## pip install -r base_install.txt

# & "C:\Users\piotr.styrkowiec\AppData\Local\Programs\Python\Python38\python.exe" -m venv .venv
# python -m pip install --upgrade pip
# uv run pip install psychopy --no-deps
# uv run pip install pandas arabic_reshaper python-bidi pyyaml pillow astunparse cryptography esprima freetype-py future gevent gitpython javascripthon jedi markdown-it-py MeshPy msgpack msgpack-numpy opencv-python psutil psychtoolbox pyarrow pyparallel pypiwin32 pyqt6 pyserial python-gitlab python-vlc pywinhook pyzmq questplus ujson websockets wxPython xmlschema setuptools==70.3.0 pyglet==1.4.11
