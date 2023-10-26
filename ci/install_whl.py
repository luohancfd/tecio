#!/usr/bin/env python
import pathlib
import subprocess
import os


if __name__ == "__main__":
    script_path = pathlib.Path(os.path.dirname(__file__))
    repo_dir = script_path.parent
    files = sorted(repo_dir.glob('./dist/*.whl'))
    subprocess.run(["python", "-m", "pip", "install", "--force-reinstall", files[0]])
