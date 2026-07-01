from __future__ import annotations

import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install as _install


class InstallWithModels(_install):
    def run(self):
        # Run the standard install first
        _install.run(self)

        # Attempt to download embedding model into local cache
        script = os.path.join(os.path.dirname(__file__), "scripts", "download_embedding_model.py")
        if os.path.exists(script):
            try:
                print("Running embedding model cache script:", script)
                subprocess.check_call([sys.executable, script])
            except Exception as exc:  # pragma: no cover - best-effort at install time
                print("Warning: failed to cache embedding model during install:", exc)
        else:
            print("Embedding download script not found; skipping model cache step.")


setup(
    name="data-engineering-copilot",
    version="0.1.0",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    cmdclass={"install": InstallWithModels},
)
