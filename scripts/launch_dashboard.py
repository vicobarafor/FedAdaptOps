from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/fedadaptops/dashboard/app.py",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
