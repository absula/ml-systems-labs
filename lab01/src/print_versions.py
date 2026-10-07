"""Print Python and every pinned package version (from repo requirements.txt)."""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements.txt"


def pinned_packages():
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line.split("==")[0]


def main():
    print(f"Python {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform {platform.platform()} / {platform.machine()}")
    print(f"Processor {platform.processor()}")
    print()
    for name in pinned_packages():
        try:
            print(f"{name}=={version(name)}")
        except PackageNotFoundError:
            print(f"{name}: NOT INSTALLED")


if __name__ == "__main__":
    main()
