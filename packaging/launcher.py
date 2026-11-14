"""PyInstaller entry point (the package's own __main__ uses relative imports and cannot be a script)."""
import multiprocessing
import sys

from rubberstudio.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
