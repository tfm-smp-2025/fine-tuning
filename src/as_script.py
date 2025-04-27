import os
import sys


if __name__ == '__main__':
    # Import `src` module and call it's main function

    import importlib
    # Add parent directory to python import path
    sys.path.insert(0,
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))

    module = importlib.import_module('src.__main__')

    # Restore path
    sys.path.pop(0)

    exit(module.main())