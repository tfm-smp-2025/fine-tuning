import os
import sys

# Add parent directory to python import path
sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

if __name__ == '__main__':
    # Import `src` module and call it's main function
    exit(__import__('src.__main__').main())