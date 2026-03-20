import os
import sys
from pathlib import Path

import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use CPU for tests
os.environ.setdefault("DEVICE", "cpu")
