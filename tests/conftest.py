"""
conftest.py — pytest configuration and shared fixtures.

Adds src/ to sys.path so all test files can import project modules
without installing the package.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
