"""
tests/conftest.py
Global pytest configuration for ZebraID.

Configures environment variables to prevent OpenMP / threadpool runtime collisions
between PyTorch, FAISS, and AnyIO/FastAPI test runners on macOS (Darwin).
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
