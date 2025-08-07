#!/usr/bin/env python3
# shell/__main__.py
"""
Portfolio Optimization CLI Package Entry Point
==============================================

This file allows the shell module to be run as a package using:
    python -m shell [arguments]

It imports and runs the main function from main.py
"""

from .main import main

if __name__ == "__main__":
    main()