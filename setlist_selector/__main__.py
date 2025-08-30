#!/usr/bin/env python3
"""
Entry point for running setlist_selector as a module.

This file allows the package to be executed with:
python -m setlist_selector
"""

import sys
from .main import main

if __name__ == "__main__":
    sys.exit(main())
