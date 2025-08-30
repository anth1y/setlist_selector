#!/bin/bash
cd ~/setlist_selector
source .venv/bin/activate
/opt/homebrew/bin/uv run python -m setlist_selector.main
