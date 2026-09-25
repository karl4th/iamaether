#!/usr/bin/env python3
"""Entrypoint: generate the Kokoro-voiced Moshi fine-tuning dataset.

See COLAB.md for the exact uv setup and cell-by-cell invocation. All actual
logic lives in kokoro_ds/ so it can be unit tested without a GPU.
"""
from kokoro_ds.cli_generate import main

if __name__ == "__main__":
    raise SystemExit(main())
