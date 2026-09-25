#!/usr/bin/env python3
"""Entrypoint: validate a generated Kokoro Moshi fine-tuning dataset.

Pure-python/numpy: does not require torch, kokoro, or a GPU. See COLAB.md.
"""
from kokoro_ds.cli_validate import main

if __name__ == "__main__":
    raise SystemExit(main())
