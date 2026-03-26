"""Shared pipeline utilities for plate detection/recognition.

This package centralizes logic that was previously duplicated across multiple
scripts (CLI/Flask/Django wrappers). Keep core logic here and expose stable APIs
for callers.
"""

from .core import (  # noqa: F401
    load_models,
    detect_recognition_plate,
    draw_result,
)

