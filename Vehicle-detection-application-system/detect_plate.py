"""
Legacy entrypoint kept for compatibility.

Core logic has been centralized under `plate_pipeline/`.
This script now forwards to `plate_pipeline.cli_detect`.
"""

from plate_pipeline.cli_detect import main


if __name__ == "__main__":
    main()

