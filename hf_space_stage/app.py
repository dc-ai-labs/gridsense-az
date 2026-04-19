"""HuggingFace Space entrypoint. Delegates to the real dashboard."""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Run the streamlit app module
from app.streamlit_app import main  # noqa: E402
main()
