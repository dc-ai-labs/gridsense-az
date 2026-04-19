# GridSense-AZ — developer recipes.
# Run `make <target>` from the repo root with `.venv/` active.

.PHONY: help run-app test lint

help:
	@echo "Targets:"
	@echo "  run-app   Launch the Streamlit dashboard (http://localhost:8501)"
	@echo "  test      Run the pytest suite (uses pyproject testpaths)"
	@echo "  lint      Ruff + black --check over src/, app/, tests/"

run-app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check src app tests
	black --check src app tests
