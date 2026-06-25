.PHONY: setup

setup:
	@echo "Updating pip and installing Python requirements..."
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	@echo "Downloading and caching embedding model (may take a while)..."
	python scripts/download_embedding_model.py

run_app:
	@echo "Running streamlit application"
	python -m streamlit run data_engineering_copilot/ui/streamlit_app.py

