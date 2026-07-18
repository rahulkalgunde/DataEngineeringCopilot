# Python Environment Rules

You MUST always use the Python virtual environment located at `dec_venv/` at the project root.

## Execution Protocols:

1. **Running Python Scripts:** Do not run bare `python script.py` commands. Instead, explicitly use the virtual environment's executable:
   `dec_venv/bin/python <script_name>.py`
   *(Alternatively, for modules: `dec_venv/bin/python -m <module_path>`)*

2. **Installing Packages:** This project uses **`uv`** exclusively. Do not use `pip`:
   `uv pip install -e ".[dev]"`

3. **Multi-Command Terminal Sessions:** If you open a terminal state to run tests, execute interactive tools, or handle long-running background tasks, you must run the activation script first:
   `source dec_venv/bin/activate`

4. **Streamlit Application Run:** Use streamlit directly from the virtual environment's binary folder to ensure all dependencies match:
   `dec_venv/bin/python -m streamlit run <app_script>.py`
