# Python Environment Rules

You MUST always use the Python virtual environment located at:
`/home/rahul/PythonVenvs/data_eng_copilot_env`

## Execution Protocols:

1. **Running Python Scripts:** Do not run bare `python script.py` commands. Instead, explicitly use the absolute path to the virtual environment's executable:
   `/home/rahul/PythonVenvs/data_eng_copilot_env/bin/python <script_name>.py`
   *(Alternatively, for modules: `/home/rahul/PythonVenvs/data_eng_copilot_env/bin/python -m <module_path>`)*

2. **Installing Packages:** Do not run bare `pip install`. Explicitly invoke pip via the environment's executable to avoid permission issues and scope leakage:
   `/home/rahul/PythonVenvs/data_eng_copilot_env/bin/pip install <package>`

3. **Multi-Command Terminal Sessions:** If you open a terminal state to run tests, execute interactive tools, or handle long-running background tasks, you must run the activation script first:
   `source /home/rahul/PythonVenvs/data_eng_copilot_env/bin/activate`

4. **Streamlit Application Run:** Use streamlit directly from the virtual environment's binary folder to ensure all dependencies match:
   `/home/rahul/PythonVenvs/data_eng_copilot_env/bin/streamlit run <app_script>.py`