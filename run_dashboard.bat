@echo off
pushd "%~dp0"
echo Starting LLM Lie Detector Dashboard from %CD%...
echo Activating environment and launching Streamlit...
.\.venv\Scripts\streamlit run app.py
popd
pause
