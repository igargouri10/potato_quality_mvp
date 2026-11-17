# Recording demo

Use an environment with the project's core runtime dependencies plus Streamlit. The demo preparation was tested on Python 3.12 with Streamlit 1.66, NumPy 2.3.5 and pandas 3.0.1; some full-project requirements differ. These versions describe the tested demo environment, not full training reproducibility.

Run from this repository:

```powershell
python scripts/create_demo_data.py
python -m streamlit run src/app.py
```

Select Use synthetic demo files, then batches 1042, 1043, 1044. Expected labels: Good, Warning, Poor. Files are generated locally and contain no employer data. The image and farm workflows require separate checks.

Record a 60–120 second walkthrough with Windows Snipping Tool (Win+Shift+R), show a real interaction and its output, then save and review the video. Explain the implementation was AI-assisted. Do not show secrets or private records. A working demo does not establish every feature or reported experiment works.
