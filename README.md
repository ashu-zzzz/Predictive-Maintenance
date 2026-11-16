# Predictive Maintenance & Machine Failure Detection

This educational prototype classifies whether a machine is likely to fail from `Type`, air temperature, process temperature, rotational speed, torque, and tool wear. It does not predict remaining useful life or exact time-to-failure.

It uses the synthetic [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), Matzka, S. (2020), DOI: `10.24432/C5HS5C`. This is not a certified industrial safety system.

## Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
```

The pipeline uses stratified 70/15/15 train/validation/test splits, leakage-safe preprocessing, logistic regression, random forest, and extra trees. It selects by validation average precision and tunes the alert threshold on validation data only. Final metrics and plots are written under `results/metrics/`; the selected model is under `models/`.

The raw dataset is excluded from version control by `.gitignore`. The data is synthetic, may not represent a real plant, and should be validated against real operational data before any production use.
