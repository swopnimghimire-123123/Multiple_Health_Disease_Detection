# Multi-Disease Predictor

This Flask-based demo bundles three medical prediction flows (brain tumor image classification, heart disease logistic regression, and a diabetes Random Forest) together with shared UI/UX and model explanations. The project is designed so you can upload an MRI image or provide tabular values, see a prediction, and explore how each model is built from data and architecture choices.

## Repository structure

```
final_med_application/
├── app.py                  # Flask routes, uploads, and prediction logic
├── requirements.txt        # Runtime dependencies (Flask, TensorFlow, sklearn, etc.)
├── Procfile               # Production entrypoint for gunicorn
├── README.md              # This document
├── .gitignore             # Ignored artifacts
├── templates/             # HTML views (home, brain tumor form, model info, etc.)
├── static/                # Shared styles, scripts, and assets
├── models/                # Serialized ML artifacts (.h5, .pkl, .sav)
├── uploads/               # Runtime upload directory (ignored)
├── Brain_Tumor_Prediction_Using_DL.ipynb
├── Heart_Disease_Prediction_Using_ML.ipynb
├── Diabetes_Prediction_Using_ML.ipynb
└── ...
```

The `model_info.html` page summarizes each model’s architecture, parameters, datasets, and evaluation so users can understand what is powering every prediction.

## Environment setup

1. Create/activate a Python 3.10+ environment (matching `tensorflow`’s requirements).
2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. (Optional) Copy `.env.example` to `.env` and fill values if you plan to move secrets/configuration to environment variables. The app uses `python-dotenv` so any `FLASK_*`, `MODEL_*`, or `SECRET_KEY` variables defined there will be loaded automatically.

## Local development

```bash
# ensure the upload directory exists
mkdir -p uploads

# launch in Flask dev mode
python app.py
```

Flask listens on `http://127.0.0.1:5000` by default; open it to see brain tumor, heart disease, diabetes, and model info flows. Use `/model-info` to read the human-centric descriptions of each model.

## Production deployment

- The included `Procfile` runs `gunicorn app:app`, so platforms like Heroku, Render, and Fly.io can start the app directly.
- Ensure the hosting environment sets `PYTHONPATH`, `FLASK_APP=app.py`, and (if exposed) `PORT`. Gunicorn respects `PORT` automatically.
- Keep the `models/` directory committed or copy it to storage that the host can access before booting the service.
- You can use `python-dotenv` locally and configure real secrets (e.g., upload size limit) in the host’s environment settings.

Optional enhancements:
- Run `gunicorn --worker-class=gthread --workers=3 --timeout=120 app:app` if memory allows for heavier TensorFlow loads.
- Add `models/` to a CDN/Blob storage and load them via path URL if the host enforces strict upload size limits.

## Navigation summary

- `/` – Home dashboard linking to each disease flow.
- `/brain-tumor` – Upload an MRI to classify tumors via the CNN.
- `/heart-disease` – Fill validated vitals and ECG fields to score heart risk.
- `/diabetes` – Provide clinical values to compute diabetes probability using the stored Random Forest.
- `/model-info` – Explore the architecture, hyperparameters, data prep, and metrics for every model.

## Notes

- Keep `uploads/` empty in the repo; it is listed in `.gitignore` because it accumulates user-submitted files.
- The heart disease model was trained with scikit-learn 1.0.2; loading it with newer versions may emit an `InconsistentVersionWarning`. Re-training with 1.3.2 and re-serializing (or pinning to 1.0.2) avoids the warning.
- The TensorFlow brain model expects 150×150 images; the app already resizes and converts to BGR/GRAYSCALE as needed.
