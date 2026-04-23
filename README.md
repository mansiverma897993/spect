# Spect: Spam Detection System

`Spect` is a professional spam detection project built with:
- **Machine Learning**: TF-IDF + Logistic Regression (with hyperparameter tuning)
- **Backend (simple and clean)**: Python service module in `backend/model_service.py`
- **Frontend**: Modern Streamlit UI with confidence scores and model summary

The training pipeline is designed to keep model performance strong while reducing overfitting risk (small train/test gap).

## Features

- Clean backend prediction service
- Beautiful UI named **Spect: Spam Detection System**
- Tuned ML model using `GridSearchCV`
- Stratified train/test split for stable evaluation
- Saved metrics (train accuracy, test accuracy, F1 score, accuracy gap)
- Easy local setup

## Project Structure

- `app.py` - Streamlit frontend
- `train_model.py` - model training + tuning + evaluation + artifact saving
- `backend/model_service.py` - backend logic for loading model and predicting
- `requirements.txt` - dependencies
- `spam_model.pkl` - trained model artifact (generated after training)

## Quick Start

### 1) Create and activate virtual environment

Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Train model

```bash
python train_model.py
```

This command will:
- download the SMS spam dataset
- tune hyperparameters with cross-validation
- print train/test metrics
- save `spam_model.pkl`

### 4) Run frontend

```bash
python -m streamlit run app.py
```

Open: [http://localhost:8501](http://localhost:8501)

## Model Quality and Generalization

`train_model.py` reports:
- training accuracy
- testing accuracy
- training F1
- testing F1
- accuracy gap

Goal: keep **high test accuracy** and **small train/test gap** (good generalization).


## Deployment

You can deploy this app to Streamlit Community Cloud:
1. Push project to GitHub
2. Open [https://share.streamlit.io/](https://share.streamlit.io/)
3. Select repository and set main file to `app.py`
