
🕵️‍♂️ Fake Customer Classifier

A small end-to-end machine learning project that generates synthetic e-commerce customer data, trains a classifier to detect fake customers, and serves predictions via a simple Streamlit app.

🔗 Live Demo: https://fake-customer-classifier.streamlit.app

🚀 Overview
This project demonstrates a full ML pipeline — from data generation to model deployment — using a reproducible and modular codebase. It includes tools for creating synthetic datasets, preprocessing data, training and saving models, and serving predictions interactively.

🧩 Features
- 🧠 Synthetic Data Generation → Easily create labeled e-commerce customer data.
- 🧹 Preprocessing Utilities → Clean, encode, and prepare data for training.
- 🤖 Training Pipeline → Automates model training, evaluation, and saving artifacts.
- 📦 Saved Artifacts → Includes trained model and encoders for inference.
- 🌐 Streamlit App → Simple user interface for real-time predictions.

📁 Project Structure
```
Fake-Customer-Classifier/
│
├── app.py                   # Streamlit app entrypoint
├── configs/                 # Global paths, constants, and hyperparameters
├── data/                    # Generated datasets (CSV files)
├── models/                  # Saved models and label encoders
├── pipeline/
│   └── train_pipeline.py    # Model training and evaluation pipeline
├── src/
│   ├── data_generator.py    # Synthetic data generation
│   └── preprocessing.py     # Data preprocessing utilities
├── utils/                   # Logging and custom exception handling
└── requirements.txt         # Project dependencies
```

⚙️ Quickstart
1. Environment Setup
   - Create a virtual environment
     - macOS/Linux:
       ```bash
       python -m venv venv && source venv/bin/activate
       ```
     - Windows (PowerShell):
       ```powershell
       python -m venv venv; venv\Scripts\activate
       ```
2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Generate Synthetic Data
   ```bash
   python -c "from src.data_generator import generate_synthetic_data; generate_synthetic_data()"
   ```
4. Train the Model
   ```bash
   python -m pipeline.train_pipeline
   ```
5. Run the Streamlit App
   ```bash
   streamlit run app.py
   ```

🔧 Configuration
Modify parameters in `configs/config.py` to customize:
- Dataset size (`NUM_SAMPLES`)
- Train/test split
- File paths and storage directories
- Model hyperparameters

📝 Notes
- The data generator creates independent rows equal to `NUM_SAMPLES` (no fixed “cases per customer”).
- `customer_id` is not used as a feature; predictions are based on behavior/attributes only.

🧠 Requirements
- Python 3.10 or higher
- Dependencies listed in `requirements.txt`

💡 Future Improvements
- Add explainability (SHAP/feature importance)
- Enhance UI with confidence scores and detailed insights

