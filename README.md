# 🍔 End-to-End Logistics ETA Predictor

## Overview
A machine learning web application designed to predict food delivery transit times based on driver metrics, weather conditions, and geographic routing data. 

🌐 **Live Application:** [[Paste your Streamlit Cloud URL here](https://delivery-eta-app-2x5e9gwdolj8gaqbz8a5xc.streamlit.app/)]

## Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Random Forest Regressor
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud

## Key Features
* **Feature Engineering:** Implemented the Haversine formula to dynamically calculate the great-circle distance between restaurants and delivery locations using raw GPS coordinates.
* **Modular Pipeline:** Built a robust Scikit-Learn `ColumnTransformer` and `Pipeline` to handle missing value imputation, standard scaling, and one-hot encoding without data leakage.
* **Interactive Dashboard:** Deployed a responsive frontend using Streamlit with custom CSS injections for a premium user experience.

## How to Run Locally
1. Clone the repository: `git clone https://github.com/vansham-teotia/delivery-eta-app.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
