# 🍔 End-to-End Logistics ETA Predictor

![Python]
![XGBoost]
![Streamlit]
![Scikit-Learn]

## 📌 Overview
An industry-grade machine learning web application designed to forecast food delivery transit times. By analyzing driver metrics, weather conditions, traffic density, and geographic routing data, this engine provides real-time Estimated Time of Arrival (ETA) predictions to optimize logistics and improve customer satisfaction.

🌐 **Live Application:** [Launch the ETA Predictor](https://delivery-eta-app-2x5e9gwdolj8gaqbz8a5xc.streamlit.app/)

## 🚀 Key Features & Feature Engineering
* **Haversine Distance Calculation:** Raw GPS coordinates (latitude and longitude for both the restaurant and delivery location) are mathematically transformed using the Haversine formula to calculate the exact great-circle distance in kilometers. This engineered feature serves as the strongest predictor of delivery time.
* **Robust Preprocessing Pipeline:** Built a modular `ColumnTransformer` pipeline that applies `StandardScaler` to numerical inputs, handles missing value imputation (median/mode strategies), and applies `OneHotEncoder` to categorical features, completely eliminating data leakage.
* **Interactive UI:** Deployed a responsive, premium frontend using Streamlit with custom CSS to simulate a production SaaS dashboard.

## 🧠 Model Selection & Evaluation 
During the exploratory data analysis (EDA) and modeling phase, four different algorithms were evaluated. **XGBoost** was selected as the final production model. Here is the breakdown of the selection process:

### Models Evaluated but Discarded:
1. **Multiple Linear Regression:** 
   * *Why it was discarded:* Delivery times are highly non-linear. The relationship between a "Stormy" weather condition, "High" traffic, and driver age cannot be accurately captured by a simple straight line. The model severely underfit the data.
2. **Decision Tree Regressor:** 
   * *Why it was discarded:* While capable of capturing non-linear relationships, single decision trees are highly prone to overfitting the training data and struggling with variance when exposed to unseen test data.
3. **Random Forest Regressor:** 
   * *Why it was discarded:* Random Forest performed very well by bagging multiple decision trees to reduce variance. However, it was slightly computationally heavier and lacked the sequential error-correction capabilities of gradient boosting, resulting in a slightly lower $R^2$ score than XGBoost.

### The Final Choice: XGBoost (`XGBRegressor`)
* **Why it was deployed:** XGBoost (Extreme Gradient Boosting) was chosen for the final `.pkl` artifact. Unlike Random Forest which builds trees independently, XGBoost builds trees *sequentially*, with each new tree correcting the residual errors of the previous one. 
* **Advantages:** It handles sparse data (created by our One-Hot Encoding of categories like City and Weather) exceptionally well, utilizes built-in L1/L2 regularization to prevent overfitting, and delivered the highest accuracy (lowest RMSE) on the holdout test set.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Web Framework:** Streamlit
* **Serialization:** Joblib / Pickle

## 💻 How to Run Locally

If you wish to run this project on your local machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/vansham-teotia/delivery-eta-app.git](https://github.com/vansham-teotia/delivery-eta-app.git)
cd delivery-eta-app

2. Create and activate a virtual environment

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install the dependencies
pip install -r requirements.txt

4. Run the Streamlit server
streamlit run app.py

📂 Project Structure
delivery-eta-app/
│
├── models/
│   ├── xgb_best.pkl            # Serialized XGBoost model
│   └── standard_scaler.pkl     # Fitted StandardScaler for numerical features
│
├── app.py                      # Main Streamlit application script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation


