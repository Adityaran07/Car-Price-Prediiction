# 🚗 Car Price Prediction using Machine Learning

A machine learning project that predicts used car selling prices using real-world car listing data. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and sample price prediction using a Gradient Boosting Regressor pipeline.

---

## 📌 Project Overview

This notebook-based project builds a regression model capable of estimating the selling price of used cars based on features such as:

- Manufacturing year
- Kilometers driven
- Fuel type
- Seller type
- Transmission type
- Ownership history
- Engine capacity
- Maximum power
- Number of seats
- Car age

The workflow includes:

1. Data loading and cleaning
2. Feature extraction and preprocessing
3. Exploratory Data Analysis (EDA)
4. Model training using Gradient Boosting Regressor
5. Model evaluation using R² Score and MAE
6. Predicting prices for new car inputs
7. Exporting dashboard-ready JSON statistics

---

## 📂 Project Structure

```bash
car-price-prediction/
│
├── Car_Price_Prediction.ipynb
├── README.md
├── dashboard/
│   └── data.json
└── data/
    └── Car details v3.csv
```

---

## ⚙️ Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook
- JSON

---

## 📊 Dataset Information

The dataset contains used car information including:

| Feature | Description |
|---|---|
| year | Manufacturing year |
| km_driven | Total kilometers driven |
| fuel | Fuel type of the car |
| seller_type | Dealer or individual seller |
| transmission | Manual or automatic |
| owner | Ownership history |
| engine | Engine capacity |
| max_power | Maximum engine power |
| seats | Seating capacity |
| selling_price | Target variable |

---

## 🧹 Data Preprocessing

- Removed missing values using `dropna()`
- Extracted numeric engine values
- Extracted numeric max power values
- Renamed columns
- Created `car_age` feature

---

## 📈 Exploratory Data Analysis (EDA)

Visualizations included:

- Selling price distribution
- Price by fuel type
- Price by transmission type

---

## 🤖 Machine Learning Model

### Model Used

- Gradient Boosting Regressor

### Pipeline Components

- ColumnTransformer
- OneHotEncoder
- Pipeline

---

## 🧪 Model Evaluation

Metrics used:

- R² Score
- Mean Absolute Error (MAE)

---

## 🚘 Sample Prediction

Example output:

```python
Predicted Price: ₹4,85,000
```

---

## 📦 Dashboard JSON Export

Exports statistics into:

```bash
dashboard/data.json
```

---

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Open:

```bash
Car_Price_Prediction.ipynb
```

Run all cells sequentially.

---

## 📌 Future Improvements

- Add more models
- Hyperparameter tuning
- Flask/FastAPI deployment
- Live dashboard
- Real-time prediction system

---

## 📜 License

Educational use only.
