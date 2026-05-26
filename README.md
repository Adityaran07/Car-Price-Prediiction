# 🚗 Car Price Prediction

An end-to-end machine learning pipeline to predict used car prices with an interactive Power BI-style dashboard.

---

## Features

| Feature | Detail |
|---|---|
| ML Models | Ridge, Random Forest, Gradient Boosting |
| Best R² Score | ~0.91 |
| Feature Engineering | Car age, power-to-engine ratio, log KM |
| REST API | Flask endpoints for live predictions |
| Dashboard | Interactive HTML BI dashboard |
| EDA | Auto-generated charts & heatmaps |

---

## Project Structure

```
car-price-prediction/
├── src/
│   ├── car_price_model.py       # ML pipeline (train + predict)
│   └── eda_and_data_gen.py      # EDA + synthetic data generator
├── data/
│   ├── car_data.csv             # Dataset (auto-generated or real)
│   ├── eda_charts.png           # EDA visualisations
│   └── correlation_heatmap.png
├── models/
│   └── car_price_model.pkl      # Trained model (auto-saved)
├── dashboard/
│   └── powerbi_dashboard.html   # Interactive BI dashboard
├── app.py                       # Flask REST API
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data & run EDA
```bash
python src/eda_and_data_gen.py
```

### 3. Train the model
```bash
python src/car_price_model.py
```

### 4. Run the Flask API
```bash
python app.py
```

### 5. Open the dashboard
Open `dashboard/powerbi_dashboard.html` in your browser.

---

## API Reference

### POST `/predict`
```json
{
  "year": 2019,
  "km_driven": 35000,
  "fuel": "Petrol",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": "First Owner",
  "engine_cc": 1197,
  "max_power_bhp": 82.0,
  "seats": 5
}
```
Response:
```json
{ "predicted_price_inr": 485000.0 }
```

### POST `/predict/batch`
Send a JSON array of car objects. Returns `{ "predictions": [...] }`.

---

## Model Performance

| Model | R² | MAE (₹) | RMSE (₹) |
|---|---|---|---|
| Ridge Regression | 0.81 | 68,400 | 91,200 |
| Random Forest | 0.89 | 41,300 | 62,800 |
| Gradient Boosting | **0.91** | **36,700** | **57,400** |

---

## Tech Stack

- Python 3.10+
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- Flask
- HTML5 / Chart.js (dashboard)
