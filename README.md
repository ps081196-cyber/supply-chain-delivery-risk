# Supply Chain Delivery Risk Predictor

A machine-learning portfolio project that predicts whether a shipment is at risk of arriving late, then explains the operational drivers behind the prediction.

## Business problem
Late deliveries increase customer contacts, replacement costs and SLA failures. This application helps operations teams prioritize risky shipments before the promised date.

## Features
- Synthetic shipment generator for a reproducible demo
- Random Forest classification pipeline
- Risk probability for every shipment
- Feature-importance chart and high-risk exception queue
- CSV upload and downloadable scored results
- Unit-tested business logic

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input columns
`distance_km, promised_days, carrier_rating, weather_severity, traffic_index, order_value, dispatch_delay_hours`

## Technology
Python · pandas · scikit-learn · Streamlit · Plotly · pytest

## Portfolio outcome
Demonstrates how predictive analytics can support proactive delivery management and SLA protection. The data and implementation are original and created for portfolio learning.
