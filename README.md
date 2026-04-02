# Ice Cream Sales Predictor

A unique Flask web app for predicting ice cream sales based on weather data.

## Features

- Dataset summary with key statistics
- Prediction form restricted to dates in the dataset
- Custom frosty design with responsive layout
- Linear regression model trained on 70% of the data

## Requirements

- Python 3.13+
- Flask
- Pandas
- Scikit-learn

## Installation

1. Install dependencies:

   ```
   pip install flask pandas scikit-learn
   ```

2. Run the app:

   ```
   python app.py
   ```

3. Open http://localhost:5000 in your browser

## Model Details

- Features: Day of week, Month, Temperature, Rainfall
- Target: Ice cream sales
- Algorithm: Linear Regression
- Train/Test split: 70/30 with random_state=42
