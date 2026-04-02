from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('ice-cream.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Encode categoricals
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
month_map = {'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10}
df['DayOfWeek'] = df['DayOfWeek'].map(day_map)
df['Month'] = df['Month'].map(month_map)

# Features and target
features = ['DayOfWeek', 'Month', 'Temperature', 'Rainfall']
X = df[features]
y = df['IceCreamsSold']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Compute R2 score
r2_score = model.score(X_test, y_test)

# Create date to features mapping
date_features = {}
for _, row in df.iterrows():
    date_str = row['Date'].strftime('%Y-%m-%d')
    date_features[date_str] = [row['DayOfWeek'], row['Month'], row['Temperature'], row['Rainfall']]

# Group dates by month
month_to_dates = {}
month_names = {4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October'}
for date_str, feats in date_features.items():
    month_num = feats[1]
    month_name = month_names[month_num]
    if month_name not in month_to_dates:
        month_to_dates[month_name] = []
    month_to_dates[month_name].append(date_str)

# Compute summary
summary = {
    'total_records': len(df),
    'avg_sales': round(df['IceCreamsSold'].mean(), 2),
    'max_sales': df['IceCreamsSold'].max(),
    'min_sales': df['IceCreamsSold'].min(),
    'avg_temp': round(df['Temperature'].mean(), 2),
    'avg_rain': round(df['Rainfall'].mean(), 2),
    'r2_score': round(r2_score, 4)
}

@app.route('/')
def index():
    return render_template(
        'index.html',
        summary=summary,
        month_to_dates=month_to_dates,
        selected_date=None,
        selected_month=list(month_to_dates.keys())[0]
    )

@app.route('/predict', methods=['POST'])
def predict():
    date = request.form.get('date')
    selected_month = request.form.get('month')
    if date not in date_features:
        return render_template(
            'index.html',
            summary=summary,
            month_to_dates=month_to_dates,
            prediction=None,
            selected_date=None,
            selected_month=selected_month
        )

    features = date_features[date]
    pred = model.predict([features])[0]
    # If month not provided, infer from selected_date
    if not selected_month and date:
        selected_month = month_names[features[1]]

    return render_template(
        'index.html',
        summary=summary,
        month_to_dates=month_to_dates,
        prediction=round(pred),
        selected_date=date,
        selected_month=selected_month
    )

if __name__ == '__main__':
    app.run(debug=True)