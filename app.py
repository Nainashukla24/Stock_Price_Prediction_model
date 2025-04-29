from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Load models
models = {}
supported_stocks = ['AAPL', 'AMZN', 'DIS', 'GOOGL', 'META', 'MSFT', 'NFLX', 'NVDA', 'SPY', 'TSLA']
scaler = MinMaxScaler(feature_range=(0, 1))

# Load models at startup
for stock in supported_stocks:
    model_path = f'{stock}_lstm_model.h5'
    if os.path.exists(model_path):
        models[stock] = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html', stocks=supported_stocks)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.form['company_symbol'].upper()
        
        if symbol not in models:
            return jsonify({'error': f'Model not available for {symbol}'})
        
        # Get stock data
        data = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if len(data) < 60:
            return jsonify({'error': 'Insufficient data for prediction'})
        
        # Prepare data
        prices = data['Close'].values.reshape(-1, 1)
        scaled = scaler.fit_transform(prices)
        X = np.array([scaled[-60:]])
        
        # Predict
        model = models[symbol]
        pred_scaled = model.predict(X, verbose=0)
        pred_price = float(scaler.inverse_transform(pred_scaled)[0][0])
        curr_price = float(prices[-1][0])
        change = ((pred_price - curr_price) / curr_price) * 100
        
        return jsonify({
            'current_price': round(curr_price, 2),
            'predicted_price': round(pred_price, 2),
            'price_change': round(change, 2),
            'company_symbol': symbol,
            'last_update': data.index[-1].strftime('%Y-%m-%d')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
