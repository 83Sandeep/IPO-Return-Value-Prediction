from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

model = LinearRegression()
# Load the model
with open("stock_return_model1.pkl", "rb") as file:
    model = pickle.load(file)

print("type is", type(model))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    try:
        # Convert inputs to NumPy arrays
        ipo_price = float(request.form['ipo_price'])
        current_value = float(request.form['current_value'])

        # Create a feature array
        features = np.array([[ipo_price, current_value]])

        # Convert to DataFrame
        df = pd.DataFrame(features, columns=['IPO Price', 'Current'])

        # Predict
        prediction = model.predict(df)[0]

        # Calculate the return percentage
        return_percentage = ((current_value - ipo_price) / ipo_price) * 100

        # Determine investment decision based on return value
        if return_percentage > 0:
            message = f'Good Investment! Expected Profit: {round(return_percentage, 2)}%'
        else:
            message = f'Bad Investment! Expected Loss: {abs(round(return_percentage, 2))}%'

        return render_template('index.html', prediction_text=message,ipo_price_val=ipo_price,current_value_val=current_value)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
