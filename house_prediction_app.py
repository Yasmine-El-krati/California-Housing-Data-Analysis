<<<<<<< HEAD
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd



FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income',
    'ocean_proximity_<1H OCEAN',
    'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND',
    'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN'
]





app = Flask(__name__)

# load model
with open(r"C:\Users\PC\Downloads\AI_project\house_price_model_v1.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')
from sklearn.preprocessing import LabelEncoder

# Load the same encoder you used during training
# (If you didn't save it, re-fit it with the same categories in the same order!)

label_encoder = LabelEncoder()
label_encoder.fit(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'longitude': float(request.form['longitude']),
        'latitude': float(request.form['latitude']),
        'housing_median_age': float(request.form['housing_median_age']),
        'total_rooms': float(request.form['total_rooms']),
        'total_bedrooms': float(request.form['total_bedrooms']),
        'population': float(request.form['population']),
        'households': float(request.form['households']),
        'median_income': float(request.form['median_income']),
        'ocean_proximity': request.form['ocean_proximity']
    }

    df = pd.DataFrame([data])

    # Encode ocean_proximity the same way as training
    df['ocean_proximity'] = label_encoder.transform(df['ocean_proximity'])

    # Predict
    prediction = model.predict(df)[0]

    return render_template('index.html',
                           prediction_text=f"Predicted House Price: ${prediction:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd



FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income',
    'ocean_proximity_<1H OCEAN',
    'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND',
    'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN'
]





app = Flask(__name__)

# load model
with open(r"C:\Users\PC\Downloads\AI_project\house_price_model_v1.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')
from sklearn.preprocessing import LabelEncoder

# Load the same encoder you used during training
# (If you didn't save it, re-fit it with the same categories in the same order!)

label_encoder = LabelEncoder()
label_encoder.fit(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'longitude': float(request.form['longitude']),
        'latitude': float(request.form['latitude']),
        'housing_median_age': float(request.form['housing_median_age']),
        'total_rooms': float(request.form['total_rooms']),
        'total_bedrooms': float(request.form['total_bedrooms']),
        'population': float(request.form['population']),
        'households': float(request.form['households']),
        'median_income': float(request.form['median_income']),
        'ocean_proximity': request.form['ocean_proximity']
    }

    df = pd.DataFrame([data])

    # Encode ocean_proximity the same way as training
    df['ocean_proximity'] = label_encoder.transform(df['ocean_proximity'])

    # Predict
    prediction = model.predict(df)[0]

    return render_template('index.html',
                           prediction_text=f"Predicted House Price: ${prediction:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> d0371b35cc76f251d1080b8badcd47b81b221a35
