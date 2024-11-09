from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo entrenado
model = joblib.load('modelo_smart_roach_gbm.pkl')

# Iniciar la aplicación Flask
app = Flask(__name__)

# Definir la ruta de la página principal
@app.route('/')
def home():
    return render_template('index.html')

# Definir la ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    age = request.form['age']
    tumor_stage = request.form['tumor_stage']
    psa = request.form['psa']
    gleason_score = request.form['gleason_score']
    race = request.form['race']

    # Preprocesar los datos como se hizo en el entrenamiento
    features = pd.DataFrame({
        'psa': [float(psa)],
        'gleason_score': [gleason_score],
        'num_tumors_in_situ': [1],
        'age': [age],
        'examined_nodes': [10]
    })

    # Generar predicción
    prediction = model.predict_proba(features)[0][1] * 100  # Porcentaje de riesgo

    return render_template('index.html', prediction_text=f'El riesgo estimado de ganglios positivos es de {prediction:.2f}%')

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)