from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load(r'C:\Users\agust\Documents\SmartRoachTest\modelo_smart_roach_final.pkl')

# Cargar los nombres de las características
feature_names_filename = r'C:\Users\agust\Documents\SmartRoachTest\feature_names.json'
with open(feature_names_filename, 'r') as f:
    feature_names = json.load(f)

# Imprimir los nombres de las características para verificar
print("Nombres de las características en el modelo:", feature_names)

# Función para preparar los datos en el formato necesario para el modelo
def prepare_input_data(edad, t, race, psa, gleason):
    # Crear un diccionario con las características inicializadas en cero
    input_dict = dict.fromkeys(feature_names, 0)

    # Asignar los valores numéricos
    if 'Edad' in input_dict:
        input_dict['Edad'] = edad
    else:
        print("Advertencia: La característica 'Edad' no se encuentra en el modelo.")

    if 'PSA' in input_dict:
        input_dict['PSA'] = psa
    else:
        print("Advertencia: La característica 'PSA' no se encuentra en el modelo.")

    if 'Gleason' in input_dict:
        input_dict['Gleason'] = gleason
    else:
        print("Advertencia: La característica 'Gleason' no se encuentra en el modelo.")

    # Características derivadas
    input_dict['PSA_Gleason'] = psa * gleason
    input_dict['Log_PSA'] = np.log1p(psa)
    input_dict['T_Length'] = len(str(t))

    # Binning de Edad
    if edad < 50:
        edad_bin = '<50'
    elif 50 <= edad < 60:
        edad_bin = '50-59'
    elif 60 <= edad < 70:
        edad_bin = '60-69'
    elif 70 <= edad < 80:
        edad_bin = '70-79'
    else:
        edad_bin = '80+'
    edad_bin_feature = f'Edad_Binned_{edad_bin}'
    if edad_bin_feature in input_dict:
        input_dict[edad_bin_feature] = 1
    else:
        print(f"Advertencia: '{edad_bin_feature}' no se encuentra en las características del modelo.")

    # Codificar la variable 'T'
    t_feature_name = f'T_{t}'
    if t_feature_name in input_dict:
        input_dict[t_feature_name] = 1
    else:
        print(f"Advertencia: '{t_feature_name}' no se encuentra en las características del modelo.")

    # Codificar la variable 'Race'
    race_feature_name = f'Race_{race}'
    if race_feature_name in input_dict:
        input_dict[race_feature_name] = 1
    else:
        print(f"Advertencia: '{race_feature_name}' no se encuentra en las características del modelo.")

    # Interacción entre Race y Grupo_Riesgo
    # Necesitamos asignar 'Grupo_Riesgo' basado en 'T'
    ct_stage = str(t).strip()
    if ct_stage in ['1', 'T1', 'T1a', 'T1b', 'T1c']:
        grupo_riesgo = 'Muy bajo'
    elif ct_stage in ['2', 'T2', 'T2a']:
        grupo_riesgo = 'Bajo'
    elif ct_stage in ['2b', 'T2b', '2c', 'T2c']:
        grupo_riesgo = 'Intermedio'
    elif ct_stage in ['3a', 'T3a']:
        grupo_riesgo = 'Alto'
    elif ct_stage in ['3b', 'T3b', '4', 'T4']:
        grupo_riesgo = 'Muy alto'
    else:
        grupo_riesgo = 'No clasificado'

    race_grupo_feature = f'Race_Grupo_Riesgo_{race}_{grupo_riesgo}'
    if race_grupo_feature in input_dict:
        input_dict[race_grupo_feature] = 1
    else:
        print(f"Advertencia: '{race_grupo_feature}' no se encuentra en las características del modelo.")

    # Crear un DataFrame con las características en el mismo orden
    input_df = pd.DataFrame([input_dict], columns=feature_names)

    # Verificar el número de características y el contenido
    print("Número de características de entrada:", len(input_df.columns))  # Debe coincidir con el modelo
    print("Características de entrada:\n", input_df)

    return input_df

# Ruta para el formulario principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtener los datos del formulario
        edad = int(request.form['edad'])
        t = request.form['t']
        race = request.form['race']
        psa = float(request.form['psa'])
        gleason = int(request.form['gleason'])

        # Preparar los datos para el modelo
        input_data = prepare_input_data(edad, t, race, psa, gleason)

        # Realizar la predicción
        prediction = model.predict_proba(input_data)[0][1]  # Probabilidad de tener ganglios positivos

        # Mostrar el resultado como porcentaje
        risk_percentage = round(prediction * 100, 2)

        # Enviar datos al template con los valores ingresados
        return render_template('index.html', result=risk_percentage, edad=edad, t=t, race=race, psa=psa, gleason=gleason)

    # Solicitud GET: Inicializar valores predeterminados para evitar errores de "UndefinedError"
    return render_template('index.html', result=None, edad='', t='T1a', race='Unknown', psa='', gleason=6)

if __name__ == '__main__':
    app.run(debug=True)
