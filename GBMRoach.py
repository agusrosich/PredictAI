# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import joblib
from imblearn.over_sampling import SMOTE

# Cargar los datos desde el archivo Excel
data = pd.read_excel(r'C:\Users\agust\Documents\SmartRoachTest\BasedeDatosRoach.xlsx')

# Verificar los nombres de las columnas
print("Nombres de las columnas en el archivo:")
print(data.columns)

# Especificar la columna objetivo y las columnas a ignorar o transformar
target_column = 'Nodos regionales positivos'  # Cambia aquí si el nombre de la columna objetivo es diferente
text_column = 'Anatomia patologica'  # Cambia aquí al nombre exacto de la columna de texto
t_column = 'T'  # Cambia aquí al nombre exacto de la columna con el estadio clínico alfanumérico
race_column = 'Race'  # Cambia aquí al nombre exacto de la columna de raza

# Verificar si las columnas especificadas existen
if target_column not in data.columns:
    print(f"Error: La columna objetivo '{target_column}' no existe en el archivo.")
else:
    # Transformar la columna de ganglios a binaria: 1 si al menos un ganglio es positivo, 0 en caso contrario
    data[target_column] = data[target_column].apply(lambda x: 1 if x > 0 else 0)
    print("Transformación de la columna de ganglios a binaria completada.")

    # Ignorar la columna de texto si existe
    if text_column in data.columns:
        data = data.drop(columns=[text_column])

    # Convertir la columna de estadio "T" a variables dummy usando OneHotEncoder si existe
    if t_column in data.columns:
        onehot_encoder_t = OneHotEncoder(sparse_output=False, drop='first')
        t_encoded = onehot_encoder_t.fit_transform(data[[t_column]])

        # Convertir el resultado a un DataFrame y agregar los nombres de columnas
        t_encoded_df = pd.DataFrame(t_encoded, columns=onehot_encoder_t.get_feature_names_out([t_column]))

        # Concatenar los datos codificados con el DataFrame original y eliminar la columna original
        data = pd.concat([data.drop(columns=[t_column]), t_encoded_df], axis=1)
        print(f"Columnas creadas por la codificación One-Hot de '{t_column}':")
        print(t_encoded_df.columns)

    # Convertir la columna de raza "Race" a variables dummy usando OneHotEncoder si existe
    if race_column in data.columns:
        onehot_encoder_race = OneHotEncoder(sparse_output=False, drop='first')
        race_encoded = onehot_encoder_race.fit_transform(data[[race_column]])

        # Convertir el resultado a un DataFrame y agregar los nombres de columnas
        race_encoded_df = pd.DataFrame(race_encoded, columns=onehot_encoder_race.get_feature_names_out([race_column]))

        # Concatenar los datos codificados con el DataFrame original y eliminar la columna original
        data = pd.concat([data.drop(columns=[race_column]), race_encoded_df], axis=1)
        print(f"Columnas creadas por la codificación One-Hot de '{race_column}':")
        print(race_encoded_df.columns)

    # Verificar las características después de la codificación
    print("Características finales después de la codificación:", data.columns)

    # Separar las características y la variable objetivo, excluyendo 'Roach Clasica' si existe
    if 'Roach Clasica' in data.columns:
        features = data.drop(columns=[target_column, 'Roach Clasica'])
    else:
        features = data.drop(columns=[target_column])

    target = data[target_column]

    # Confirmar el número de características
    print("Características finales para el entrenamiento:", features.columns)
    print("Número total de características:", len(features.columns))

    # Verificar si existen valores nulos
    print("Número de valores nulos en cada columna:")
    print(features.isnull().sum())

    # Llenado de valores nulos si existen
    if features.isnull().sum().any():
        features = features.fillna(features.mean())
        print("Valores nulos llenados con la media de cada columna.")

    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(features, target)

    # Dividir los datos en conjuntos de entrenamiento (60%) y prueba (40%)
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.4, random_state=42)

    # Confirmar el número de características
    print("Características finales para el entrenamiento:", features.columns)
    print("Número total de características:", len(features.columns))  # Debe ser 17

    # Guardar los nombres de las características en un archivo JSON
    feature_names = features.columns.tolist()
    import json
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    # Crear y entrenar el modelo
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Guardar el modelo entrenado en un archivo .pkl
    model_filename = r'C:\Users\agust\Documents\SmartRoachTest\modelo_smart_roach.pkl'
    try:
        joblib.dump(model, model_filename)
        print(f"Modelo guardado como '{model_filename}'")
    except Exception as e:
        print("Ocurrió un error al guardar el modelo:", e)
