# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score
)
from sklearn.preprocessing import OneHotEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Definir una función para calcular la especificidad
def specificity_score_func(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return specificity
    else:
        return None

# Cargar los datos desde el archivo Excel
ruta_archivo = r'C:\Users\agust\Documents\SmartRoachTest\BasedeDatosRoach.xlsx'
data = pd.read_excel(ruta_archivo)

# Verificar los nombres de las columnas
print("Columnas disponibles en el DataFrame:")
print(data.columns.tolist())

# Especificar la columna objetivo y las columnas a ignorar o transformar
target_column = 'Nodos regionales positivos'  # Confirmado
text_column = 'Anatomia patologica'           # Confirmado
t_column = 'T'                                # Confirmado
race_column = 'Race'                          # Confirmado

# Asignar los nombres de las columnas según tu dataset
gleason_column = 'Gleason'          # Confirmado
psa_column = 'PSA'                  # Confirmado
clinical_stage_column = 'T'         # Usamos la columna 'T' como el estadio clínico

# Asegurarse de que estas columnas existen
required_columns = [gleason_column, psa_column, clinical_stage_column]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Las siguientes columnas faltan en el DataFrame: {missing_columns}")
else:
    print("Todas las columnas necesarias están presentes.")

# Transformar la columna de ganglios a binaria
data[target_column] = data[target_column].apply(lambda x: 1 if x > 0 else 0)

# Ignorar la columna de texto si existe
if text_column in data.columns:
    data = data.drop(columns=[text_column])

# Antes de aplicar la función, verifica los valores únicos de 'T'
print("Valores únicos de 'T' en el DataFrame:")
print(data['T'].unique())

# Definir la función para asignar grupos de riesgo basados en 'T' solamente
def asignar_grupo_riesgo_por_T(row):
    ct_stage = row[clinical_stage_column]
    ct_stage = str(ct_stage).strip()
    
    if ct_stage in ['1', 'T1', 'T1a', 'T1b', 'T1c']:
        return 'Muy bajo'
    elif ct_stage in ['2', 'T2', 'T2a']:
        return 'Bajo'
    elif ct_stage in ['2b', 'T2b', '2c', 'T2c']:
        return 'Intermedio'
    elif ct_stage in ['3a', 'T3a']:
        return 'Alto'
    elif ct_stage in ['3b', 'T3b', '4', 'T4']:
        return 'Muy alto'
    else:
        return 'No clasificado'

# Aplicar la función al dataset
data['Grupo_Riesgo'] = data.apply(asignar_grupo_riesgo_por_T, axis=1)

# Mostrar la distribución de los grupos de riesgo
print("\nDistribución de los Grupos de Riesgo:")
print(data['Grupo_Riesgo'].value_counts())

# Calcular la probabilidad según la fórmula de Roach
def calcular_probabilidad_roach(row):
    psa = row[psa_column]
    gleason_score = row[gleason_column]
    
    # Asegurarse de que PSA y Gleason son numéricos
    try:
        psa = float(psa)
        gleason_score = int(gleason_score)
    except ValueError:
        return None  # Manejar valores faltantes o inválidos
    
    # Calcular la probabilidad
    prob = (2/3) * psa + (max(gleason_score - 6, 0)) * 10
    prob = min(max(prob, 0), 100)  # Limitar entre 0 y 100%
    return prob

# Aplicar la función para calcular la probabilidad de Roach
data['Roach_Prob'] = data.apply(calcular_probabilidad_roach, axis=1)

# Verificar si hay valores nulos en 'Roach_Prob'
if data['Roach_Prob'].isnull().any():
    print("Existen valores nulos en 'Roach_Prob'. Se eliminarán estas filas.")
    data = data.dropna(subset=['Roach_Prob'])

# Binarizar la probabilidad de Roach utilizando un umbral (por ejemplo, 15%)
threshold = 15  # Ajusta el umbral según sea necesario
data['Roach_Binaria'] = data['Roach_Prob'].apply(lambda x: 1 if x >= threshold else 0)

# **Ingeniería de Características**

# a. Interacción entre PSA y Gleason
data['PSA_Gleason'] = data['PSA'] * data['Gleason']

# b. Transformación Logarítmica de PSA
data['Log_PSA'] = np.log1p(data['PSA'])  # Usamos log1p para manejar PSA = 0

# c. Binning de Edad
bins = [0, 50, 60, 70, 80, 100]
labels = ['<50', '50-59', '60-69', '70-79', '80+']
data['Edad_Binned'] = pd.cut(data['Edad'], bins=bins, labels=labels, right=False)

# d. Características derivadas de 'T'
data['T_Length'] = data['T'].apply(len)

# e. Interacción entre Race y Grupo_Riesgo
data['Race_Grupo_Riesgo'] = data['Race'] + '_' + data['Grupo_Riesgo']

# Codificar las nuevas características categóricas

# Codificar la columna 'Edad_Binned' con OneHotEncoder
if 'Edad_Binned' in data.columns:
    onehot_encoder_edad = OneHotEncoder(sparse_output=False, drop='first')
    edad_encoded = onehot_encoder_edad.fit_transform(data[['Edad_Binned']])
    edad_encoded_df = pd.DataFrame(edad_encoded, columns=onehot_encoder_edad.get_feature_names_out(['Edad_Binned']))
    data = pd.concat([data.drop(columns=['Edad_Binned']), edad_encoded_df], axis=1)
else:
    print("La columna 'Edad_Binned' no existe en el conjunto de datos.")

# Codificar la nueva columna 'Race_Grupo_Riesgo' si existe
if 'Race_Grupo_Riesgo' in data.columns:
    onehot_encoder_race_grupo = OneHotEncoder(sparse_output=False, drop='first')
    race_grupo_encoded = onehot_encoder_race_grupo.fit_transform(data[['Race_Grupo_Riesgo']])
    race_grupo_encoded_df = pd.DataFrame(race_grupo_encoded, columns=onehot_encoder_race_grupo.get_feature_names_out(['Race_Grupo_Riesgo']))
    data = pd.concat([data.drop(columns=['Race_Grupo_Riesgo']), race_grupo_encoded_df], axis=1)
else:
    print("La columna 'Race_Grupo_Riesgo' no existe en el conjunto de datos.")

# Ahora podemos codificar la columna 'T' y eliminarla del DataFrame
if t_column in data.columns:
    # Codificar la columna 'T' con OneHotEncoder
    onehot_encoder_t = OneHotEncoder(sparse_output=False, drop='first')  # Cambio aquí
    t_encoded = onehot_encoder_t.fit_transform(data[[t_column]])
    t_encoded_df = pd.DataFrame(t_encoded, columns=onehot_encoder_t.get_feature_names_out([t_column]))
    # Concatenar los datos codificados con el DataFrame original y eliminar la columna original
    data = pd.concat([data.drop(columns=[t_column]), t_encoded_df], axis=1)
else:
    print("La columna 'T' no existe en el conjunto de datos.")

# Codificar la columna 'Race' si existe
if race_column in data.columns:
    onehot_encoder_race = OneHotEncoder(sparse_output=False, drop='first')  # Cambio aquí
    race_encoded = onehot_encoder_race.fit_transform(data[[race_column]])
    race_encoded_df = pd.DataFrame(race_encoded, columns=onehot_encoder_race.get_feature_names_out([race_column]))
    data = pd.concat([data.drop(columns=[race_column]), race_encoded_df], axis=1)
else:
    print("La columna 'Race' no existe en el conjunto de datos.")

# Separar las características y la variable objetivo
features = data.drop(columns=[target_column, 'Roach_Prob', 'Roach_Binaria', 'Grupo_Riesgo'])
target = data[target_column]

# Llenar valores nulos si existen
if features.isnull().sum().any():
    features = features.fillna(features.mean())

# Definir el número de folds
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Inicializar listas para almacenar las métricas
metricas_list = []

# Inicializar diccionarios para almacenar las predicciones y verdaderos valores por grupo
predicciones_por_grupo = {grupo: {'y_true': [], 'y_prob': []} for grupo in ['Muy bajo', 'Bajo', 'Intermedio', 'Alto', 'Muy alto']}

fold_number = 1

for train_index, test_index in skf.split(features, target):
    print(f"\n=== Fold {fold_number} ===")
    
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    # Entrenar el modelo
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predicciones del modelo
    y_pred_model = model.predict(X_test)
    y_prob_model = model.predict_proba(X_test)[:, 1]
    
    # Obtener las predicciones de la fórmula de Roach
    y_pred_roach = data.iloc[test_index]['Roach_Binaria'].values
    y_prob_roach = data.iloc[test_index]['Roach_Prob'].values
    
    # Asignar el grupo de riesgo del conjunto de prueba
    grupos_test = data.iloc[test_index]['Grupo_Riesgo'].values
    
    # Iterar sobre cada muestra en el conjunto de prueba
    for i in range(len(y_test)):
        grupo = grupos_test[i]
        if grupo in predicciones_por_grupo:
            predicciones_por_grupo[grupo]['y_true'].append(y_test.iloc[i])
            predicciones_por_grupo[grupo]['y_prob'].append(y_prob_model[i])
    
    fold_number += 1

# Generar una figura para las curvas ROC por grupo
plt.figure(figsize=(10, 8))

# Iterar sobre cada grupo de riesgo y generar la curva ROC
for grupo in predicciones_por_grupo:
    y_true = np.array(predicciones_por_grupo[grupo]['y_true'])
    y_prob = np.array(predicciones_por_grupo[grupo]['y_prob'])
    
    if len(y_true) == 0:
        print(f"No hay datos para el grupo de riesgo '{grupo}'.")
        continue
    
    # Calcular las métricas
    precision = precision_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
    recall = recall_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
    specificity = specificity_score_func(y_true, (y_prob >= 0.5).astype(int))
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    
    print(f"\n=== Grupo de Riesgo: {grupo} ===")
    print(f"Precisión: {precision:.2f}")
    print(f"Sensibilidad (Recall): {recall:.2f}")
    print(f"Especificidad: {specificity:.2f}")
    print(f"AUC: {auc:.2f}")
    
    # Calcular la curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    # Graficar la curva ROC
    plt.plot(fpr, tpr, label=f'{grupo} (AUC = {auc:.2f})')

# Configurar la gráfica
plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC por Grupo de Riesgo (Agregadas)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Crear un DataFrame para almacenar las métricas agregadas
metricas_agregadas = []
for grupo in predicciones_por_grupo:
    y_true = np.array(predicciones_por_grupo[grupo]['y_true'])
    y_prob = np.array(predicciones_por_grupo[grupo]['y_prob'])
    
    if len(y_true) == 0:
        continue
    
    precision = precision_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
    recall = recall_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
    specificity = specificity_score_func(y_true, (y_prob >= 0.5).astype(int))
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    
    metricas_agregadas.append({
        'Grupo de Riesgo': grupo,
        'Precisión': precision,
        'Sensibilidad (Recall)': recall,
        'Especificidad': specificity,
        'AUC': auc
    })

metricas_df = pd.DataFrame(metricas_agregadas)

# Mostrar las métricas agregadas en una ventana (usando seaborn para una mejor visualización)
plt.figure(figsize=(10, 6))
sns.heatmap(metricas_df.set_index('Grupo de Riesgo'), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Métricas de Desempeño Agregadas por Grupo de Riesgo')
plt.show()

# Guardar el DataFrame de métricas si lo deseas
metricas_filename = r'C:\Users\agust\Documents\SmartRoachTest\metricas_desempeno_agregadas.xlsx'
try:
    metricas_df.to_excel(metricas_filename, index=False)
    print(f"Métricas agregadas guardadas como '{metricas_filename}'")
except Exception as e:
    print("Ocurrió un error al guardar las métricas agregadas:", e)

# Guardar el modelo entrenado del último fold si lo deseas
# Nota: En este caso, no se guarda un modelo específico ya que estamos agregando las predicciones.
# Sin embargo, puedes entrenar un modelo final con todo el dataset si lo prefieres.

# Entrenar un modelo final con todo el dataset
model_final = GradientBoostingClassifier(random_state=42)
model_final.fit(features, target)

# Guardar el modelo final
model_filename = r'C:\Users\agust\Documents\SmartRoachTest\modelo_smart_roach_final.pkl'
try:
    joblib.dump(model_final, model_filename)
    print(f"Modelo final guardado como '{model_filename}'")
except Exception as e:
    print("Ocurrió un error al guardar el modelo final:", e)
