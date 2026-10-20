import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Cargar los datos que generamos antes
df = pd.read_csv('datos_sapi_pediatria.csv')

# 2. Seleccionar variables de entrada (X) y lo que queremos predecir (y)
# Usaremos las constantes vitales para predecir el "Nivel_Riesgo"
X = df[['FC', 'FR', 'SpO2', 'Apariencia']]
y = df['Riesgo_Real']

# 3. Dividir datos: 80% para entrenar el modelo y 20% para ponerlo a prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear y entrenar el Árbol de Decisión
# Limitamos la profundidad (max_depth) del árbol
modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_train, y_train)

# 5. Probar el modelo con los datos que no ha visto (X_test)
predicciones = modelo.predict(X_test)

# 6. Ver qué tan bueno es
print(f"Precisión del modelo: {accuracy_score(y_test, predicciones) * 100:.2f}%")
print("\nInforme de Clasificación:")
print(classification_report(y_test, predicciones))

# 7. Visualización del árbol
plt.figure(figsize=(15,10))
plot_tree(modelo, feature_names=X.columns, class_names=modelo.classes_, filled=True)
plt.title("Árbol de Decisión: Lógica del Sistema SAPI")
plt.show()

# Crear un DataFrame con los datos de prueba y las predicciones
df_resultados = X_test.copy()
df_resultados['SAPI_Score'] = df.loc[X_test.index, 'SAPI_Score']
df_resultados['Riesgo_Real'] = y_test
df_resultados['Riesgo_Predicho'] = predicciones

# Guardar a CSV
df_resultados.to_csv('resultados_modelo_sapi.csv', index=False)

print("Archivo 'resultados_modelo_sapi.csv' generado.")