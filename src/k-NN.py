import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. Cargar los datos
df = pd.read_csv('datos_sapi_pediatria.csv')

# Limpieza de columnas
df.columns = df.columns.str.strip().str.lower()

# 2. Selección de variables
X = df[['fc', 'fr', 'spo2', 'apariencia']]
y = df['riesgo_real']

# 3. Normalización 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. División 80/20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Entrenamiento de k-NN (k=5 es el estándar)
modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(X_train, y_train)

# 6. Evaluación
predicciones_knn = modelo_knn.predict(X_test)
print(f"Precisión k-NN: {accuracy_score(y_test, predicciones_knn):.2%}")
print("\nInforme de Clasificación:")
print(classification_report(y_test, predicciones_knn))