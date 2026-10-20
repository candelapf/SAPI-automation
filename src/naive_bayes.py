import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar los datos
df = pd.read_csv('datos_sapi_pediatria.csv')
# Limpieza de columnas
df.columns = df.columns.str.strip().str.lower()


# 2. Selección de variables
# Utilizamos variables continuas, por lo que el modelo GaussianNB es el adecuado
X = df[['fc', 'fr', 'spo2', 'apariencia']]
y = df['riesgo_real']

# 3. División 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenamiento del modelo Naive Bayes (Gaussiano)
modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)

# 5. Evaluación de resultados
predicciones_nb = modelo_nb.predict(X_test)
precision_global = accuracy_score(y_test, predicciones_nb)

print(f"Precisión Global Naive Bayes: {precision_global:.2%}")
print("\nInforme de Clasificación:")
print(classification_report(y_test, predicciones_nb))

# 6. Matriz de Confusión para comparar con los anteriores
cm = confusion_matrix(y_test, predicciones_nb)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Alto', 'Bajo', 'Medio'], 
            yticklabels=['Alto', 'Bajo', 'Medio'])
plt.title('Matriz de Confusión: Naive Bayes')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()