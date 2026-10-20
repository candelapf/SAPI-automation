import pandas as pd
import numpy as np

# Configuración de la muestra
n_pacientes = 500
np.random.seed(42)

def generar_datos_sapi(n):
    data = []
    for _ in range(n):
        # 1. Generamos signos vitales
        frec_cardiaca = np.random.randint(60, 200)   # bpm
        frec_respiratoria = np.random.randint(10, 80) # rpm
        spo2 = np.random.randint(80, 100)            # %
        
        # 2. Lógica SAPI 
        
        # Puntos por Frecuencia Cardíaca (Peligro en extremos)
        if frec_cardiaca <= 90 or frec_cardiaca >= 160:
            puntos_fc = 2
        elif frec_cardiaca >= 140:
            puntos_fc = 1
        else:
            puntos_fc = 0
            
        # Puntos por Frecuencia Respiratoria (Peligro en extremos)
        if frec_respiratoria <= 20 or frec_respiratoria >= 60:
            puntos_fr = 2
        elif frec_respiratoria >= 40:
            puntos_fr = 1
        else:
            puntos_fr = 0
            
        # Puntos por Saturación de Oxígeno
        puntos_spo2 = 0 if spo2 >= 95 else (1 if spo2 >= 90 else 2)
        
        # Componente de Apariencia/Esfuerzo (basado en hipoxia y fatiga)
        # Si hay bradipnea o hipoxia severa, la apariencia suele ser peor (3)
        if spo2 < 90 or frec_respiratoria <= 20:
            apariencia = np.random.choice([2, 3])
        else:
            apariencia = np.random.choice([0, 1])
        
        sapi_total = puntos_fc + puntos_fr + puntos_spo2 + apariencia
        
        # Clasificación del riesgo
        riesgo = "Bajo" if sapi_total <= 3 else ("Medio" if sapi_total <= 6 else "Alto")
        
        data.append([frec_cardiaca, frec_respiratoria, spo2, apariencia, sapi_total, riesgo])
    
    return pd.DataFrame(data, columns=['FC', 'FR', 'SpO2', 'Apariencia', 'SAPI_Score', 'Riesgo_Real'])

# Crear el dataset
df_sapi = generar_datos_sapi(n_pacientes)

# Guardar a CSV
df_sapi.to_csv('datos_sapi_pediatria.csv', index=False)

print("¡Dataset corregido generado con éxito!")
print(df_sapi.head())