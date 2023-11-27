import pandas as pd #Libreria para manipulara datos
import numpy as np #libreria para anlizar datos menejar arreglos, matrices, 

#Hoja de datos 
datasheet = pd.read_csv("datasheet.csv")

from sklearn.preprocessing import OrdinalEncoder #funcion para codificar datos a ordinal
from scipy.stats import ttest_ind, chi2_contingency # funcion para realizar la prueba t student y rechazar hipotesis

# Categorias por codificar
categoria_frecuencia = ['Nunca', 'Raramente', 'Ocasionalmente', 'Frecuentemente', 'Siempre']
categoria_rendimiento = ['Ha empeorado significativamente', 'Ha disminuido ligeramente', 'Ha mejorado ligeramente', 'Ha mejorado significativamente']
categoria_herramientas = ['Google', 'ChatGPT']
categoria_comprendimiento = ['No, en absoluto', 'No, en cierta medida', 'No estoy seguro/a', 'Sí, en cierta medida', 'Sí, considerablemente']
categoria_influencia = ['No ha tenido ningún efecto', 'Sí, de manera negativa', 'No estoy seguro/a', 'Sí, de manera neutra', 'Sí, de manera positiva']
categoria_tendencia = ['No, definitivamente no', 'No, en cierta medida', 'No estoy seguro/a', 'Sí, en cierta medida', 'Sí, definitivamente']

# Codificador
codificador_4 = OrdinalEncoder(
    categories=[
        categoria_frecuencia
    ]
)

# Codificador
codificador_5 = OrdinalEncoder(
    categories=[
        categoria_rendimiento
    ]
)

# Codificador
codificador_6 = OrdinalEncoder(
    categories=[
        categoria_herramientas
    ]
)

# Codificador
codificador_7 = OrdinalEncoder(
    categories=[
        categoria_comprendimiento
    ]
)

# Codificador
codificador_8 = OrdinalEncoder(
    categories=[
        categoria_influencia
    ]
)

# Codificador
codificador_9 = OrdinalEncoder(
    categories=[
        categoria_tendencia
    ]
)

# Codificar los datos como de costumbre
datos_ord_4 = codificador_4.fit_transform(datasheet[["¿Con qué frecuencia utilizas herramientas de IA en tu proceso de aprendizaje?"]])
datos_ord_5 = codificador_5.fit_transform(datasheet[["¿Cómo crees que el uso de IA ha afectado tu rendimiento académico?"]])
datos_ord_6 = codificador_6.fit_transform(datasheet[["¿ Usas mas Google o ChatGPT?"]])
datos_ord_7 = codificador_7.fit_transform(datasheet[["¿Sientes que la IA ha mejorado tu capacidad para comprender y aplicar conceptos académicos?"]])
datos_ord_8 = codificador_8.fit_transform(datasheet[["¿Crees que la IA ha influido en tus preferencias de estudio y enfoques de aprendizaje?"]])
datos_ord_9 = codificador_9.fit_transform(datasheet[["¿Consideras que el uso de IA en la educación es una tendencia positiva y beneficiosa?"]])


#Correcciones: 
columnas_ordinales = [
    datos_ord_4, datos_ord_5, datos_ord_6, datos_ord_7, datos_ord_8, datos_ord_9
]

# Hipótesis alternativa: El uso frecuente de herramientas de IA mejora el rendimiento academico
# Hipótesis nula: El uso frecuente de herramientas de IA no mejora el rendimiento academico
hipotesis_1 = ttest_ind(columnas_ordinales[0], columnas_ordinales[1])

#Imprmir el p value
print(" p-valor:", hipotesis_1.pvalue)

if hipotesis_1.pvalue > 0.05:
    print("No hay evidencia suficiente para rechazar la hipótesis nula.")
    print("No hay diferencia significativa en el rendimiento académico.")
else:
    print("Se rechaza la hipótesis nula.")
    print("Se acepta la hipotesis alternativa")
    print("Hay evidencia significativa de una diferencia en el rendimiento académico.")