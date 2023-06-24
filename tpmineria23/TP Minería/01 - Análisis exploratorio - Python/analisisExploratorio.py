import pandas as pd                 # Análisis de datos
import numpy as np                  # Funciones matemáticas
import matplotlib.pyplot as plt     # Gráficas
import seaborn as sns               # Gráficas mejoradas
import statsmodels.api as sm        # ?
import pylab as py                  # ?

# Importe de datos
path = "C:/Users/paulo/Downloads/clie/clientes.csv" 
data = pd.read_csv(path, sep=',', encoding='latin-1')



# Previsualización de datos
print(data.head())                   # Para corroborar que se importó correctamente
data.info()                          # Para ver las columnas y tipos de datos de forma general
print(data.describe())               # Para ver algunas medidas estadísticas



# Generamos todos los valores que adquieren las columnas "Propietario" y "ComproBicicleta"
# con el objetivo de asegurar que solo toman los valores 0 y 1.
print(data["Propietario"].value_counts())
print(data["ComproBicicleta"].value_counts())

#############################################################################################
#Generamos el grafuco circular de los que compraron  o no
# utilizando la variable comprobicicleta
conteouno=data["ComproBicicleta"].value_counts()
plt.pie(conteouno, autopct="%0.2f %%")
plt.title("Grafico circular de ComproBicicleta")
plt.show()


conteodos=data["EstadoCivil"].value_counts()
plt.pie(conteodos, autopct="%0.2f %%", labels=["C","S","D"])
plt.title("Grafico circular de EstadoCivil")
plt.show()

data["Educacion"].value_counts().plot(kind='bar')
plt.show()

data["Genero"].value_counts().plot(kind='bar')
plt.show()

conteotres=data["Genero"].value_counts()
plt.pie(conteotres, autopct="%0.2f %%")
plt.title("Grafico circular de Genero")
plt.show()

##############################################################################################
#Analisis Multivariante
#Relacion entre Comprobicicleta y Edad
edad=data['Edad']
compro=data['ComproBicicleta']
i=0
arr=[edad,compro]
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print(arr)
print('la edad minima es:',np.amin(edad),'edad maxima',np.amax(edad))


  




#######################################################################################################

# Diagrama de caja y bigotes con cuatro subplots
sns.set_style("whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(11, 6))               # Crea la figura con dos filas y dos columnas
plt.subplots_adjust(bottom=0.2, wspace=0.5, hspace=0.4)     # Ajusta los márgenes para evitar solapamientos

sns.boxplot(data=data, x="IngresoAnual", ax=ax[0, 0], color="steelblue")
ax[0, 0].set_title("Boxplot de IngresoAnual")

sns.boxplot(data=data, x="Edad", ax=ax[0, 1], color="steelblue")
ax[0, 1].set_title("Boxplot de la Edad")

sns.boxplot(data=data, x="TotalHijos", ax=ax[1, 0], color="steelblue")
ax[1, 0].set_title("Boxplot de TotalHijos")

sns.boxplot(data=data, x="CantAutomoviles", ax=ax[1, 1], color="steelblue")
ax[1, 1].set_title("Boxplot de CantAutomoviles")

plt.show()



# Matriz S para analizar si hay alguna relación con "ComproBicicletas"
data_numeric = data.select_dtypes(include=['float64', 'int64'])     # Nuevo DataFrame con solo columnas numéricas
print(data_numeric.corr())

# Matriz R para analizar la relación de otra forma
# Debido a que los datos varían mucho, no es muy recomendable ver esta matriz
print(data_numeric.cov())



# Gráficas de dispersión
sns.set_style("whitegrid")
fig, ax = plt.subplots(2, 3, figsize=(11, 6))
plt.subplots_adjust(bottom=0.2, wspace=0.5, hspace=0.4)

sns.scatterplot(data=data, x="Edad", y="IngresoAnual", ax=ax[0, 0], color="steelblue", hue="ComproBicicleta")
ax[0, 0].set_title("Edad vs. IngresoAnual")

sns.scatterplot(data=data, x="Edad", y="TotalHijos", ax=ax[0, 1], color="steelblue", hue="ComproBicicleta")
ax[0, 1].set_title("Edad vs. TotalHijos")

sns.scatterplot(data=data, x="Edad", y="CantAutomoviles", ax=ax[0, 2], color="steelblue", hue="ComproBicicleta")
ax[0, 2].set_title("Edad vs. CantAutomoviles")

sns.scatterplot(data=data, x="TotalHijos", y="IngresoAnual", ax=ax[1, 0], color="steelblue", hue="ComproBicicleta")
ax[1, 0].set_title("TotalHijos vs. IngresoAnual")

sns.scatterplot(data=data, x="TotalHijos", y="CantAutomoviles", ax=ax[1, 1], color="steelblue", hue="ComproBicicleta")
ax[1, 1].set_title("TotalHijos vs. CantAutomoviles")

sns.scatterplot(data=data, x="IngresoAnual", y="CantAutomoviles", ax=ax[1, 2], color="steelblue", hue="ComproBicicleta")
ax[1, 2].set_title("IngresoAnual vs. CantAutomoviles")

plt.show()



# Rellenamos los datos nulos de "IngresoAnual"
# Primero analizamos con qué valor imputar, si la media o mediana
sns.histplot(data=data, x="IngresoAnual", color="steelblue") # Histograma de la variable "Velocidad"
plt.title("Histograma del ingreso anual")
plt.show()

# El resultado no es una distribución normal. Luego, imputamos con la mediana
c_ingAnual_na = data["IngresoAnual"].isna()                             # Generamos la serie booleana
print(data[c_ingAnual_na])                                              # Imprime todos los nulos (en este caso, 10)

data.loc[c_ingAnual_na, "IngresoAnual"] = data.IngresoAnual.median()    # Imputamos con la mediana

c_ingAnual_na2 = data["IngresoAnual"].isna()                          # Generamos otra serie booleana para corroborar
print(data[c_ingAnual_na2])                                           # No imprime nada (empty DataFrame)



# Valores atípicos
# Según el diagrama de caja y bigotes, los valores atípicos para cada uno se dan si el ingreso anual es 
# mayor a $149000, si la edad es mayor a 84 años y si tiene más de 3 autos
print(data[data["IngresoAnual"] > 149000])      # 100 valores
print(data[data["Edad"] > 84])                  # 28 valores
print(data[data["CantAutomoviles"] > 3])        # 480 valores



# Exportamos el CSV ya sin datos nulos
data.to_csv('clientes_sin_nulos.csv', index=False)
