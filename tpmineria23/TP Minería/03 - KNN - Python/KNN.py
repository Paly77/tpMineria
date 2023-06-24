import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split        # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier          # Para usar el clasificador KNN para la clasificación y regresión
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay  # Para evaluar la precisión del modelo


# Importe de datos
path = "CSV/clientes_sin_nulos.csv"
data = pd.read_csv(path, sep=',', encoding='latin-1')

print(data.head())      # Para corroborar que se importó correctamente


# Preparación del algoritmo KNN
list_x = ['IngresoAnual', 'TotalHijos', 'Propietario', 'CantAutomoviles', 'Edad']
scaler = StandardScaler()
scaler.fit(data[list_x])
X = scaler.transform(data[list_x])

y = data.ComproBicicleta

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.349, random_state=17, stratify = y)


# Cálculo de la K que vamos a utilizar
for k in range(1, 20, 1):
    knn = KNeighborsClassifier(k, weights= "distance")
    knn.fit(X_train, y_train)
    pred_train = knn.predict(X_train)
    pred_test = knn.predict(X_test)
    print("_______________________________________")
    print(f"K igual a {k}: ")
    print(f'Accuracy de K-NN train: {knn.score(X_train, y_train):.2f}')
    print(f"F1 de K-NN train: {f1_score(y_train,pred_train, average='macro'):.2f}")
    print(f'Accuracy de K-NN test: {knn.score(X_test, y_test):.2f}')
    print(f"F1 de K-NN test: {f1_score(y_test, pred_test,average='macro'):.2f}")
# Conclusión: nos quedamos con K = 12


# Algoritmo KNN
knn = KNeighborsClassifier(12, weights= "distance")
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

cm = confusion_matrix(y_test, pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels = knn.classes_)
cm_display.plot()

plt.xticks(rotation=90)
plt.show()