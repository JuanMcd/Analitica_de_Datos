import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configurar la URI de MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Lista de posiciones
positions = ['delantero', 'mediocampista', 'defensa', 'arquero']

# Modelos a entrenar
models = {
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SGDClassifier": SGDClassifier(),
    "svm": svm.SVC(),
    "RandomForestClassifier": RandomForestClassifier(),
}

# Procesar cada posición
for position in positions:
    # Cargar los datos de características y etiquetas
    X = np.load(f"Features_{position}.npy")
    y = np.load(f"Labels_{position}.npy")

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, shuffle=True)

    # Escalar los datos
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar y evaluar cada modelo
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{position}"):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Registrar parámetros, métricas y el modelo
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, f"model_{position}")

            print(f"Modelo: {model_name}, Posición: {position}, Accuracy: {accuracy}")
