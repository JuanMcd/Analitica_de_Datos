Análisis de Datos de la Premier League con Streamlit

Descripción del Proyecto

Este proyecto es un análisis detallado de las estadísticas de la Premier League utilizando Python, Pandas, Matplotlib, Seaborn y Streamlit. Se presentan visualizaciones interactivas y modelos de clasificación de jugadores según su rendimiento.

Características Principales:

Limpieza y procesamiento de datos estadísticos de jugadores de la Premier League.

Visualización de desempeño de jugadores y equipos con Streamlit.

Predicciones de clasificación de jugadores utilizando Machine Learning con modelos como Random Forest, SVM y KNN.

Uso de MLflow para gestionar modelos.

Tecnologías Utilizadas

Python (Numpy, Pandas, Matplotlib, Seaborn)

Streamlit (para la interfaz interactiva)

Scikit-Learn (para entrenar modelos de clasificación)

MLflow (para el seguimiento y gestión de modelos)

OpenPyXL (para carga de datos en Excel)

Instalación y Configuración

Para ejecutar este proyecto en tu máquina local, sigue los siguientes pasos:

1️⃣ Clonar el Repositorio

git clone https://github.com/tu_usuario/nombre_repositorio.git
cd nombre_repositorio

2️⃣ Instalar Dependencias

Ejecuta el siguiente comando para instalar los paquetes necesarios:

pip install -r requirements.txt

3️⃣ Ejecutar la Aplicación en Streamlit

Para iniciar la aplicación, ejecuta:

streamlit run Analitica.py

La interfaz de usuario se abrirá en tu navegador.

Uso del Proyecto

📊 Visualización de Estadísticas

Vista de Equipos: Comparación de goles marcados y concedidos, distribución de edad y eficacia defensiva/ofensiva.

Vista de Jugadores: Gráficos interactivos según la posición seleccionada.

🤖 Clasificación de Jugadores con Machine Learning

Modelos entrenados con datos de diferentes posiciones (delanteros, mediocampistas, defensas, arqueros).

Se pueden ingresar estadísticas de un jugador y obtener su clasificación en "Alto", "Medio" o "Bajo" rendimiento.

Modelado de Machine Learning

El archivo modelado.py entrena y evalúa varios modelos de clasificación:

Decision Tree

Random Forest

SVM

KNN

SGDClassifier

Cada modelo se entrena con datos de features_*.npy y labels_*.npy.

📁 Estructura del Proyecto


├── Analitica.py          # Script principal con visualización Streamlit

├── modelado.py           # Entrenamiento de modelos de clasificación

├── requirements.txt      # Dependencias del proyecto

├── Descripcion_Analitica.ipynb # Análisis exploratorio en Jupyter Notebook

├── features_arquero.npy  # Features para arqueros

├── features_defensa.npy  # Features para defensas

├── features_delantero.npy # Features para delanteros

├── features_mediocampista.npy # Features para mediocampistas

├── labels_arquero.npy    # Etiquetas para arqueros

├── labels_defensa.npy    # Etiquetas para defensas

├── labels_delantero.npy  # Etiquetas para delanteros

├── labels_mediocampista.npy # Etiquetas para mediocampistas


📢 Contribuciones

Si deseas contribuir, abre un issue o haz un pull request con mejoras.

📌 Autor

Proyecto desarrollado por Juan Manuel Calvo Duque.
