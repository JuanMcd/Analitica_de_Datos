#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient


#Se cargan los datos desde el archivo .xlsx
datos = pd.read_excel('Premier League Stats_2020.xlsx')
#Se eliminan las columnas "Jersey Number", "Nationality" ya que no se tendrán en cuenta
data = datos.drop(columns=["Jersey Number", "Nationality"])
#%%------------ Se inicia la limpieza de datos ---------------------------
#Eliminar datos inválidos, debido a que estos datos no aportan información relevante para nuestro estudio
Datos_invalidos_ = data[(data["Appearances"] == 0)]
data = data.drop(Datos_invalidos_.index)
#Para agrupaciones y cálculos, se rellenan los datos vacios con cero
data = data.fillna(0)

#%%------------------------Procesamiento de datos para la vista de jugadores----------------------------------

def plot_columns(data, position, columns_to_plot):
    position_data = data[data['Position'] == position]
    num_columns = len(columns_to_plot)
    fig, axes = plt.subplots(nrows=num_columns, figsize=(15, 7 * num_columns))
    
    for i, column in enumerate(columns_to_plot):
        ax = axes[i] if num_columns > 1 else axes
        filtered_data = position_data[position_data[column] > 0]
        filtered_data = filtered_data.sort_values(by=column, ascending=False)
        num_players = min(len(filtered_data), 13)
        
        # Configuraciones de fondo y color
        ax.set_facecolor('black') 
        fig.patch.set_facecolor('black')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(axis='x', colors='white') 
        ax.tick_params(axis='y', colors='white')  
        
        for spine in ax.spines.values():
            spine.set_color('white')
        
        # Dibujar la gráfica
        ax.bar(x=filtered_data['Name'].head(num_players), height=filtered_data[column].head(num_players), color='blue')  # Elige el color de las barras
        ax.set_title(f'{position} vs {column}')
        ax.set_xlabel('Nombre')
        ax.set_ylabel(column)
    
    plt.tight_layout()
    return fig



# Definir columnas disponibles por posición
columnas_por_posicion = {
    'Forward': ['Appearances', 'Wins', 'Losses', 'Goals', 'Goals per match', 'Headed goals','Goals with right foot','Goals with left foot','Penalties scored','Freekicks scored','Shots', 'Shots on target', 'Shooting accuracy %', 'Hit woodwork', 'Big chances missed', 'Tackles', 'Tackle success %', 'Blocked shots', 'Interceptions', 'Clearances', 'Headed Clearance', 'Recoveries', 'Duels won', 'Duels lost', 'Aerial battles won', 'Aerial battles lost', 'Errors leading to goal', 'Assists', 'Passes', 'Passes per match', 'Big chances created', 'Crosses', 'Cross accuracy %', 'Through balls', 'Accurate long balls', 'Yellow cards', 'Red cards', 'Fouls', 'Offsides'],
    'Midfielder': ['Appearances', 'Wins', 'Losses', 'Goals', 'Goals per match', 'Headed goals','Goals with right foot','Goals with left foot','Penalties scored','Freekicks scored','Shots', 'Shots on target', 'Shooting accuracy %', 'Hit woodwork', 'Big chances missed', 'Clean sheets', 'Goals conceded', 'Tackles', 'Tackle success %', 'Blocked shots', 'Interceptions', 'Clearances', 'Headed Clearance', 'Recoveries', 'Duels won', 'Duels lost', 'Aerial battles won', 'Aerial battles lost', 'Own goals', 'Errors leading to goal', 'Assists', 'Passes', 'Passes per match', 'Big chances created', 'Crosses', 'Cross accuracy %', 'Through balls', 'Accurate long balls', 'Yellow cards', 'Red cards', 'Fouls', 'Offsides'],
    'Defender': ['Appearances', 'Wins', 'Losses', 'Goals', 'Goals per match', 'Headed goals','Goals with right foot','Goals with left foot','Shots', 'Shots on target', 'Shooting accuracy %', 'Hit woodwork', 'Big chances missed', 'Clean sheets', 'Goals conceded', 'Tackles', 'Tackle success %', 'Last man tackles', 'Blocked shots', 'Interceptions', 'Clearances', 'Headed Clearance', 'Clearances off line', 'Recoveries', 'Duels won', 'Duels lost', 'Aerial battles won', 'Aerial battles lost', 'Own goals', 'Errors leading to goal', 'Assists', 'Passes', 'Passes per match', 'Big chances created', 'Crosses', 'Cross accuracy %', 'Through balls', 'Accurate long balls', 'Yellow cards', 'Red cards', 'Fouls', 'Offsides'],
    'Goalkeeper': ['Appearances', 'Wins', 'Losses', 'Clean sheets', 'Goals conceded', 'Own goals', 'Errors leading to goal', 'Assists', 'Passes', 'Passes per match', 'Accurate long balls', 'Saves', 'Penalties saved', 'Punches', 'High Claims', 'Catches', 'Sweeper clearances', 'Throw outs', 'Goal Kicks', 'Yellow cards', 'Red cards', 'Fouls']
}

#%% ------------------------Procesamiento de datos para la vista de equipos----------------------------------

#Función para obtener los goles concedidos de los equipos para el rendimiento defensivo
def get_max_goals_conceded(data, team_name):
    team_data = data[data['Club'] == team_name]
    if not team_data.empty:
        max_goals_conceded = team_data['Goals conceded'].max()
        return max_goals_conceded
    return 0

#Función para obtener la gráfica de distribución de edad de cada equipo
def plot_age_distribution_by_position(data, team_name):
    team_data = data[data['Club'] == team_name]
    fig, ax = plt.subplots()
    ax.set_facecolor('black')  
    fig.patch.set_facecolor('black') 
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')  
    ax.tick_params(axis='y', colors='white') 
    for spine in ax.spines.values():
        spine.set_color('white')

    sns.boxplot(x='Position', y='Age', data=team_data, ax=ax, palette="Set3")  
    ax.set_title(f'Distribución de Edad por Posición en {team_name}')
    ax.set_xlabel('Posición')
    ax.set_ylabel('Edad')
    return fig

#Función para realizar gráficos de radar
def plot_radar_chart(data, team_name):
    team_data = data[data['Club'] == team_name]
    avg_goals_scored = team_data['Goals'].mean()
    avg_goals_conceded = team_data['Goals conceded'].mean()
    avg_interceptions = team_data['Interceptions'].mean()
    avg_tackles = team_data['Tackle success %'].mean()
    avg_clearances = team_data['Clearances'].mean()

    labels=np.array(['Goles Marcados', 'Goles Concedidos', 'Intercepciones', '% Entradas Exitosas', 'Despejes'])
    stats=np.array([avg_goals_scored, avg_goals_conceded, avg_interceptions, avg_tackles*100, avg_clearances])

    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()

    stats=np.concatenate((stats,[stats[0]]))
    angles+=angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, stats, color='red', alpha=0.25)
    ax.plot(angles, stats, color='red', marker='o')  
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    return fig

#Función para obtener los goleadores del equipo
def list_top_scorers(data, team_name, max_players=8):
    team_data = data[data['Club'] == team_name]
    top_scorers = team_data.sort_values(by='Goals', ascending=False)
    top_scorers = top_scorers.head(max_players)
    top_scorers = top_scorers[['Name', 'Goals', 'Headed goals', 'Goals with right foot', 'Goals with left foot']]
    return top_scorers


#%% IMPORTACIÓN DEL MODELADO
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
uri_modelo_delantero = "mlflow-artifacts:/0/8c34999277c34873bc9543f40c1a902b/artifacts/model_delantero"
uri_modelo_mediocampista = "mlflow-artifacts:/0/8c714051be62404db860cec9b1c1dc5c/artifacts/model_mediocampista"
uri_modelo_defensa = "mlflow-artifacts:/0/08804e567c294e69b81548e9c78f1d9a/artifacts/model_defensa"
uri_modelo_arquero = "mlflow-artifacts:/0/65b4c4ad1e8b4b86a12e284e2f2956a7/artifacts/model_arquero"

modelo_delantero = mlflow.pyfunc.load_model(uri_modelo_delantero)
modelo_mediocampista = mlflow.pyfunc.load_model(uri_modelo_mediocampista)
modelo_defensa = mlflow.pyfunc.load_model(uri_modelo_defensa)
modelo_arquero = mlflow.pyfunc.load_model(uri_modelo_arquero)



# %% -------------------------------------------------------------CONFIGURACIÓN Y DESPLIEGUE DE STREAMLIT --------------------------------------
# Configuración de la página
st.set_page_config(layout="wide")
st.title('Estadísticas de la Premier League temporadas 2006 - 2020')

# Sidebar para selección
st.sidebar.title('Configuración de Visualización')
selected_position = st.sidebar.selectbox('Selecciona una posición', list(columnas_por_posicion.keys()))
if selected_position:
    available_columns = columnas_por_posicion[selected_position]
    selected_stats = st.sidebar.multiselect('Selecciona estadísticas para visualizar', available_columns, default=available_columns[0])
else:
    st.sidebar.write("Selecciona una posición para ver las estadísticas disponibles.")

# Pestañas para diferentes vistas
tab1, tab2, tab3 = st.tabs(["Vista de Equipo", "Estadísticas de Jugadores por posición", "Clasificación de jugadores según su rendimiento"])


with tab1:
    st.header("Análisis Comparativo de Equipos")
    equipo = st.selectbox('Selecciona un equipo para análisis detallado:', data['Club'].unique())
    #------------------------------------------ COMPARACIÓN DE GOLES VS GOLES CONCEDIDOS ---------------------------------------------------
    goles_concedidos_correctos = get_max_goals_conceded(data, equipo)
    equipo_data = data[data['Club'] == equipo]
    goles_marcados = equipo_data['Goals'].sum()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('black')  
    fig.patch.set_facecolor('black') 
        
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')  
    ax.tick_params(axis='y', colors='white')  
    for spine in ax.spines.values():
        spine.set_color('white')
    
    ax.bar(['Goles Marcados', 'Goles Concedidos'], [goles_marcados, goles_concedidos_correctos], color=['green', 'red'])
    ax.set_title('Rendimiento Ofensivo y Defensivo')
    st.pyplot(fig)
    # ---------------------------------------------------- DISTRIBUCIÓN DE EDAD ----------------------------------------
    if st.button('Mostrar Distribución de Edad por Posición'):
        fig = plot_age_distribution_by_position(data, equipo)
        st.pyplot(fig)


    # --------------------------------------------------- EFICACIA DEFENSIVA Y OFENSIVA ---------------------------------------
    st.header("Eficacia Defensiva y Ofensiva")
    fig = plot_radar_chart(data, equipo)
    st.pyplot(fig)

    #---------------------------------------------------- GOLEADORES DEL EQUIPO ---------------------------------
    top_scorers_df = list_top_scorers(data, equipo)
    st.write("Goleadores del Equipo:")
    st.dataframe(top_scorers_df)


with tab2:
    st.header("Estadísticas de Jugadores según la posición seleccionada")
    if selected_stats:
        fig = plot_columns(data, selected_position, selected_stats)
        st.pyplot(fig) 
    else:
        st.write("Selecciona al menos una estadística para visualizar.")

with tab3:
    st.header("Modelo de clasificación para clasificación de jugadores según su posición")

    # Selección de la posición para la predicción
    position = st.selectbox("Elige la posición del jugador:", ["delantero", "mediocampista", "defensa", "arquero"])
    
    # Diccionario con campos por posición
    fields_by_position = {
        'arquero': ['Appearances', 'Wins', 'Interceptions', 'Clearances', 'Recoveries', 'Accurate long balls', 'Saves', 'Penalties saved', 'Catches', 'High Claims'],
        'defensa': ['Appearances', 'Wins', 'Clearances', 'Headed Clearance', 'Interceptions', 'Assists', 'Passes', 'Tackle success %', 'Blocked shots', 'Duels won', 'Aerial battles won', 'Cross accuracy %', 'Accurate long balls'],
        'mediocampista': ['Appearances', 'Goals', 'Wins', 'Interceptions', 'Freekicks scored', 'Assists', 'Passes', 'Accurate long balls', 'Through balls', 'Shooting accuracy %', 'Duels won', 'Cross accuracy %'],
        'delantero': ['Appearances', 'Goals', 'Wins', 'Penalties scored', 'Freekicks scored', 'Shooting accuracy %', 'Duels won', 'Aerial battles won', 'Cross accuracy %', 'Offsides']
    }

    # Crear inputs para cada atributo necesario
    input_data = {}
    for field in fields_by_position[position]:
        input_data[field] = st.number_input(f"Enter {field}:", value=0)

    # Botón para realizar la predicción
    if st.button("Predecir Categoría"):
        # Organizar los inputs en el orden correcto para el modelo
        feature_values = np.array([list(input_data.values())])
        # Seleccionar el modelo basado en la posición
        model = None
        if position == "delantero":
            model = modelo_delantero
        elif position == "mediocampista":
            model = modelo_mediocampista
        elif position == "defensa":
            model = modelo_defensa
        elif position == "arquero":
            model = modelo_arquero

        # Realizar la predicción
        prediction = model.predict(feature_values)
        prediction_category = {0: 'alto', 1: 'bajo', 2: 'medio'}.get(prediction[0], "Unknown")
        st.subheader(f"Predicción de Categoría de Rendimiento:")
        st.write(f"**Posición:** {position.capitalize()}")
        st.write(f"**Categoría de Rendimiento Predicha:** **{prediction_category}**")
        st.success(f"La categoría de rendimiento predicha para el {position} es: {prediction_category}")


