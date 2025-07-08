# -*- coding: utf-8 -*-
# Cargamos librerías principales
import numpy as np
import pandas as pd
import streamlit as st
import pickle

# --- Carga del Modelo ---
# Se cargan los objetos guardados: el modelo, la lista de variables y el escalador.
try:
    with open('modelo-ensamble-reg.pkl', 'rb') as file:
        model_stack, variables, min_max_scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Error: No se encontró el archivo del modelo 'modelo-ensamble-reg.pkl'.")
    st.stop()

# --- Interfaz Gráfica de Streamlit ---
st.title('Predicción de la demanda mensual por producto')
st.markdown("Introduce las características del producto y del período para predecir la demanda.")

# Creación de columnas para una mejor disposición
col1, col2 = st.columns(2)

with col1:
    EnglishProductName = st.selectbox('Producto', (variables[7:140])) # Obtenemos la lista de los 'selectbox'
    Category = st.selectbox('Categoría', ["Accessories", "Clothing", "Bikes"])
    Subcategory = st.selectbox('Subcategoría', ["Helmets", "Caps", "Jerseys", "'Road Bikes'", "'Mountain Bikes'", "Gloves", "Vests", "Shorts", "'Bottles and Cages'", "'Tires and Tubes'", "Socks", "'Bike Racks'", "Cleaners", "Fenders", "'Bike Stands'", "'Hydration Packs'", "'Touring Bikes'"])
    Color = st.selectbox('Color', ["Red", "Black", "Blue", "Multi", "Silver", "Yellow", "White", "nan"])
    Size = st.selectbox('Talla' , ["nan", "S", "M", "L", "XL", "62", "44", "48", "52", "56", "58", "60", "38", "42", "46", "40", "70", "50", "54"])

with col2:
    CalendarYear = st.selectbox('Año', [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    MonthNumberOfYear = st.selectbox('Mes', list(range(1, 13)))
    Is_Holiday_Season = st.selectbox('Es Temporada de Vacaciones', [True, False])
    ListPrice = st.slider('Precio de Venta ($)', min_value=0, max_value=4000, value=1000, step=10)
    Margin = st.slider('Margen de Ganancia ($)', min_value=0, max_value=1500, value=200, step=10)
    Product_Age_Months = st.slider('Edad del Producto (meses)', min_value=0, max_value=150, value=24, step=1)
    descuento_promedio_mes_anterior = st.slider('Descuento promedio mes anterior', min_value=0.0, max_value=1.0, value=0.1, step=0.01)

# Botón para ejecutar la predicción
if st.button('Predecir Demanda'):
    # --- Creación del DataFrame con TODOS los datos ---
    # Se incluyen las 12 variables capturadas.
    columnas = [
        'EnglishProductName', 'Category', 'Subcategory', 'ListPrice', 'Margin',
        'Color', 'Size', 'CalendarYear', 'MonthNumberOfYear', 'Is_Holiday_Season',
        'Product_Age_Months', 'descuento_promedio_mes_anterior'
    ]
    datos = [[
        EnglishProductName, Category, Subcategory, ListPrice, Margin,
        Color, Size, CalendarYear, MonthNumberOfYear, Is_Holiday_Season,
        Product_Age_Months, descuento_promedio_mes_anterior
    ]]
    data = pd.DataFrame(datos, columns=columnas)

    # --- Preparación de Datos ---
    # Copia para preparación
    data_preparada = data.copy()

    # 1. Normalización de variables numéricas (se usa el escalador ya entrenado)
    columnas_numericas = ['ListPrice', 'Margin']
    data_preparada[columnas_numericas] = min_max_scaler.transform(data_preparada[columnas_numericas])

    # 2. Conversión de la columna booleana
    data_preparada['Is_Holiday_Season'] = data_preparada['Is_Holiday_Season'].astype(bool)

    # 3. Creación de variables dummies
    columnas_categoricas = ['EnglishProductName', 'Category', 'Subcategory', 'Color', 'Size']
    data_preparada = pd.get_dummies(data_preparada, columns=columnas_categoricas, drop_first=False)

    # 4. Reindexar para asegurar que todas las columnas del modelo original estén presentes
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

    # --- Predicción ---
    try:
        prediccion = model_stack.predict(data_preparada)
        prediccion_redondeada = round(prediccion[0])

        st.success(f"**La demanda predicha es de: {prediccion_redondeada} unidades.**")
        st.write("---")
        st.write("### Detalles de la Predicción:")
        st.write("Datos de entrada:")
        st.dataframe(data)
        st.write("Datos preparados antes de la predicción (vista parcial):")
        st.dataframe(data_preparada.head())

    except Exception as e:
        st.error(f"Ocurrió un error al realizar la predicción: {e}")