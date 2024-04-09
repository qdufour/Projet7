# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to make API request and get prediction
def get_prediction(data):
    api_url = "https://oc-projet7-api.azurewebsites.net/predict"
    test_data_json = data.drop(columns=['SK_ID_CURR']).to_json(orient='records')
    response = requests.post(api_url, json={'test_data': test_data_json})
    try:
        result = response.json()
        prediction_score = result['prediction'] 
        return prediction_score
    except Exception as e:
        st.error(f"Error getting prediction: {e}")
        return None

def main():
    df = pd.read_csv('./data/data.csv')
    # Initialize session state
    if 'id' not in st.session_state:
        st.session_state.id = None
        st.session_state.score = None

    st.sidebar.header("Client ID")
    id_client = st.sidebar.selectbox("Sélection du client", df["SK_ID_CURR"])
    
    example_idx = np.searchsorted(df["SK_ID_CURR"], id_client)
    instance = df[example_idx:example_idx+1]

    if st.button('Get Predictions'):
        st.session_state.id = id_client
        st.session_state.score = get_prediction(instance)[0]

    if st.session_state.score != None and st.session_state.id == id_client:
        score = st.session_state.score
        # Titre de l'application
        st.title("Tableau de Bord Client")

        # Afficher un tableau avec les informations des clients
        st.write("### Informations du client :")
        st.dataframe(instance)
        
        # Gauge Score
        threshold = 0.5
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Score Client :", 'font': {'size': 24}},
            delta={'reference': threshold},
            gauge={
                'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "#000000"},  
                'bar': {'color': '#000000', 'thickness': 0.2},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#000000",  
                'steps': [
                    {'range': [0, threshold], 'color': '#FF6347'},
                    {'range': [threshold, 1], 'color': '#32CD32'}],
                'threshold': {
                    'line': {'color': "#000000", 'width': 4},
                    'thickness': 0.75,
                    'value': score}},
        ))

        fig.update_layout(
            paper_bgcolor="#000000",  
            font={'color': "#FFFFFF", 'family': "Arial"}, 
            height=325, 
        )

        st.plotly_chart(fig)
        
        # Distribution d'une Feature selon les classes
        st.write("### Analyse Univariée")
        selected_feature = st.selectbox("Feature :", df.columns)
        subset_data = df[selected_feature]
        client_feature = df[selected_feature][example_idx]
        
        fig = px.histogram(subset_data, nbins=100, title="Histogramme de la Feature", labels={"value": "Data Value"})
        fig.add_annotation(
            x=client_feature,
            y=10,  # Adjust the vertical position of the annotation
            text=f"Client: {client_feature}",
            showarrow=True,
            arrowhead=4,
            arrowcolor="#FFFFFF",
            bordercolor="#FFFFFF",
            ax=-20,  # Adjust the arrowhead position
            ay=30,  # Adjust the arrowhead position
        )
        st.plotly_chart(fig)
        
        # Analyse Bivariée entre 2 Features
        st.write("### Analyse Bivariée")
        
        selected_feature_1 = st.selectbox("Feature 1 :", df.columns)
        selected_feature_2 = st.selectbox("Feature 2 :", df.columns)
        
        # Créer une palette de couleurs adaptée aux daltoniens
        colorscale = px.colors.diverging.Portland
        # Créer une figure avec une palette de couleurs adaptée
        fig = px.scatter(df, x=selected_feature_1, y=selected_feature_2, title="Analyse Bivariée")

        # Mettre à jour les traces avec un contraste plus élevé
        fig.update_traces(marker=dict(size=5, opacity=0.6, line=dict(width=0.1, color='white')), selector=dict(mode='markers'))

        
        client_feature_1 = df[selected_feature_1][example_idx]
        client_feature_2 = df[selected_feature_2][example_idx]
        
        fig.add_annotation(
            x=client_feature_1,
            y=client_feature_2,
            text=f"Client: ({client_feature_1},{client_feature_2})",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FFFFFF",
            bordercolor="#FFFFFF",
            borderwidth=2,
            ax=-20,
            ay=30
        )
        
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()