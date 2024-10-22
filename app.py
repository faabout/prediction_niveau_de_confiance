import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Fonction pour charger les données (avec mise en cache)
@st.cache_data
def load_data():
    df = pd.read_excel('data_augmented.xlsx')
    # Transformation en minuscules pour les variables catégorielles
    df = df.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
    
    return df


# Chargement des données, encodeurs/scaler et modèle
df = load_data()
# Choix des colonnes à étudier
df = df[['Région', 'Commune', 'Sexe', 'Age', 'Niveau d\'étude', 'Situation professionnelle', 'Perception du coup d\'Etat', 'Décisions','Priorité de la transition',
         'Ecriture de la Constitution', 'Nombre et durée de mandat présidentiel' , 'Niveau de Confiance']].copy()
# Renommer les colonnes à étudier
df.rename(columns={
   'Région' : 'region',
   'Commune' : 'commune',
   'Sexe': 'sexe',
   'Age': 'age',
   'Niveau d\'étude' : 'niveau_etude',
   'Situation professionnelle': 'situation_professionnelle',
   'Perception du coup d\'Etat': 'perception',
   'Décisions': 'decisions',
   'Priorité de la transition': 'priorite_transition',
   'Ecriture de la Constitution': 'ecriture_constitution',
   'Nombre et durée de mandat présidentiel':'nombre_duree_mandat',
   'Niveau de Confiance' : 'niveau_de_confiance'
} , inplace = True)
#encoders_scaler = load_encoders_scaler()
#model = load_model()

# Convertir toutes les valeurs en minuscules avant l'encodage
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
for col in categorical_columns:
    df[col] = df[col]
#Créer une copie du DataFrame
df_encoded = df.copy()
df_encoded.head()

# Extraire les encodeurs et scaler du fichier encoders_scaler.pkl
encoder0 = LabelEncoder()
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
encoder3 = LabelEncoder()
encoder4 = LabelEncoder()
encoder5 = LabelEncoder()
encoder6 = LabelEncoder()
encoder7 = LabelEncoder()
encoder8 = LabelEncoder()
encoder9 = LabelEncoder()
encoder10 = LabelEncoder()
encoders = [encoder0, encoder1, encoder2,encoder3, encoder4, encoder5, encoder6, encoder7, encoder8, encoder9, encoder10]
for i in range(len(categorical_columns)):
  encoders[i].fit(df_encoded[categorical_columns[i]]) # le modèle prend le temps de reconnaitre les classes
  df_encoded[categorical_columns[i]] = encoders[i].transform(df_encoded[categorical_columns[i]]) # le modèle encode les variables
#scaler = encoders_scaler['scaler']

# Séparaer les variables d'entrée et la variable de sortie
X = df_encoded.drop('niveau_de_confiance', axis= 1)
# La vairable de sortie
y = df_encoded['niveau_de_confiance'].values

# importer StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
# Instancier StandardScaler
scaler = StandardScaler()
scaler.fit(X) # calcul de la moyenne et de l'écart type
X = scaler.transform(X) # normalisation
# Spliter les données
from sklearn.model_selection import train_test_split

X_train,  X_test,  y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=808)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 808)

from sklearn.ensemble import RandomForestClassifier

# Entrainer le modèle en utilisant les paramètres optimaux
Rfc= RandomForestClassifier(criterion='entropy', max_depth=14)
# entrainer le modèle
Rfc.fit(X_train, y_train)
# Interface Streamlit pour les saisies utilisateur
st.title('Prédiction du Niveau de Confance')

region = st.selectbox("Sélectionnez la région", options=df['region'].unique())
commune = st.selectbox("Sélectionnez la commune", options=df['commune'].unique())
sexe = st.radio("Sélectionnez le genre", options=df['sexe'].unique())
age = st.number_input("Entrez l'âge", min_value=0, max_value=100, value=25)
niveau_etude = st.selectbox("Sélectionnez le niveau d'étude", options=df['niveau_etude'].unique())
situation_professionnelle = st.selectbox("Sélectionnez la situation professionnelle", options=df['situation_professionnelle'].unique())
perception = st.selectbox("Sélectionnez la perception du coup d'Etat", options=df['perception'].unique())
decisions = st.selectbox("Sélectionnez les décisions", options=df['decisions'].unique())
priorite_transition = st.selectbox("Sélectionnez la priorité de la transition", options=df['priorite_transition'].unique())
ecriture_constitution = st.selectbox("Sélectionnez la durée d'écriture de la constitution", options=df['ecriture_constitution'].unique())
nombre_duree_mandat = st.selectbox("Sélectionnez le nombre et la durée du mandat présidentiel", options=df['nombre_duree_mandat'].unique())

# Encoder les variables catégorielles
region = encoder0.transform([region])[0]
commune = encoder1.transform([commune])[0]
sexe= encoder2.transform([sexe])[0]
niveau_etude = encoder3.transform([niveau_etude])[0]
situation_professionnelle = encoder4.transform([situation_professionnelle])[0]
perception = encoder5.transform([perception])[0]
decisions = encoder6.transform([decisions])[0]
priorite_transition = encoder7.transform([priorite_transition])[0]
ecriture_constitution = encoder8.transform([ecriture_constitution])[0]
nombre_duree_mandat = encoder9.transform([nombre_duree_mandat])[0]

# Créer un tableau avec les données encodées et l'âge
input_data = np.array([
    [region, commune, sexe, age, niveau_etude, situation_professionnelle, perception, decisions, priorite_transition, ecriture_constitution, nombre_duree_mandat]
])

# Appliquer le StandardScaler pour normaliser les données
input_data_scaled = scaler.transform(input_data)

# Bouton pour lancer la prédiction
if st.button('Prédire'):
    prediction = Rfc.predict(input_data_scaled)
    prediction_decoded = encoder10.inverse_transform([prediction[0]])

    # Retourner la prédiction
    st.write(f"Niveau de confiance : {prediction_decoded[0]}")
