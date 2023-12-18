from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd


def predict_client(model, df):    
    predictions_data = predict_model(model, df)   
    #print('aqui teste',predictions_data) 
    return predictions_data['prediction_label'][0]
    
model = load_model(r'C:\Users\emmanueltorres\Downloads\Scripts\streamlit\risco_credito\credit_scoring\Final_Model_xgboost')


st.title('App Web de pontuação de crédito')
st.write("Crie um aplicativo Web para classificar clientes de um banco com bom ou mau risco.")


Maturite_mois = st.sidebar.slider(label = 'Meses de vencimento', min_value = 16,
                          max_value = 24 ,
                          value = 23,
                          step = 1)

Montant_Pret_BAM = st.sidebar.slider(label = 'Valor do empréstimo BAM', min_value = 5000,
                          max_value = 15000 ,
                          value = 8200,
                          step = 25)

Sexe = st.sidebar.selectbox('Sexo',['Male','Female'])

Etat_Civil = st.sidebar.selectbox('Estado Civil',['Divorced','Married','Single','Widowed'])

Niveau_Formation = st.sidebar.selectbox(
    'Nivel de formacao',
    ['Completed University','Some College Courses','Completed Vocational Training','High School Diploma','Secondary School to Grade 10'])

Age_ans = st.sidebar.slider(label = 'Anos de idade', min_value = 18,
                          max_value = 65 ,
                          value = 32,
                          step = 1)                          

Ans_a_ladresse = st.sidebar.slider(label = "Anos no endereco", min_value = 0.5,
                          max_value = 30.0 ,
                          value = 1.0,
                          step = 0.5)

Locataire_Proprietaire = st.sidebar.selectbox("Proprietario do inquilino",['RENT','Own'])

Nbre_de_Dependants = st.sidebar.slider(label = 'Numero de dependentes', min_value = 0,
                          max_value = 5,
                          value = 2,
                          step = 1)

Ans_en_Activite = st.sidebar.slider(label = 'Anos em atividade', min_value = 0.5,
                          max_value = 15.0 ,
                          value = 1.0,
                          step = 0.5)

Emplacement_du_business = st.sidebar.selectbox("Localizacao da empresa",['Region1','Region2','Region3','Region4','Region5'])

Credit_Bureau_negative = st.sidebar.slider(label = 'Credito de Escritorio Negativo=1', min_value = 0,
                          max_value = 1 ,
                          value = 0,
                          step = 1)

NbreEmployesFamille = st.sidebar.slider(label = 'Numero Funcionarios Familia', min_value = 0,
                          max_value = 7 ,
                          value = 1,
                          step = 1)

Type_dActivite = st.sidebar.selectbox(
    "Tipo de atividade",
    ['Craftsperson','Personal Services','Car Repair','Child Care','Convenience Store','Small Grocers','General Contractor'])
                         
Ventes_Mensuelles_BAM = st.sidebar.slider(label = 'Vendas Mensais BAM', min_value = 400,
                          max_value = 20000,
                          value = 2550,
                          step = 25)

features = {'Meses de vencimento': Maturite_mois,
            'Valor do empréstimo BAM': Montant_Pret_BAM,
            'Sexo': Sexe,
            'Estado Civil': Etat_Civil,
            'Nivel de formacao':Niveau_Formation,
            'Anos de idade':Age_ans,
            "Anos no endereco":Ans_a_ladresse,
            "Proprietario do inquilino": Locataire_Proprietaire,
            'Numero de dependentes': Nbre_de_Dependants,
            'Anos em atividade': Ans_en_Activite,
            "Localizacao da empresa": Emplacement_du_business,
            'Credito de Escritorio Negativo=1': Credit_Bureau_negative, 
            'Numero Funcionarios Familia': NbreEmployesFamille,
            "Tipo de atividade":Type_dActivite,
            'Vendas Mensais BAM':Ventes_Mensuelles_BAM
            }
 

features_df  = pd.DataFrame([features])

map_dict = {0:'Bom Cliente',
            1:'Mal Cliente'}
st.table(features_df)  

if st.button('Previsão'):
    
    prediction = predict_client(model, features_df)
    
    st.write("A previsão é de: {}".format(map_dict [prediction]))
    
