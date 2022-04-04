###############################
# This program lets you       #
# - enter values in Streamlit #
# - get prediction            #  
###############################
import joblib 
import pandas as pd
import streamlit as st
import numpy as np

# Load the model from the file
temp_joblib = joblib.load('mTemporal.pkl')
#############  
# Main page #
#############                
st.title("Predicting the probability of the need to deliver within 7 days")
st.write("""
Created by Paula Dominguez del Olmo
\nThis is a Streamlit web app created so users could explore the Support Vector Machine (SVM) model, which acts as a predictive model of ePE need for delivery within 7 days of diagnosis,
considering this as the window of effect of antenatal corticosteroids
\nThe used data has been provided by the Hospital Universitario 12 de Octubre after conducting a single-centre, observational, retrospective cohort study
with 211 patients in singleton pregnancies diagnosed with early PE in which expectant management was attempted between January 2014 and December 2020.     
""")  
# AJUSTAMOS LOS VALORES MÍNIMOS Y MÁXIMOS

min_sflt1=1000
max_sflt1=85000

min_plgf=0.1
max_plgf=300.0

min_numFar=0
max_numFar=4

min_pvaginal=0
max_pvaginal=8

min_PFEecoincl=200
max_PFEecoincl=4000

min_IPAUti=0.05
max_IPAUti=6.00

min_IPAUtd=0.05
max_IPAUtd=6.00

min_ipmAUTeco=0.400
max_ipmAUTeco=4.000

min_EGPE=20.00
max_EGPE=36.00


# ENTER NUMERICAL DATA FOR PREDICTION
st.header('**Preeclampsia Episode Variables:**')
EG_PE_user = st.number_input('Gestational age at PE onset (calculated in weeks):',
                                 min_value = min_EGPE,
                                 max_value = max_EGPE
                                ) 
sFlt1_user = st.number_input('Anti-angiogenic marker (sFlt-1) absolute value:',
                                 min_value = min_sflt1,
                                 max_value = max_sflt1
                                )                               
PlGF_user = st.number_input('Angiogenic marker (PlGF) absolute value:',
                                 min_value = min_plgf,
                                 max_value = max_plgf
                                )                                                                 
PFE_ecoincl_user = st.number_input('Estimated fetal weight on the inclusion ultrasound:',
                                 min_value = min_PFEecoincl,
                                 max_value = max_PFEecoincl
                                )
IPAUti_eco_incl = st.number_input('Pulsatility index of the left uterine artery:',
                                 min_value = min_IPAUti,
                                 max_value = max_IPAUti
                                )
IPAUtd_eco_incl = st.number_input('Pulsatility index of the right uterine artery:',
                                 min_value = min_IPAUtd,
                                 max_value = max_IPAUtd
                                )                               
NumFar_user = st.slider('Number of anti-hypertensive drugs:',
                                 min_value = min_numFar,
                                 max_value = max_numFar
                                ) 
CirStadio = st.selectbox('Fetal Growth Restriction (FGR) stage at PE diagnosis:',
                              ('No',
                               'Stage I',
                               'Stage II',
                               'Stage III', 
                               'Stage IV')
                             )
               
if CirStadio=='No':
        CIRestadio_3 =0
        CIRestadio_4 =0
elif CirStadio=='Stage I':
        CIRestadio_3 =0
        CIRestadio_4 =0
elif CirStadio=='Stage II':
        CIRestadio_3 =0
        CIRestadio_4 =0
elif CirStadio=='Stage III':
        CIRestadio_3 =1
        CIRestadio_4 =0
elif CirStadio=='Stage IV':
        CIRestadio_3 =0
        CIRestadio_4 =1
st.write("""
    The different FGR stages are defined as:
    \nStage I: antegrade umbilical artery flow
    \nStage II: absent end-diastolic umbilical artery flow
    \nStage III: reversed umbilical artery flow or ductus venosus pulsatility index >95th centile
    \nStage IV: reversed a wave in ductus venosus, unprovoked decelerations
    """)
                                                                                                                            
# CALCULAMOS EL RATIO A PARTIR DE LOS VALORES DE SFLT-1 Y PIGF
ratio_user=(float(sFlt1_user)/float(PlGF_user))
IPmAUt_ecoincl_user=(float(IPAUtd_eco_incl)+float(IPAUti_eco_incl))/2
# NORMALIZANDO LOS VALORES INTRODUCIDOS POR EL USUARIO PARA QUE SE CORRESPONDAN CON LOS VALORES ENTRENADOS (VARIABLES PE)
ratio =(float(ratio_user))/(1975.73)
NumFar=(int(NumFar_user))/(4)
EG_PE=(float(EG_PE_user)-20.86)/(34.14-20.86)
IPmAUt_eco_incl=(float(IPmAUt_ecoincl_user)-0.465)/(3.960-0.465)
PFE_eco_incl=(float(PFE_ecoincl_user)-192)/(3459-192)

# ENTER CATEGORICAL DATA FOR PREDICTION (aparecerán a la izquierda de la aplicación)
st.sidebar.header('**Pregestational or first trimester variables**:')
Etnia = st.sidebar.selectbox('Maternal ethnicity:',
                              ('Caucasian',
                               'Asian',
                               'Moroccan/Arab',
                               'Hispanic', 
                               'Other')
                             )
               
if Etnia=='Caucasian':
        Etnia_5= 0
elif Etnia=='Asian':
        Etnia_5= 0
elif Etnia=='Moroccan/Arab':
        Etnia_5= 0
elif Etnia=='Hispanic':
        Etnia_5= 0
elif Etnia=='Other':
        Etnia_5= 1

p_vaginal_user = st.sidebar.slider('Number of previous vaginal births:',
                                 min_value = min_pvaginal,
                                 max_value = max_pvaginal
                                ) 
imc35 = st.sidebar.selectbox('Body Mass Index > 35 kg/m²:',
                              ('No',
                               'Yes')
                             )               
if imc35=='No':
        IMCmayor35 =0
elif imc35=='Yes':
        IMCmayor35 =1

af_PE = st.sidebar.selectbox('PE family history:',
                              ('No',
                               'Yes, her mother',
                               'Yes, her sister',
                               'Yes, both the mother and the sister')
                             )
               
if af_PE=='No':
        af_pe_2 =0
elif af_PE=='Yes, her mother':
        af_pe_2 =0
elif af_PE=='Yes, her sister':
        af_pe_2 =1
elif af_PE=='Yes, both the mother and the sister':
        af_pe_2 =0


AAS = st.sidebar.selectbox('Aspirin intake:',
                              ('No',
                               'Yes, before 16 weeks',
                               'Yes, after 16 weeks')
                             )               
if AAS=='No':
        AAS_0 =1
        AAS_2 =0
elif AAS=='Yes, before 16 weeks':
        AAS_0 =0
        AAS_2 =0
elif AAS=='Yes, after 16 weeks':
        AAS_0 =0
        AAS_2 =1

# NORMALIZANDO LOS VALORES INTRODUCIDOS POR EL USUARIO PARA QUE SE CORRESPONDAN CON LOS VALORES ENTRENADOS (VARIABLES BASALES)
p_vaginal=(int(p_vaginal_user))/(8)                  

# when 'Predict' is clicked, make the prediction and store it 
if st.button('Get Your Prediction'): 
    
    X = pd.DataFrame({'ratio':[ratio],
                      'NumFar':[NumFar], 
                      'p_vaginal':[p_vaginal],
                      'CIRestadio_3':[CIRestadio_3],
                      'af_pe_2':[af_pe_2], 
                      'IMCmayor35':[IMCmayor35],
                      'Etnia_5':[Etnia_5],
                      'CIRestadio_4':[CIRestadio_4], 
                      'AAS_0':[AAS_0],
                      'IPmAUt_eco_incl':[IPmAUt_eco_incl], 
                      'PFE_eco_incl':[PFE_eco_incl],
                      'AAS_2':[AAS_2],
                      'EG_PE':[EG_PE]
                     })
               
    # Making predictions            
    prediction = temp_joblib.predict(X)
    prediction_proba = temp_joblib.predict_proba(X)
    prediction_proba1 = temp_joblib.predict_proba(X)[:,1] #probabilidad de 1
    prediction_proba0 = temp_joblib.predict_proba(X)[:,0] #probabilidad de 0 

    
    st.subheader('**Prediction Output**')
    st.write('Need to deliver within 7 days from the PE diagnosis date with a probability of:')
    st.write(prediction_proba0)


   