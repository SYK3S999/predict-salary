import streamlit as st
import pickle
import numpy as np


def load_model():
    with open ('saved_steps.pkl','rb') as file :
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title('Software developer salary prediction')
    st.write("""### We need some informations to predict the salary """)
    

    countries = (
        "United States of America",                            
        "Germany",                                                  
        "United Kingdom of Great Britain and Northern Ireland",     
        "Canada",                                                   
        "India",                                                    
        "France",                                                   
        "Netherlands",                                              
        "Australia",                                                 
        "Brazil",                                                    
        "Spain",                                                    
        "Sweden",                                                    
        "Italy",                                                     
        "Poland",                                                    
        "Switzerland",                                               
        "Denmark",                                                   
        "Norway"                                                  
    )
    educations =(
        "Bachelor’s degree",
        "less than a Bachelors",
        "Master’s degree",
        "post grad"
    )

    country = st.selectbox("Country",countries)
    education = st.selectbox("education Level",educations)
    experience = st.slider("years of experience", 0, 50 , 3)

    ok = st.button("Calculate Salary")
    if ok:
        x = np.array([[country, education,experience]])
        x[:,0] = le_country.transform(x[:,0])
        x[:,1] = le_education.transform(x[:,1])
        x = x.astype(float)

        salary = regressor.predict(x)
    
        st.subheader(f"the predicted salary is ${salary[0]:.2f}")