import numpy as np
import pickle
import streamlit as st


# laoding the save model using pickle.load
load_model = pickle.load(open(r"C:\Users\rushi\Downloads\diabetes prediction\Diabetis_prediction_system\Diabetis prediction system\trained_model.sav",'rb'))

# creating function for prediction
def diab_prediction(input_data):

    diab_data = np.asarray(input_data).reshape(1,-1)

    prediction = load_model.predict(diab_data)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
         return 'The person is diabetic'

def main():
    
    #giving a title
    st.title('Diabetes Prediction System')

    # creating input data fields 
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('SkinThickness')
    Insulin = st.text_input('Insulin level')
    bmi = st.text_input('Body-Mass-index')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
    Age = st.text_input('Age of person')

    #pixel = st.camera_input('Input')

    # code for prediction
    diagnosis = ""

    # creating the button 
    if st.button('Test Diabetes'):
        diagnosis = diab_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, bmi, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()







