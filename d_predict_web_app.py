import pickle
import numpy as np
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('d_prediction/diabetes_model.sav', 'rb'))

# creating a function for prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'The person is not diabetic.'
    else:
        return 'The person is diabetic.'

def main():
    # title
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from the user 
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('Body Mass Index Value')
    DiabetesPedigreeFunctionAge = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the person')
    
    # code for prediction
    diagnosis = ''
    
    # creating a button 
    if st.button('Diabetes Test Result'):
        # convert user inputs to float
        input_data = [float(Pregnancies), float(Glucose), float(BloodPressure),
                      float(SkinThickness), float(Insulin), float(BMI),
                      float(DiabetesPedigreeFunctionAge), float(Age)]
        
        # pass the user input as a list to the prediction function
        diagnosis = diabetes_prediction(input_data)
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
