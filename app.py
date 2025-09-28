import streamlit as st 
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r"C:\Users\HP\OneDrive\Documents\SPYDER WORK\linear_regression_model.pkl",'rb'))

# set the title of the streamlit app
st.title('Salary predection app')
# Add a brief description
st.write('This App Predict the salary based on years of exprience using the simple linear regression model')

# Add input widget for user to enter years of exprience
years_exprience = st.number_input('Enter Years of Expriences:',min_value=0.0,max_value=50.0,value=1.0,step=0.5) 

# Where the butten is clicked , make predection 
if st.button('Predict Salary'):
    # make a predection using the trained model 
    exprience_input = np.array([[years_exprience]]) # converted the input to a 2D array for predection
    predection = model.predict(exprience_input)
    # Display the result
    st.success(f'The predected salary for {years_exprience} years of exprience is :${predection[0]:,.2f}')
    
# Display the information about the model
st.write('The model was trained using a dataset of salaries and years of expriences')

