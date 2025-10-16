# import libraries
import streamlit as st 
import pandas as pd 
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestRegressor

# load the trained model
model_path = 'random_forest_model.pkl'
with open(model_path, 'rb') as file:
    model_rf = pickle.load(file)
    
# function to predect based on user input
def predict_btc_price(input_data):
    # Make predection using the model
    prediction = model_rf.predict(input_data)
    return prediction[0] # Assuming model returns a single predection

def main():
    # Title of web app
    st.title('Predict BTC Close price')
    
    # sidebar for user input
    st.sidebar.title('Input features')
    
    # Input for USDT ,BNB closing price and volumes
    usdt_close = st.sidebar.number_input('USDT Close Price', min_value=0.0,format='%.2f')
    usdt_volume = st.sidebar.number_input('USDT Volume',min_value= 0.0,format='%.2f')
    bnb_close = st.sidebar.number_input('BNB Close Price',min_value=0.0,format='%.2f')
    bnb_volume = st.sidebar.number_input('BNB Volume',min_value=0.0,format='%.2f')
    # create input dataframe (ensure column names/order match what the model expects)
    input_data = pd.DataFrame({
        'USDT_Close': [usdt_close],
        'USDT_Volume': [usdt_volume],
        'BNB_Close': [bnb_close],
        'BNB_Volume': [bnb_volume]
    })

    # button to trigger prediction
    if st.button('Predict BTC Close Price'):
        predicted_price = predict_btc_price(input_data)
        st.write(f'Predicted BTC Close Price : {predicted_price}')


if __name__ == '__main__':
    main()
    
