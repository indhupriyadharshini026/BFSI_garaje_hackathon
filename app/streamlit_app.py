import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the model and scaler
model = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
features = pickle.load(open('model/features.pkl', 'rb'))

# Streamlit app interface
st.title('Credit Risk Prediction')

# Define the input fields
age = st.number_input('Age', min_value=18, max_value=100)
sex = st.selectbox('Sex', ['male', 'female'])
job = st.selectbox('Job', ['0', '1', '2', '3'])
housing = st.selectbox('Housing', ['own', 'rent', 'free'])
saving_accounts = st.selectbox('Saving Accounts', ['little', 'moderate', 'quite rich', 'rich'])
checking_account = st.selectbox('Checking Account', ['little', 'moderate', 'quite rich', 'rich'])
credit_amount = st.number_input('Credit Amount (in DM)', min_value=0)
duration = st.number_input('Duration (in months)', min_value=1)
purpose = st.selectbox('Purpose', ['car', 'furniture/equipment', 'radio/TV', 'domestic appliances', 'repairs',
                                   'education', 'business', 'vacation/others'])

# Prepare the input data for prediction
input_data = np.array([[age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose]])

# Perform the label encoding directly in the Streamlit app
label_encoders = {
    'Sex': LabelEncoder(),
    'Housing': LabelEncoder(),
    'Saving accounts': LabelEncoder(),
    'Checking account': LabelEncoder(),
    'Purpose': LabelEncoder()
}

# Fit the label encoders and encode the data
input_data[:, 1] = label_encoders['Sex'].fit_transform(input_data[:, 1].astype(str))
input_data[:, 3] = label_encoders['Housing'].fit_transform(input_data[:, 3].astype(str))
input_data[:, 4] = label_encoders['Saving accounts'].fit_transform(input_data[:, 4].astype(str))
input_data[:, 5] = label_encoders['Checking account'].fit_transform(input_data[:, 5].astype(str))
input_data[:, 8] = label_encoders['Purpose'].fit_transform(input_data[:, 8].astype(str))

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction button
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    prediction_prob = model.predict_proba(input_data_scaled)[0][1]

    # Display the results
    if prediction[0] == 0:
        st.write('Prediction: Bad Risk')
        st.write(f'Probability of Bad Risk: {prediction_prob:.2f}')
    else:
        st.write('Prediction: Good Risk')
        st.write(f'Probability of Good Risk: {prediction_prob:.2f}')