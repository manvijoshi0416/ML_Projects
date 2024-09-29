import streamlit as st
import pickle
import numpy as np
import pandas as pd
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
st.title("Laptop Predictor")
company = st.selectbox('Brand',df['Company'].unique())
type_name = st.selectbox('Type',df['TypeName'].unique())
ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight of the laptop')
cpu = st.selectbox('Brand',df['CPU_Brand'].unique())
touchscreen = st.selectbox('Touchscreen',['No','Yes'])
ips = st.selectbox('IPS',['No','Yes'])
screen_size = st.number_input("Screen Size")
resolution = st.selectbox("Screen Resolution",['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
gpu = st.selectbox('GPU',df['GPU_Brand'].unique())
os = st.selectbox('OS',df['OS'].unique())
if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2)+(y_res**2))**0.5/screen_size
    # Assuming 'query' is a list or array
    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'Weight': [weight],
        'CPU_Brand': [cpu],
        'Touchscreen': [touchscreen],
        'IPS_Pannel': [ips],
        'PPI': [ppi],
        'SSD': [ssd],
        'HDD': [hdd],
        'GPU_Brand': [gpu],
        'OS': [os]
    })
    query['Ram'] = query['Ram'].astype(np.int32)  # Cast Ram to int32
    query['Weight'] = query['Weight'].astype(np.float32)  # Cast Weight to float32
    query['SSD'] = query['SSD'].astype(np.int32)  # Cast SSD to int32
    query['HDD'] = query['HDD'].astype(np.int32)
    # Reshape the query for prediction
    #query = query.reshape(1,12)
    st.write(query.dtypes)
    try:
        # Make prediction
        predicted_price = pipe.predict(query)
        st.title(f"The predicted price of this configuration is ₹{predicted_price:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
    #st.title("The predicted price of this configuration is " + np.exp(pipe.predict(query)[0]))

