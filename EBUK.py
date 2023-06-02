
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder


import pickle
import streamlit as st


# loading the saved models
model = pickle.load(open('big_mart_model.pkl', 'rb'))




# page title
st.title('Sales Prediction using ML')

#Image
st.image('Sale-revenue.webp')

# getting the input data from the user
col1, col2 = st.columns(2)

with col1:
    Item_Visibility = st.number_input('Item Visibility', min_value=0.00, max_value=0.40, step=0.01)

with col1:
    Item_MRP = st.number_input('Item MRP', min_value=30.00, max_value=270.00, step=1.00)

with col1:
    Outlet_Size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])

with col2:
    Item_Fat_Content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])

with col2:
    Outlet_Location_Type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])

#Data Preprocessing

data = {
        'Item_Visibility': Item_Visibility,
        'Item_MRP' : Item_MRP,
        'Outlet_Size' : Outlet_Size,
        'Item_Fat_Content_Regular': Item_Fat_Content,
        'Outlet_Location_Type' : Outlet_Location_Type
            }

oe = OrdinalEncoder(categories = [['Small','Medium','High']])
scaler = StandardScaler()

def make_prediction(data):
    df = pd.DataFrame(data, index=[0])

    if df['Item_Fat_Content_Regular'].values == 'Low Fat':
        df['Item_Fat_Content_Regular'] = 0.0

    if df['Item_Fat_Content_Regular'].values == 'Regular':
        df['Item_Fat_Content_Regular'] = 1.0

    if df['Outlet_Location_Type'].values == 'Tier 1':
        df[['Outlet_Location_Type_Tier 1','Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3']] = [1.0, 0.0, 0.0]

    if df['Outlet_Location_Type'].values == 'Tier 2':
        df[['Outlet_Location_Type_Tier 1','Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3']] = [0.0, 1.0, 0.0]

    if df['Outlet_Location_Type'].values == 'Tier 3':
        df[['Outlet_Location_Type_Tier 1','Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3']] = [0.0, 0.0, 1.0]

    df['Outlet_Size'] = oe.fit_transform(df[['Outlet_Size']])
    df = df.drop(columns = ['Outlet_Location_Type'], axis = 1 )
    df[['Item_Visibility', 'Item_MRP']] = StandardScaler().fit_transform(df[['Item_Visibility', 'Item_MRP']])

    prediction = model.predict(df)

    return round(float(prediction),2)



# code for Prediction
# sales_prediction_output = ""

# creating a button for Prediction
sales_prediction_output = ""
if st.button('Predict Sales'):
    sales_prediction = make_prediction(data)
    sales_prediction_output = f"The sales is predicted to be {sales_prediction}"


st.success(sales_prediction_output)





