import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from main import HDB

from sklearn import pipeline
from sklearn import compose
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


st.write("""
# Singapore HDB Price Prediction App
""")
#st.write('---')


hdb = HDB()
df = hdb.df

#st.write("Data (first 100 records)")
#st.write(df[:100])

st.sidebar.header('Specify Input Parameters')
selected_locations = st.sidebar.selectbox('Location', list(df['town'].unique()), index=None, placeholder="Select location",)

selected_room_size = st.sidebar.selectbox('Room Size', list(df['flat_type'].unique()), index=None, placeholder="Select room size",)

selected_storey = st.sidebar.selectbox('Storey Level', sorted(list(df['storey_range'].unique())), index=None, placeholder="Select storey",)
selected_model = st.sidebar.selectbox('Flat Model', sorted(list(df['flat_model'].unique())), index=None, placeholder="Select model",)

selected_lease = st.sidebar.slider('Remaining Lease', df['remaining_lease'].min(), df['remaining_lease'].max(),(df['remaining_lease'].min(), df['remaining_lease'].max()))
st.sidebar.write("Enter below to predict")
selected_year = st.sidebar.selectbox('Year', list(range(2025,2050)), index=None, placeholder="Select Year",)
selected_month = st.sidebar.selectbox('Month', list((range(1,12))), index=None, placeholder="Select Month to predict",)
selected_sqm = st.sidebar.slider('Floor area m^2', df['floor_area_sqm'].min(), df['floor_area_sqm'].max(),)


def get_selected_df(df, selected_locations, selected_room_size, selected_storey,selected_lease, selected_model):
    df_selected = df.copy()
    if selected_locations!=None:
        #print(selected_locations)
        df_selected = df[df.town == selected_locations]
        #print(df_selected)
    if selected_room_size!=None:
        df_selected = df_selected[df_selected.flat_type == selected_room_size]
    if selected_storey != None:
        df_selected = df_selected[df_selected.storey_range == selected_storey]
    if selected_model!=None:
        df_selected = df_selected[df_selected.flat_model == selected_model]
    df_selected = df_selected[(selected_lease[0] <= df_selected.remaining_lease) & (df_selected.remaining_lease <= selected_lease[1])]
    return df_selected

def display_chart(df_selected):
    plt.figure(figsize=(10, 5))
    plot = sns.lineplot(x='month', y='resale_price', data=df_selected, hue="town")
    try:
        sns.move_legend(
            plot, "lower center",
            bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
        )
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15))
        plt.xticks(rotation=90)
        plt.xlabel('Time', fontweight='bold')
        plt.ylabel('Resale price',fontweight='bold')
        plt.yscale("linear")
        st.pyplot(plot.get_figure())
    except:    
        st.write("No data to display")

def predict_input(selected_locations, selected_room_size, selected_storey,selected_lease, selected_year, selected_month, selected_sqm, selected_model):
    input_value_df = pd.DataFrame({
    'town': [f'{selected_locations}'],
    'flat_type': [f'{selected_room_size}'],
    'storey_range': [f'{selected_storey}'],
    'flat_model':[f'{selected_model}'],
    'floor_area_sqm': [selected_sqm],
    'remaining_lease': [selected_lease[0]],
    'time_year': [selected_year],
    'time_month': [selected_month]
    })
    return input_value_df   

def to_pred(df, input_df):
    df = df[['floor_area_sqm','remaining_lease','time_year','time_month','town','flat_type','storey_range','flat_model','resale_price']]
    numeric_features = ['floor_area_sqm','remaining_lease','time_year','time_month']
    categorical_features = ['town','flat_type','storey_range', 'flat_model']

    num_pipeline = pipeline.Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', preprocessing.StandardScaler())
    ])
    ohe_pipeline = pipeline.Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    col_trans = compose.ColumnTransformer(
        transformers = [
            ('num',num_pipeline, numeric_features),
            ('ohe',ohe_pipeline, categorical_features)
        ], remainder = 'passthrough', n_jobs=-1
    )
    pipeline1 = pipeline.Pipeline(steps=[
    ('preprocessing', col_trans)
    ])  
    X = df.drop('resale_price', axis=1)
    y = df['resale_price']
    X_processed = pipeline1.fit_transform(X)
    input_df_processed = pipeline1.transform(input_df)

    lr = LinearRegression()
    lr.fit(X_processed, y)
    y_pred_lr = lr.predict(input_df_processed)

    rfr = RandomForestRegressor(random_state=42, n_jobs=-1)
    rfr.fit(X_processed, y)
    y_pred = rfr.predict(input_df_processed)

    knn = KNeighborsRegressor(n_neighbors=9)
    knn.fit(X_processed,y)
    y_pred_knn = knn.predict(input_df_processed)

    return y_pred_lr, y_pred, y_pred_knn

df_selected = get_selected_df(df, selected_locations, selected_room_size, selected_storey,selected_lease, selected_model)
#st.header('Display result')
try:
    display_chart(df_selected)
except:
    st.write("No data returned")
st.dataframe(df_selected[['town','flat_type','storey_range','flat_model','floor_area_sqm','remaining_lease','resale_price','month']])


prediction = st.sidebar.button("Make a prediction")


if prediction:
    if selected_year == None or selected_month == None or selected_sqm == None or selected_locations == None or selected_room_size == None or selected_storey == None or selected_lease==None:
        st.toast('Please enter all the fields', icon="⚠️")
    elif selected_lease[0] != selected_lease[1]:
        st.toast('Please make remaining lease single value', icon="⚠️")
    else:
        input_df = predict_input(selected_locations, selected_room_size, selected_storey,selected_lease, selected_year, selected_month, selected_sqm, selected_model)
        st.write("Inputs for prediction:")
        st.dataframe(input_df)
        st.write("Predicted price is")
        y_pred_lr, y_pred, y_pred_knn = to_pred(df,input_df)
        price_lr, price_rf, price_knn = int(y_pred_lr), int(y_pred), int(y_pred_knn)
        st.write(f"Linear regression: SGD {price_lr}")
        st.write(f"Random Forest: SGD {price_rf}")
        st.write(f"KNN: SGD {price_knn}")


