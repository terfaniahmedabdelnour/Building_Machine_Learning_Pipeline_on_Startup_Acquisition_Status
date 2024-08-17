
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

def load_data():
    df = pd.read_csv("/home/ahmed/Desktop/Technocolabs_Internship/Week8(ML Pipline deployment)/new_preprocessed.csv")
    for col in ['status', 'country_code', 'category_code']:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
    return df


def train_model(df):
    X = df.drop(['status'], axis=1)
    y = df['status'] 
    encoder = LabelEncoder()
    y_encoded = encoder.fit(df['status'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_binary = np.where(y_train == 3, 1, 0)
    binary_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  
    #('pca', PCA(n_components=10)),
    ('binary_classifier', LogisticRegression(penalty='l1', solver='saga', max_iter=50, multi_class='multinomial', C=1))])
    binary_pipeline.fit(X_train, y_train_binary)
    binary_predictions = binary_pipeline.predict(X_test)
    X_test_not_class_3 = X_test[binary_predictions == 0]
    y_test_not_class_3 = y_test[binary_predictions == 0]
    X_train_not_class_3 = X_train[y_train != 3]
    y_train_not_class_3 = y_train[y_train != 3]
    multiclass_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    #('pca', PCA(n_components=10)),
    ('multiclass_classifier', RandomForestClassifier(random_state=42))
    ])
    multiclass_pipeline.fit(X_train_not_class_3, y_train_not_class_3)
    return multiclass_pipeline

def main():
    st.title('Streamlit Startup_Acquisition_Status App')
    
    # Load data
    data_load_state = st.text('Loading data...')
    df = load_data()
    data_load_state.text('Data loaded successfully!')
    
    # Train model
    model_load_state = st.text('Training model...')
    model = train_model(df)
    model_load_state.text('Model trained successfully!')

    # Input features
    
    # category_code
    category_code = ['advertising', 'cleantech', 'web', 'games_video', 'mobile',
       'software', 'network_hosting', 'consulting', 'finance',
       'education', 'medical', 'manufacturing', 'enterprise', 'other',
       'ecommerce', 'public_relations', 'hardware', 'search', 'analytics',
       'news', 'government', 'security', 'biotech', 'photo_video',
       'travel', 'health', 'semiconductor', 'social', 'legal',
       'transportation', 'hospitality', 'sports', 'nonprofit', 'fashion',
       'messaging', 'music', 'automotive', 'design', 'real_estate',
       'local', 'pets', 'nanotech']
    selected_category_code = st.selectbox("Category : ", category_code)
    st.text(selected_category_code)
    
    # country_code
    country_code = ['MAR', 'IND', 'USA', 'AUS', 'FRA', 'JPN', 'NLD', 'EGY', 'ISR',
       'GBR', 'THA', 'AUT', 'IRL', 'SWE', 'DEU', 'BRA', 'FIN', 'RUS',
       'SGP', 'MEX', 'CHN', 'ESP', 'ISL', 'KOR', 'TUR', 'DNK', 'PAK',
       'HUN', 'CAN', 'POL', 'GRC', 'PRT', 'BLR', 'CSS', 'MKD', 'ARG',
       'CHE', 'UKR', 'ITA', 'NZL', 'CZE', 'VNM', 'NOR', 'HRV', 'BEN',
       'CHL', 'GHA', 'ZAF', 'MYS', 'EST', 'BEL', 'SVK', 'TWN', 'CRI',
       'HKG', 'BGD', 'BOL', 'LBN', 'LUX', 'COL', 'PHL', 'ARE', 'IDN',
       'ROM', 'ANT', 'NGA', 'LKA', 'SWZ', 'VGB', 'MLT', 'SAU', 'KEN',
       'BGR', 'PER', 'LVA', 'CYP', 'LAO', 'NPL', 'MDA', 'CMR', 'UGA',
       'TUN', 'URY', 'MUS', 'VEN', 'OMN', 'ECU', 'KWT', 'JOR', 'LTU',
       'ALB', 'SVN', 'QAT', 'PST', 'REU', 'GTM', 'PCN', 'BHS', 'GEO',
       'BWA', 'DZA', 'GRD', 'GIB', 'PAN', 'MTQ', 'PRI', 'BIH', 'DMA',
       'BHR', 'SYC', 'SLE', 'TTO', 'VCT', 'ARA', 'BRB', 'NAM', 'SLV',
       'AFG', 'BLZ', 'UZB', 'LIE', 'IRN', 'ARM', 'ATG', 'UMI', 'DOM',
       'MDV', 'MMR', 'KAZ', 'JAM', 'GLP', 'IOT', 'MDG', 'VIR', 'AZE',
       'TZA', 'ZWE', 'PRY', 'PRK', 'SMR', 'IRQ', 'BMU', 'NRU', 'ETH',
       'BDI', 'CYM', 'SEN', 'NCL', 'CUB', 'FST', 'AGO', 'NFK', 'AND',
       'MCO', 'ZMB', 'KGZ', 'YEM', 'BRN', 'HTI', 'SUR', 'SYR', 'SOM',
       'RWA', 'CIV', 'SDN']
    selected_country_code = st.selectbox("Select Your Country Type: ", country_code)
    st.text(selected_country_code)
    
    # founded_at
    founded_at = st.number_input("Select founded year", min_value=1900, max_value=2024, value=1994)
    st.text(founded_at)
    
    # closed_at
    closed_at = st.number_input("Select closed year", min_value=1900, max_value=2024, value=2021)
    st.text(closed_at)
    
    # first_funding_at
    first_funding_at = st.number_input("Select first funding year", min_value=1900, max_value=2024, value=2005)
    st.text(first_funding_at)
    
    # last_funding_at
    last_funding_at = st.number_input("Select last funding year", min_value=1900, max_value=2024, value=2020)
    st.text(last_funding_at)
    
    # first_milestone_at
    first_milestone_at = st.number_input("Select first milestone year", min_value=1900, max_value=2024, value=2010)
    st.text(first_milestone_at)
    
    # last_milestone_at
    last_milestone_at = st.number_input("Select last milestone year", min_value=1900, max_value=2024, value=2021)
    st.text(last_milestone_at)
    
    # funding_rounds
    funding_rounds = st.radio("Select funding_rounds: ", [1,2,3])
    st.text(funding_rounds)
    
    # funding_total_usd
    funding_total_usd = st.number_input("Select last milestone year", min_value=291, max_value=26728460, value=30233)
    st.text(funding_total_usd)
    
    # milestones
    milestones = st.radio("Select milestones: ", [1,2,3,4,5,6,7])
    st.text(milestones)
    
    # relationships
    relationships = st.slider(' A float between 1-105', value=22.4,min_value=1.00,max_value=105.00)
    st.write(relationships)
    
    # IsClosed
    isClosed = st.radio("is it closed: ", ['Yes','No'])
    isClosed = 1 if isClosed == 'No' else 0
    st.text(isClosed)
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    selected_category_code_encoded = label_encoder.fit_transform([selected_category_code])[0]
    selected_country_code_encoded = label_encoder.fit_transform([selected_country_code])[0]
    
    
    # y_encoded = label_encoder.fit(df['bank_account'])

    # Prediction
    if st.button('Predict'):

        X_user = [selected_category_code_encoded, selected_country_code_encoded, 
                  founded_at, closed_at, first_funding_at, last_funding_at, 
                  first_milestone_at, last_milestone_at, funding_rounds, milestones, 
                  relationships, isClosed, funding_total_usd]  # Now includes 13 features
    
        prediction = model.predict([X_user])
        
        st.write('Prediction:', prediction)


if __name__ == '__main__':
    main()

