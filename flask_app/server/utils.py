import numpy as np
import json
import pickle
__locations = None
__data_columns = None
__model = None
# Functions
def get_estimated_price(location,total_sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)
def get_location_names():
    return __locations
def load_saved_artifacts():
    print("Loading saved artifacts ...start")
    global __data_columns
    global __locations
    with open("/home/ahmed/Desktop/DataScience_projects/BangaluruHousePriceAnalysis/server/artifacts/bangaluru_House_Price_Analysis_Columns.json",'r')as  f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    global __model
    with open("/home/ahmed/Desktop/DataScience_projects/BangaluruHousePriceAnalysis/server/artifacts/bangaluru_House_Price_Analysis_model.pickle",'rb')as  f:
        __model = pickle.load(f)
    print("Loding saved artifacts ...done")


# main 
if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Indira Nagar',1000,2,2))
    print(get_estimated_price('Indira Nagar',1000,3,3))
    print(get_estimated_price('2nd Stage Nagarbhavi',1000,3,3))
    print(get_estimated_price('2nd Stage Nagarbhavi',1000,2,2))
    print(get_estimated_price('Kalhalli',1000,2,2)) # other location
    print(get_estimated_price('Ejipura',1000,2,2))   # other location
