def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

class DataPipeline:

    def load_data(self):
        # Get the Data path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = Path(dir_path)
        parent_path = path.parent
        data_path = os.path.join(parent_path, "Data")

        # Load in the csv 
        csv_path = os.path.join(data_path, "new-york-city-airbnb")
        csv = os.path.join(csv_path, "AB_NYC_2019.csv")
        # Only use first 10 columns of data
        columns = [1, 2, 3, 4, 5, 8, 9]

        return pd.read_csv(csv, skipinitialspace=True, usecols=columns)

    def drop_missing_elements(self, data):
        # Drop the rows which have missing elements
        data = data.dropna()

        return data

    def preprocess_data(self, dataframe):
        # Convert text categories into numerical attributes
        X_columns = ["name", "host_id", "host_name", "neighbourhood_group", "neighbourhood", "room_type", "price"]
        X = dataframe[X_columns]
        ordinal_encoder = OrdinalEncoder()
        
        name = X[["name"]]
        host_name = X[["host_name"]]
        neighborhood_group = X[["neighbourhood_group"]]
        neighbourhood = X[["neighbourhood"]]
        room_type = X[["room_type"]]

        encoded_name = ordinal_encoder.fit_transform(name)
        encoded_host = ordinal_encoder.fit_transform(host_name)
        encoded_neighbourhood_group = ordinal_encoder.fit_transform(neighborhood_group)
        encoded_neighbourhood = ordinal_encoder.fit_transform(neighbourhood)
        encoded_room = ordinal_encoder.fit_transform(room_type)

        X["name"] = encoded_name
        X["host_name"] = encoded_host
        X["neighbourhood_group"] = encoded_neighbourhood
        X["neighbourhood"] = encoded_neighbourhood
        X["room_type"] = encoded_room

        # # Scale the features
        # pipeline = Pipeline([
        #     ('std_scaler', StandardScaler())
        # ])

        return X

    def split_train_test(self, data):
        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
        return train_set, test_set

    def train_model(self, samples):
        knn = NearestNeighbors(n_neighbors=20)
        knn.fit(samples)
        return knn

    def visualize_clusters(classifier):
        a=1

data_pipeline = DataPipeline()
data = data_pipeline.load_data()
data = data_pipeline.drop_missing_elements(data)

processed_data = data_pipeline.preprocess_data(data)

# *** At this point, the data has been ordinally encoded so that text attributes are now numerical. From this point I 
# ... will train the model and see what the outcomes are like. ***


classifier = data_pipeline.train_model(processed_data)
x = [processed_data.iloc[0]]
prediction = classifier.kneighbors(X=x, n_neighbors=4, return_distance=False)
print(prediction)

value = processed_data.iloc[13666]
print(data.iloc[0])
print(data.iloc[21527])
print(data.iloc[16485])

# *** THIS IS WHERE I AM. I need to checkout the predictions to see how accurate they are. Probably need to modify what attributes are being trained 
# ... (need to probably remove most features I am currently using and just keep super relevant ones)
