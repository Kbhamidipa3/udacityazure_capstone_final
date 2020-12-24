#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://raw.githubusercontent.com/Kbhamidipa3/udacityazure_capstone_final/main/LVhotels-dataset.csv"
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required


subscription_id = 'f9d5a085-54dc-4215-9ba6-dad5d86e60a0'
resource_group = 'aml-quickstarts-131644'
workspace_name = 'quick-starts-ws-131644'


workspace = Workspace(subscription_id, resource_group, workspace_name)

ds = Dataset.get_by_name(workspace, name='LVhotels-dataset')


# In[6]:


def clean_data(data):
    # Dict for cleaning data
    months = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6, "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}
    weekdays = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
    # The following dictionaries are only used to replace missing information, other data is untouched
    country_cont = {"Australia":"Oceania", "Brazil":"South America", "Canada":"North America", "Denmark":"Europe", "Finland":"Europe", "Germany":"Europe", "Hawaii":"North America", "India":"Asia", "Ireland":"Europe", "Israel":"Asia", "Japan":"Asia", "Kuwait":"Asia", "Mexico":"North America", "Netherlands":"Europe", "Norway":"Europe", "Taiwan":"Asia", "Thailand":"Asia", "UK":"Europe", "USA":"North America", "Saudi Arabia":"Asia"}
    hotel_rooms = {"The Cromwell":188, "Hilton Grand Vacations on the Boulevard":1228, "Marriott's Grand Chateau":643, "Wyndham Grand Desert":787}
    period_month = {"Dec-Feb":"January", "Mar-May":"April", "Jun-Aug":"July", "Sep-Nov":"October"}
    hotel_day = {"The Cromwell":"Sunday", "Hilton Grand Vacations on the Boulevard":"Sunday", "Marriott's Grand Chateau":"Sunday", "Wyndham Grand Desert":"Sunday"}
    # Clean the data
    x_df = data.to_pandas_dataframe()
    x_df.columns = [c.replace(' ', '_') for c in x_df.columns]
    # Aligning the data to work as a classification problem
    # Redefining "Score" to have a value of 1 if score is >=3 and 0 otherwise
    x_df["Score"] = x_df.Score.apply(lambda s: 1 if s >= 3 else 0)
    # Fill in the missing data
    x_df.User_continent_dup = x_df.User_continent.fillna(x_df.User_country)
    x_df["User_continent_dup"] = x_df.User_continent_dup.map(country_cont)
    x_df.columns = x_df.columns.str.replace('User_continent_dup', 'User_continent')
    x_df.columns = x_df.columns.str.replace('Nr._', 'Nr_')
    x_df.Nr_rooms_dup = x_df.Nr_rooms.fillna(x_df.Hotel_name)
    x_df["Nr_rooms_dup"] = x_df.Nr_rooms_dup.map(hotel_rooms)
    x_df.columns = x_df.columns.str.replace('Nr_rooms_dup', 'Nr_rooms')
    x_df.Review_month_dup = x_df.Review_month.fillna(x_df.Period_of_stay)
    x_df["Review_month_dup"] = x_df.Review_month_dup.map(period_month)
    x_df.columns = x_df.columns.str.replace('Review_month_dup', 'Review_month')
    x_df.Review_weekday_dup = x_df.Review_weekday.fillna(x_df.Hotel_name)
    x_df["Review_weekday_dup"] = x_df.Review_weekday_dup.map(hotel_day)
    x_df.columns = x_df.columns.str.replace('Review_weekday_dup', 'Review_weekday')
    s = x_df.stack()
    x_df = s.unstack()
    #Replace blank and negative cells under Member years in a dataframe with random values between 1 and 10
    x_df['Member_years'] = x_df['Member_years'].apply(lambda l: l if l>0 else np.random.choice([1, 10]))
    # Restoring all entries to their default datatypes    x_df[['Nr_reviews','Nr_rooms','Nr_hotel_reviews','Member_years','Score', 'Helpful_votes', 'Hotel_stars', 'Nr_reviews']]=x_df[['Nr_reviews','Nr_rooms','Nr_hotel_reviews','Member_years','Score', 'Helpful_votes', 'Hotel_stars', 'Nr_reviews']].astype(np.int64)
    x_df.to_csv("LV-github-automl.csv")    
    # Replace with one hot encode data
    User_countries = pd.get_dummies(x_df.User_country, prefix="User_country")
    x_df.drop("User_country", inplace=True, axis=1)
    x_df = x_df.join(User_countries)
    Stay_periods = pd.get_dummies(x_df.Period_of_stay, prefix="Period_of_stay")
    x_df.drop("Period_of_stay", inplace=True, axis=1)
    x_df = x_df.join(Stay_periods)
    Traveler_types = pd.get_dummies(x_df.Traveler_type, prefix="Traveler_type")
    x_df.drop("Traveler_type", inplace=True, axis=1)
    x_df = x_df.join(Traveler_types)
    Hotel_names = pd.get_dummies(x_df.Hotel_name, prefix="Hotel_name")
    x_df.drop("Hotel_name", inplace=True, axis=1)
    x_df = x_df.join(Hotel_names)
    User_continents = pd.get_dummies(x_df.User_continent, prefix="User_continent")
    x_df.drop("User_continent", inplace=True, axis=1)
    x_df = x_df.join(User_continents)
    x_df["Pool"] = x_df.Pool.apply(lambda s: 1 if s else 0)
    x_df["Gym"] = x_df.Gym.apply(lambda s: 1 if s else 0)
    x_df["Tennis_court"] = x_df.Tennis_court.apply(lambda s: 1 if s else 0)
    x_df["Spa"] = x_df.Spa.apply(lambda s: 1 if s else 0)
    x_df["Casino"] = x_df.Casino.apply(lambda s: 1 if s else 0)
    x_df["Free_internet"] = x_df.Free_internet.apply(lambda s: 1 if s else 0)
    x_df["month"] = x_df.Review_month.map(months)
    x_df["weekday"] = x_df.Review_weekday.map(weekdays)
    x_df.drop("Review_month", inplace=True, axis=1)
    x_df.drop("Review_weekday", inplace=True, axis=1)
    x_df.to_csv("LV-github-hypertune.csv")   
    # Separate out the label data from the remainder of the dataframe
    y_df = x_df.pop("Score").astype(int)
    x_df.to_csv("LV-github-data.csv") 
    y_df.to_csv("LV-github-label.csv") 

    return (x_df,y_df)
    
x, y = clean_data(ds)



# In[7]:


# TODO: Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)

run = Run.get_context()


# In[8]:


def main():
    # Add arguments to script

    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations to converge")


    args = parser.parse_args(args=[]) #call from notebook


    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    print(accuracy)

    # Save the trained model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/hp_trained_model.pkl')

if __name__ == '__main__':
    main()

