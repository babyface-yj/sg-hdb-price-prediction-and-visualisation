import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from datetime import datetime

class HDB:

    def __init__(self) -> None:
        
        # read data from files
        self.read_data()
    
    def read_data(self):
        self.df = pd.read_csv('resale_price.csv')
        current_time = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        self.df['To_current_date'] = (current_time - pd.to_datetime(self.df['month']) ).dt.days
        self.df['time_month'] = self.df['month'].str[5:].astype(int)
        self.df['time_year'] = self.df['month'].str[:4].astype(int)
        first_col = self.df.pop(self.df.columns[0]) 
        self.df[first_col.name] = first_col 
        self.df['remaining_lease'] = 99 + self.df['lease_commence_date'].astype(int) - self.df['time_year']
        self.df.drop(columns=['lease_commence_date', 'block','street_name','To_current_date'], inplace=True)
