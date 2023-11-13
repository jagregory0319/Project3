
#Data Cleaning and Wrangling 
import pandas as pd
import numpy as np
df=pd.read_csv ('Running log.csv')
#Parsing Date Column
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.date
#Filter specefied rows 







 # Graph visualization of breakdown of activities and number of days for each 





 # Graph visualization of frequency of activities over certain periods of time





 # Graph visualization of performance improvement/regression over period of time 





 #  Predictive model to estimate times for upcoming races based on previous performance




# Graph visualization of predictive model.




