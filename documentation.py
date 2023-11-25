
#Data Cleaning and Wrangling 
import pandas as pd
import numpy as np
df=pd.read_csv ('Running log.csv')
#Parsing Date Column
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.date

#Removing last 18 rows of data
df = df.iloc[:-18]
#Selecting the columns
columns_to_keep = ['Activity Type', 'Date', 'Title', 'Distance', 'Time', 'Avg HR', 'Avg Pace', 'Total Ascent']
df_cleaned = df[columns_to_keep]




 # Graph visualization of breakdown of activities and number of days for each 

activity_counts = df_cleaned['Activity Type'].value_counts()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))

ax = activity_counts.plot(kind='bar', color='skyblue')

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.title('Number of Activities by Type')
plt.xlabel('Activity Type')
plt.ylabel('Number of Activities')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()





 # Line Chart visualization of frequency of activities over certain periods of time

df_cleaned_copy['Year'] = pd.to_datetime(df_cleaned_copy['Date']).dt.year

df_filtered = df_cleaned_copy[df_cleaned_copy['Year'].isin([2020, 2021, 2022, 2023])]

top_activities = df_filtered['Activity Type'].value_counts().nlargest(5).index

df_top_activities = df_filtered[df_filtered['Activity Type'].isin(top_activities)]

activity_year_count_top = df_top_activities.groupby(['Year', 'Activity Type']).size().reset_index(name='Count')

plt.figure(figsize=(15, 10))

for activity_type in top_activities:
    subset = activity_year_count_top[activity_year_count_top['Activity Type'] == activity_type]
    plt.plot(subset['Year'], subset['Count'], marker='o', label=activity_type)

plt.title('Changing Frequency of Top 5 Activities (2020-2023)')
plt.xlabel('Year')
plt.ylabel('Frequency of Activities')
plt.xticks([2020, 2021, 2022, 2023])
plt.legend(title='Activity Type')
plt.grid(True)

plt.show()





# Stacked bar charts visualization of frequency of activities over certain periods of time

df_cleaned_copy['Year'] = pd.to_datetime(df_cleaned_copy['Date']).dt.year

df_filtered = df_cleaned_copy[df_cleaned_copy['Year'].isin([2020, 2021, 2022, 2023])]

top_activities = df_filtered['Activity Type'].value_counts().nlargest(5).index

df_top_activities = df_filtered[df_filtered['Activity Type'].isin(top_activities)]

activity_year_count_top = df_top_activities.groupby(['Year', 'Activity Type']).size().reset_index(name='Count')

activity_pivot = activity_year_count_top.pivot(index='Year', columns='Activity Type', values='Count').fillna(0)

activity_pivot.plot(kind='bar', stacked=True, figsize=(12, 8))

plt.title('Frequency of Top 5 Activities (2020-2023)')
plt.xlabel('Year')
plt.ylabel('Frequency of Activities')
plt.xticks(rotation=45)

plt.show()




# Area chart visualization of frequency of activities over certain periods of time

df_cleaned_copy['Year'] = pd.to_datetime(df_cleaned_copy['Date']).dt.year

df_filtered = df_cleaned_copy[df_cleaned_copy['Year'].isin([2020, 2021, 2022, 2023])]

top_activities = df_filtered['Activity Type'].value_counts().nlargest(5).index

df_top_activities = df_filtered[df_filtered['Activity Type'].isin(top_activities)]

activity_year_count_top = df_top_activities.groupby(['Year', 'Activity Type']).size().reset_index(name='Count')

activity_pivot = activity_year_count_top.pivot(index='Year', columns='Activity Type', values='Count').fillna(0)

plt.figure(figsize=(12, 8))
for activity in top_activities:
    plt.fill_between(activity_pivot.index, activity_pivot[activity], label=activity, alpha=0.5)

plt.title('Frequency of Top 5 Activities (2020-2023)')
plt.xlabel('Year')
plt.ylabel('Frequency of Activities')
plt.xticks([2020, 2021, 2022, 2023])
plt.legend(title='Activity Type')

plt.show()





# Graph visualization of performance improvement/regression over period of time using Avg Pace indicator among Running activities

# Create a copy to avoid modifying the original DataFrame
df_running = df_cleaned_copy.copy()

# Filtering for the years 2020 to 2023 and Running activities
df_running = df_running[(df_running['Year'].isin([2020, 2021, 2022, 2023])) & 
                        (df_running['Activity Type'] == 'Running')]

# Function to convert 'mm:ss' to minutes
def convert_pace_to_minutes(pace_str):
    try:
        minutes, seconds = map(int, pace_str.split(':'))
        return minutes + seconds / 60
    except:
        return None  

df_running['Avg Pace Numeric'] = df_running['Avg Pace'].apply(convert_pace_to_minutes)

average_pace_per_year = df_running.groupby('Year')['Avg Pace Numeric'].mean()

plt.figure(figsize=(10, 6))
line_plot = average_pace_per_year.plot(kind='line', marker='o', color='blue')

# Adding the values of the average pace on the graph with an offset to avoid overlap with the line
offset = (average_pace_per_year.max() - average_pace_per_year.min()) * 0.02  # offset as 3% of range
for x, y in average_pace_per_year.items():
    plt.text(x, y + offset, f"{y:.2f}", fontsize=10, verticalalignment='bottom', horizontalalignment='center', color='black')

plt.title('Average Pace of Running Activities (2020-2023)')
plt.xlabel('Year')
plt.ylabel('Average Pace (minutes per kilometer)')
plt.xticks([2020, 2021, 2022, 2023])

plt.show()


#Frequency of Activity Type during COVID (2020-2021)
df_cleaned['Year'] = pd.to_datetime(df_cleaned['Date']).dt.year

df_filtered = df_cleaned[df_cleaned['Year'].isin([2020, 2021])]

activity_counts_over_years = df_filtered.groupby(['Year', 'Activity Type']).size().reset_index(name='Count')

plt.figure(figsize=(15, 10))

for activity_type in df_filtered['Activity Type'].unique():
    subset = activity_counts_over_years[activity_counts_over_years['Activity Type'] == activity_type]
    plt.plot(subset['Year'], subset['Count'], marker='o', label=activity_type)

plt.title('Frequency of Activities (2020-2021)')
plt.xlabel('Year')
plt.ylabel('Frequency of Activities')
plt.xticks([2020, 2021])
plt.legend(title='Activity Type')
plt.grid(True)

plt.show()


# Avg HR for each activity
avg_hr_by_activity = df_cleaned.groupby('Activity Type')['Avg HR'].mean().reset_index()

avg_hr_by_activity = avg_hr_by_activity.sort_values(by='Avg HR', ascending=False)

plt.figure(figsize=(12, 8))
plt.bar(avg_hr_by_activity['Activity Type'], avg_hr_by_activity['Avg HR'], color='skyblue')
plt.title('Average Heart Rate by Activity Type')
plt.xlabel('Activity Type')
plt.ylabel('Average Heart Rate')
plt.xticks(rotation=45, ha='right')

plt.show()




 #  Predictive model (RandomForestRegressor) to estimate average pace for upcoming races based on previous performance


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

if df_cleaned['Total Ascent'].dtype == object:
    df_cleaned['Total Ascent'] = pd.to_numeric(df_cleaned['Total Ascent'].str.replace(',', ''), errors='coerce')
else:
    df_cleaned['Total Ascent'] = pd.to_numeric(df_cleaned['Total Ascent'], errors='coerce')

# Filter for running activities with total ascent below 100
df_running = df_cleaned[(df_cleaned['Activity Type'] == 'Running') & (df_cleaned['Total Ascent'] < 100)]

# Convert 'Time' to total seconds for duration
df_running['Total Seconds'] = pd.to_timedelta(df_running['Time']).dt.total_seconds()

# Convert 'Avg Pace' to total seconds
def pace_to_seconds(pace_str):
    if isinstance(pace_str, str):
        minutes, seconds = map(int, pace_str.split(':'))
        return minutes * 60 + seconds
    return np.nan

df_running['Avg Pace Seconds'] = df_running['Avg Pace'].apply(pace_to_seconds)

# Handle non-numeric values for 'Distance'
df_running['Distance'] = pd.to_numeric(df_running['Distance'], errors='coerce')

# Dropping rows with missing values
df_running_clean = df_running.dropna(subset=['Distance', 'Total Seconds', 'Total Ascent', 'Avg Pace Seconds'])

# Features and target
features = ['Distance', 'Total Seconds', 'Total Ascent']
target = 'Avg Pace Seconds'

# Splitting the data
X = df_running_clean[features]
y = df_running_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting and evaluating
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('RMSE:', rmse)
print('R-squared:', r2)





# Graph visualization of predictive model.




