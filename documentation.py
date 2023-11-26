
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

import matplotlib.pyplot as plt

# Assuming df_cleaned and other processing steps are already done

# Create the figure
plt.figure(figsize=(15, 10))

# Plotting each activity type
for activity_type in top_activities:
    subset = activity_year_count_top[activity_year_count_top['Activity Type'] == activity_type]
    plt.plot(subset['Year'], subset['Count'], marker='o', label=activity_type)
    
    # Annotate each point with the count value
    for i, row in subset.iterrows():
        plt.annotate(str(row['Count']), (row['Year'], row['Count']), textcoords="offset points", xytext=(0,10), ha='center')

# Setting the chart title and labels
plt.title('Changing Frequency of Top 5 Activities (2020-2023)')
plt.xlabel('Year')
plt.ylabel('Frequency of Activities')
plt.xticks([2020, 2021, 2022, 2023])
plt.legend(title='Activity Type')
plt.grid(True)

# Show the plot
plt.show()





# Stacked bar charts visualization of frequency of activities over certain periods of time

df_cleaned['Year'] = pd.to_datetime(df_cleaned['Date']).dt.year

df_filtered = df_cleaned[df_cleaned['Year'].isin([2020, 2021, 2022, 2023])]

top_activities = df_filtered['Activity Type'].value_counts().nlargest(5).index

df_top_activities = df_filtered[df_filtered['Activity Type'].isin(top_activities)]

activity_year_count_top = df_top_activities.groupby(['Year', 'Activity Type']).size().reset_index(name='Count')

activity_pivot = activity_year_count_top.pivot(index='Year', columns='Activity Type', values='Count').fillna(0)

# Plotting the stacked bar chart
ax = activity_pivot.plot(kind='bar', stacked=True, figsize=(12, 8))

# Adding annotations for each bar
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy() 
    if height > 0:  # To avoid displaying annotations for empty bars
        ax.annotate(f'{int(height)}', (x + width/2, y + height/2), ha='center')

# Setting the chart title and labels
plt.title('Frequency of Top 5 Activities (2020-2023)')
plt.xlabel('Year')
plt.ylabel('Frequency of Activities')
plt.xticks(rotation=45)

# Show the plot
plt.show()








# Graph visualization of performance improvement/regression over period of time using Avg Pace indicator among Running activities

# A gpraph to show changes in average pace for Running activities over the past years in minutes/mile
import matplotlib.pyplot as plt

# Assuming df_running is your DataFrame with the necessary columns

# Function to convert 'mm:ss' to minutes per kilometer, then to minutes per mile
def convert_pace_to_minutes_per_mile(pace_str):
    try:
        minutes, seconds = map(int, pace_str.split(':'))
        pace_per_km = minutes + seconds / 60
        pace_per_mile = pace_per_km * 1.60934  # converting km to miles
        return pace_per_mile
    except:
        return None

# Function to convert minutes to 'mm:ss' format
def convert_minutes_to_mm_ss(minutes):
    total_seconds = int(minutes * 60)
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"

df_running['Avg Pace Numeric Miles'] = df_running['Avg Pace'].apply(convert_pace_to_minutes_per_mile)

average_pace_per_year_miles = df_running.groupby('Year')['Avg Pace Numeric Miles'].mean()

# Convert average pace from minutes to 'mm:ss' format
average_pace_per_year_miles_formatted = average_pace_per_year_miles.apply(convert_minutes_to_mm_ss)

plt.figure(figsize=(10, 6))
line_plot = average_pace_per_year_miles.plot(kind='line', marker='o', color='blue')

# Adding the values of the average pace on the graph with an offset to avoid overlap with the line
offset = (average_pace_per_year_miles.max() - average_pace_per_year_miles.min()) * 0.02  # offset as 2% of range
for x, y in zip(average_pace_per_year_miles.index, average_pace_per_year_miles):
    formatted_pace = convert_minutes_to_mm_ss(y)
    plt.text(x, y + offset, formatted_pace, fontsize=10, verticalalignment='bottom', horizontalalignment='center', color='black')

plt.title('Average Pace of Running Activities (2020-2023)')
plt.xlabel('Year')
plt.ylabel('Average Pace (minutes per mile)')
plt.xticks([2020, 2021, 2022, 2023])

plt.show()



# A gpraph to show changes in average pace for Running activities over the past years in minutes/km
# Function to convert 'mm:ss' to minutes
# Convert pace to minutes
def convert_pace_to_minutes(pace_str):
    try:
        minutes, seconds = map(int, pace_str.split(':'))
        return minutes + seconds / 60
    except:
        return None

# Convert minutes to 'mm:ss' format
def convert_minutes_to_mm_ss(minutes):
    total_seconds = int(minutes * 60)
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"
    
# Apply conversion to DataFrame
df_running['Avg Pace Numeric'] = df_running['Avg Pace'].apply(convert_pace_to_minutes)

# Calculate average pace per year
average_pace_per_year = df_running.groupby('Year')['Avg Pace Numeric'].mean()

# Convert average pace from minutes to 'mm:ss' format
average_pace_per_year_formatted = average_pace_per_year.apply(convert_minutes_to_mm_ss)

# Create the line plot
plt.figure(figsize=(10, 6))
line_plot = average_pace_per_year.plot(kind='line', marker='o', color='blue')

# Adding the values of the average pace on the graph
offset = (average_pace_per_year.max() - average_pace_per_year.min()) * 0.02  # offset as 2% of range
for x, y in average_pace_per_year.items():
    formatted_pace = convert_minutes_to_mm_ss(y)
    plt.text(x, y + offset, formatted_pace, fontsize=10, verticalalignment='bottom', horizontalalignment='center', color='black')

# Set chart details
plt.title('Average Pace of Running Activities (2020-2023)')
plt.xlabel('Year')
plt.ylabel('Average Pace (minutes per kilometer)')
plt.xticks([2020, 2021, 2022, 2023])

# Display the plot
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



#Testing the model

# Hypothetical data for a 10km race with 150 meters total ascent
hypothetical_data = {
    'Distance': [10],  # 10 kilometers
    'Total Seconds': [0],  # placeholder, will not be used in prediction
    'Total Ascent': [50]  # 50 meters of total ascent
}

# Creating a DataFrame from the hypothetical data
df_predict = pd.DataFrame(hypothetical_data)

# Predicting the average pace using the model
predicted_pace_seconds = model.predict(df_predict)[0]  # [0] to extract the single prediction value

# Display the predicted average pace in seconds per kilometer
print("Predicted Average Pace (seconds per kilometer):", predicted_pace_seconds)

# Converting the result in to mm:ss format

# Function to convert seconds to mm:ss format
def seconds_to_pace(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes):02d}:{int(seconds):02d}"

# Example: Converting the predicted pace from seconds to mm:ss per km format
predicted_pace_seconds = 230.68 
predicted_pace_mm_ss = seconds_to_pace(predicted_pace_seconds)

print("Predicted Average Pace:", predicted_pace_mm_ss, "per kilometer")


#Predictive Model for Avg HR based on Time, Distance and Total Ascent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df=pd.read_csv ('Running log.csv')
#Parsing Date Column
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.date
#Removing last 18 rows of data
df = df.iloc[:-18]
#Selecting the columns
columns_to_keep = ['Activity Type', 'Date', 'Title', 'Distance', 'Time', 'Avg HR', 'Avg Pace', 'Total Ascent']
df_cleaned = df[columns_to_keep]
df_cleaned = df_cleaned.replace('--', pd.NA)
df_cleaned = df_cleaned.dropna(subset=['Distance', 'Time', 'Avg HR', 'Total Ascent'], how='any')
df_cleaned['Time'] = pd.to_timedelta(df_cleaned['Time']).dt.total_seconds()
df_cleaned['Total Ascent'] = pd.to_numeric(df_cleaned['Total Ascent'].str.replace(',', ''), errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['Distance', 'Time', 'Avg HR', 'Total Ascent'], how='any')


X = df_cleaned[['Distance', 'Time', 'Total Ascent']]
y = df_cleaned['Avg HR']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)


# Graph visualization of predictive model for Avg HR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df=pd.read_csv ('Running log.csv')
#Parsing Date Column
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.date
#Removing last 18 rows of data
df = df.iloc[:-18]
#Selecting the columns
columns_to_keep = ['Activity Type', 'Date', 'Title', 'Distance', 'Time', 'Avg HR', 'Avg Pace', 'Total Ascent']
df_cleaned = df[columns_to_keep]
df_cleaned = df_cleaned.replace('--', pd.NA)
df_cleaned = df_cleaned.dropna(subset=['Distance', 'Time', 'Avg HR', 'Total Ascent'], how='any')
df_cleaned['Time'] = pd.to_timedelta(df_cleaned['Time']).dt.total_seconds()
df_cleaned['Total Ascent'] = pd.to_numeric(df_cleaned['Total Ascent'].str.replace(',', ''), errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['Distance', 'Time', 'Avg HR', 'Total Ascent'], how='any')


X = df_cleaned[['Distance', 'Time', 'Total Ascent']]
y = df_cleaned['Avg HR']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

#Scatter Plot for Actual vs Predicted Avg HR
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Avg HR")
plt.ylabel("Predicted Avg HR")
plt.title("Actual vs. Predicted Avg HR")
plt.show()

# Histrogram of Distribution Errors
differences = y_test - y_pred

plt.hist(differences, bins=30, edgecolor='black')
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# Graph visualization of predictive model for Avg Pace 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


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

#Scatter Plot for Actual vs Predicted Avg Pace 
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Avg Pace (Seconds)")
plt.ylabel("Predicted Avg Pace (Seconds)")
plt.title("Actual vs. Predicted Avg Pace")
plt.show()

#Histogram of Distribution Errors
difference = y_test - y_pred

plt.hist(difference, bins=30, edgecolor='black')
plt.xlabel("Actual - Predicted")
plt.ylabel("Frequency")
plt.title("Distribution Errors")
plt.show()

