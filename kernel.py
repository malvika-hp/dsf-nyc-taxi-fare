
# coding: utf-8

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import sklearn.preprocessing as prep


# In[5]:


# Read the input csv
inputFile = './train.csv'
totalLines = sum(1 for l in open(inputFile)) - 1
sampleSize = 20000000
skip = sorted(random.sample(range(1,totalLines+1),totalLines-sampleSize)) 
df_train = pd.read_csv(inputFile, skiprows=skip)


# CLEANING THE DATA

# In[6]:


# Understanding the summary statistics
df_train.describe() 


# In[7]:


# Checking for missing values across all columns
df_train.isnull().sum()


# In[8]:


# Removing the rows with missing values ( only dropoff_longitude is missing )
df_train = df_train[np.isfinite(df_train['dropoff_longitude'])] 


# In[14]:


#Observing the maximum value of different fields to check for anomalies
df_train.loc[df_train['dropoff_longitude'].idxmax()]


# In[15]:


#Observing the maximum value of different fields to check for anomalies
df_train.loc[df_train['dropoff_latitude'].idxmax()] 


# In[16]:


#Observing the maximum value of different fields to check for anomalies
df_train.loc[df_train['pickup_longitude'].idxmax()]


# In[17]:


#Observing the maximum value of different fields to check for anomalies
df_train.loc[df_train['pickup_latitude'].idxmax()]


# In[18]:


#Observing the count of non permissable values of latitudes
print(df_train.query('pickup_latitude>90 or pickup_latitude<-90').count());


# In[19]:


#Observing the count of non permissable values of longitudes
print(df_train.query('pickup_longitude>180 or pickup_longitude<-180').count());


# In[20]:


#Removing the non permissable values for pickup latitutdes and longitudes
df_train = df_train.query('pickup_longitude>-180 and pickup_longitude<180 and pickup_latitude>-90 and pickup_latitude<90')


# In[21]:


#Removing the non permissable values for dropoff latitutdes and longitudes
df_train = df_train.query('dropoff_longitude>-180 and dropoff_longitude<180 and dropoff_latitude>-90 and dropoff_latitude<90')


# In[22]:


#Observing the max value of fare_amount to judge anomaly
df_train.loc[df_train['fare_amount'].idxmax()]


# In[27]:


#Observing the no of rows with high fare
print(df_train.query('fare_amount>500').count())


# In[26]:


#Observing the histogram to get an idea of fare frequecy and judge outliers
plt.figure(figsize = (10, 6))
sns.distplot(df_train['fare_amount']);
plt.title('Distribution of Fare');


# In[28]:


#Removing anomalous fare amount rows
df_train = df_train.query('fare_amount<500')


# In[29]:


#Removing negtaive fare amounts
df_train = df_train.query('fare_amount>0')


# In[30]:


# Plotting pickup_latitude to understand its distribution and remove anomalous values
plt.figure(figsize = (10, 6))
sns.distplot(df_train['pickup_latitude']);
plt.title('Distribution of pickup_latitude');


# In[31]:


# Removing the pickup latitude that are unreasonable, or too much deviated from the population as observed in plot above
df_train = df_train.loc[df_train["pickup_latitude"].between(40, 42)]


# In[32]:


# Plotting dropoff_latitude to understand its distribution and remove anomalous values
plt.figure(figsize = (10, 6))
sns.distplot(df_train['dropoff_latitude']);
plt.title('Distribution of dropoff_latitude');


# In[33]:


# Removing the dropoff latitude that are unreasonable, or too much deviated from the population as observed in plot above
df_train = df_train.loc[df_train["dropoff_latitude"].between(40, 42)]


# In[34]:


# Plotting pickup_longitude to understand its distribution and remove anomalous values
plt.figure(figsize = (10, 6))
sns.distplot(df_train['pickup_longitude']);
plt.title('Distribution of pickup_longitude');


# In[35]:


# Removing the pickup longitudes that are unreasonable, or too much deviated from the population as observed in plot above
df_train = df_train.loc[df_train["pickup_longitude"].between(-75, -72)]


# In[36]:


# Plotting dropoff_longitude to understand its distribution and remove anomalous values
plt.figure(figsize = (10, 6))
sns.distplot(df_train['dropoff_longitude']);
plt.title('Distribution of dropoff_longitude');


# In[37]:


# Removing the dropoff longitudes that are unreasonable, or too much deviated from the population as observed in plot above
df_train = df_train.loc[df_train["dropoff_longitude"].between(-75, -72)]


# In[38]:


#Observing the rows with lot of passenger_counts
df_train[df_train.passenger_count>6]


# In[39]:


#Removing suh rows with passnegers >6 as it is not allowed hence unlikely, it maybe erroneous data
df_train = df_train.drop(df_train[df_train.passenger_count>6].index)


# ENRICHING DATA ( ADDING REQUIRED FEATURES )

# In[9]:


# Converting the pickup_datetime to hour, year - this may be used as a feature
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z')
df_train['hour'] = df_train['pickup_datetime'].dt.hour
df_train['year'] = df_train['pickup_datetime'].dt.year


# In[10]:


# Function for calculating Haversine distance between longitude and latitude
def calculateHaversineDistance():
    R = 3959 # Radius of earth in miles
    phi1 = np.radians(df_train['pickup_latitude'])
    phi2 = np.radians(df_train['dropoff_latitude'])
    phi_chg = np.radians(df_train['pickup_latitude'] - df_train['dropoff_latitude'])
    delta_chg = np.radians(df_train['pickup_longitude'] - df_train['dropoff_longitude'])
    a = np.sin(phi_chg / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg/2)**2
    c = 2 * np.arcsin(a ** 0.5)
    d = R * c
    return d


# In[11]:


# Converting the latitude and longitude to measure the travel distance on earth using haversenian formula
df_train['travel_distance'] = calculateHaversineDistance()


# In[12]:


# Computing the Eucledian distance and adding to the dataframe
df_train['eucledian_distance'] = (df_train.pickup_longitude.sub(df_train.dropoff_longitude).pow(2).add(df_train.pickup_latitude.sub(df_train.dropoff_latitude).pow(2))).pow(.5)         


# In[13]:


#Computing latitude and longitude differences
df_train['longitude_diff']=np.abs(df_train['pickup_longitude']-df_train['dropoff_longitude'])
df_train['latitude_diff']=np.abs(df_train['pickup_latitude']-df_train['dropoff_latitude'])


# COMPUTING CORRELATION

# In[40]:


#Computing the pearson correlation coffecient
df_train.corr(method='pearson')


# VIZUALIZING RELATIONSHIPS ( PLOTS AND FIGURES )

# In[41]:


# Visualizing the plot between Eucledian distance and Fare amount for some sample values
df_sample = df_train.sample(5000)
plt.plot(df_sample["eucledian_distance"], df_sample['fare_amount'], 'ro')
plt.xlabel('Eucledian distance')
plt.ylabel('Fare amount')
plt.show()


# In[90]:


df_sample = df_train.sample(5000)
plt.plot(df_sample["travel_distance"], df_sample['fare_amount'], 'ro')
plt.xlabel('Haversine distance')
plt.ylabel('Fare amount')
plt.show()


# In[42]:


# Visualizing the plot between Time of the day(in hour) and Haversine distance tavelled for some sample values
df_sample = df_train.sample(5000)
plt.plot(df_sample["hour"], df_sample['travel_distance'], 'ro')
plt.xlabel('Time of the day')
plt.ylabel('Haversine distance tarvelled')
plt.show()


# In[43]:


# Visualizing the plot between Time of the day(in hour) and mean Haversine distance travelled
df_mean_distance_by_hour = df_train.groupby('hour')['travel_distance'].mean()
plt.plot(df_mean_distance_by_hour.keys(), df_mean_distance_by_hour)
plt.xlabel('Time of the day')
plt.ylabel('Mean Distance travelled')
plt.show()


# In[44]:


# Visualizing the plot between Time of the day(in hour) and Fare amount for some sample values
df_sample = df_train.sample(5000)
plt.plot(df_sample["hour"], df_sample['fare_amount'], 'ro')
plt.xlabel('Time of the day')
plt.ylabel('Fare collected')
plt.show()


# In[45]:


# Visualizing the plot between Time of the day(in hour) and mean taxi fare 
df_mean_fare_by_hour = df_train.groupby('hour')['fare_amount'].mean()
plt.plot(df_mean_fare_by_hour.keys(), df_mean_fare_by_hour)
plt.xlabel('Time of the day')
plt.ylabel('Mean fare amount')
plt.show()


# In[46]:


# Visualizing the plot between sum of fare amount collected by year
df_sum_fare_by_year = df_train.groupby('year')['fare_amount'].sum()
plt.plot(df_sum_fare_by_year.keys(), df_sum_fare_by_year)
plt.xlabel('Year')
plt.ylabel('Sum of fare amount')
plt.show()


# In[47]:


# Visualizing the plot between sum of distance travelled by year
df_sum_distance_by_year  = df_train.groupby('year')['travel_distance'].sum()
plt.plot(df_sum_distance_by_year.keys(), df_sum_distance_by_year)
plt.xlabel('Year')
plt.ylabel('Sum of distance travelled')
plt.show()


# LINEAR REGRESSION MODELING

# In[70]:


# Fitting Linear regression model
from sklearn.linear_model import LinearRegression
features = df_train.drop(['key','fare_amount','pickup_datetime','passenger_count','travel_distance','dropoff_latitude','dropoff_longitude','pickup_longitude','pickup_latitude'], axis=1)
lm = LinearRegression()
min_max_scaler = prep.MinMaxScaler()
normalized_features = min_max_scaler.fit_transform(features)
lm.fit(normalized_features,df_train['fare_amount'])
print(list(zip(lm.coef_, features)))


# In[79]:


# Checking the performance for LR
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(lm.predict(normalized_features), df_train['fare_amount'])))


# In[76]:


# Enriching test data
test_data = "./test.csv"
df_test = pd.read_csv(test_data)
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z')
df_test['hour'] = df_test['pickup_datetime'].dt.hour
df_test['year'] = df_test['pickup_datetime'].dt.year
R = 3959 # Radius of earth in miles
phi1 = np.radians(df_test['pickup_latitude'])
phi2 = np.radians(df_test['dropoff_latitude'])
phi_chg = np.radians(df_test['pickup_latitude'] - df_test['dropoff_latitude'])
delta_chg = np.radians(df_test['pickup_longitude'] - df_test['dropoff_longitude'])
a = np.sin(phi_chg / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg/2)**2
c = 2 * np.arcsin(a ** 0.5)
d = R * c
df_test['travel_distance'] = d
df_test['eucledian_distance'] = (df_test.pickup_longitude.sub(df_test.dropoff_longitude).pow(2).add(df_test.pickup_latitude.sub(df_test.dropoff_latitude).pow(2))).pow(.5)
df_test['longitude_diff']=np.abs(df_test['pickup_longitude']-df_test['dropoff_longitude'])
df_test['latitude_diff']=np.abs(df_test['pickup_latitude']-df_test['dropoff_latitude'])


# In[77]:


# Linear regression testing
features_test = df_test.drop(['key','pickup_datetime','passenger_count','travel_distance','dropoff_latitude','dropoff_longitude','pickup_longitude','pickup_latitude'], axis=1)
normalized_feature_test = min_max_scaler.transform(features_test)
predictedY = lm.predict(normalized_feature_test)
resultDataColumns = {'key': df_test['key'], 'fare_amount': predictedY }
resultDataframe = pd.DataFrame(data=resultDataColumns)


# In[78]:


# Writing data
resultDataframe.to_csv('linear_regression_output.csv')


# RANDOM FOREST MODELING

# In[95]:


# Fitting random forest regression model
from sklearn.ensemble import RandomForestRegressor
df_sample = df_train.sample(1000000)
rf = RandomForestRegressor(n_estimators = 10, min_samples_split=5)
features_rf = df_sample.drop(['key','fare_amount','pickup_datetime','passenger_count','year'], axis=1)
rf.fit(features_rf,df_sample['fare_amount']);
rfPredictions = rf.predict(features_rf)


# In[96]:


print(list(zip(rf.feature_importances_, features_rf)))


# In[97]:


# Checking the performance for RF
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(rfPredictions, df_sample['fare_amount'])))


# In[100]:


# Random forest Testing
features_test = df_test.drop(['key','pickup_datetime','passenger_count','year'], axis=1)
rfPredictedY = rf.predict(features_test)
resultDataColumns = {'key': df_test['key'],'fare_amount': rfPredictedY }
resultDataframe = pd.DataFrame(data=resultDataColumns)
resultDataframe.head()


# In[101]:


# Writing data
resultDataframe.to_csv('random_forest_output.csv')

