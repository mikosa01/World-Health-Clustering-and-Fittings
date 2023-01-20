# -*- coding: utf-8 -*-
"""
Clustering and Fitting
CO2

Michael Merry
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
import seaborn as sns
import scipy.optimize as opt
import err_ranges as err

def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f

def file_reader(path, indicator_name): 
    '''
    This function is created to read a csv file, then create a specfic
    indicator table and then transforms it. 
    
    Agrs : 
        path : The dataset(csv) file path. 
        indicator name : The indicator table to be created. 
    
    Return :
        It returns dataframe, transformed dataframe and the indicator name. 
    '''
    
    data = pd.read_csv(path, skiprows = 3)
    data.set_index('Country Name', inplace = True)
    df = data[data['Indicator Name']== indicator_name]
    df.drop(['Country Code', 'Indicator Name', 'Indicator Code',
             'Unnamed: 66'], 1, inplace = True )
    df.fillna(0, inplace = True)
    df_t =df.T
    return df, df_t, indicator_name

def heatmap (data, data_t, ind_name ): 
    '''
    This functions creates an heatmap of country and years with
    relation to the indication.
    
    Parameter : 
        data : This is the normal dataframe 
        data_t  : This is the transform dataframe 
        Ind_name : Name of the selected indicator
    Return: 
        It returns two heatmap plot with respect to the indicator 
    '''
    plt.figure(figsize = (10, 10))
    sns.heatmap(data.corr())
    plt.title ('Year correlation w.r.t {}'.format(ind_name))
    plt.show()
    
    plt.figure(figsize = (10, 10))
    plt.figure(figsize = (5, 5))
    data_t =data_t.loc[:, ['China', 'Germany', 'Iceland','Israel',
                           'Italy', 'Japan']]
    sns.heatmap(data_t.corr())
    plt.title ('Countries correlation w.r.t {}'.format(ind_name))
    plt.show()
    
    plt.show()
    
def normalizer (df, year1, year2):
    '''
    This function is a normalizer. It splits the dataframe and
    scales the dataframe.
    
    Parameters: 
        data : dataframe to splits and scaled 
        year_1 : First column
        year_2 : Second Column
        
    Return : 
        it returns a scaled dataframe and the respective columns
    '''
    df = df.loc[:, year1:year2]
    min_e = df - df.min()
    max_e = df.max() - df.min()
    scale = (min_e  / max_e) 
    df_scale = scale.values 
    return df_scale

def kmeans_cluster(scaler):
    '''
    This functions performs kmeans clustering with 3 number of clusters on
    the scaled dataframe. 
    
    Paramater:
        scaler : The scaled dataframe
    
    Return:
        A plot that shows the cluster membership and cluster centres    
    '''    
    num_cluster = 3
    kmeans = KMeans(n_clusters = num_cluster)
    kmeans.fit(scaler)
    labels = kmeans.labels_
    cen = np.array(kmeans.cluster_centers_)
    
    col1 = scaler[labels ==0]
    col2 = scaler[labels ==1]
    col3 = scaler[labels ==2]
    labels=['1', '2', '3']
    
    sns.scatterplot(x=col1[:,0], y=col1[:,1], label=labels[0])
    sns.scatterplot(x=col2[:,0], y=col2[:,1], label=labels[1])
    sns.scatterplot(x=col3[:,0], y=col3[:,1], label=labels[2])
    
    plt.title('CO2 emissions (metric tons per capita)')
    plt.scatter(cen[:, 0], cen[:, 1], s =5, marker = 'x', color = 'black')
    plt.show()

# Use file_reader function to create dataframes
data, data_t, ind_title = file_reader('API_19_DS2_en_csv_v2_4700503.csv',
                                      'CO2 emissions (metric tons per capita)')

print(data.head())
print(data_t.head())

# Choose especifics countries
data_T = data_t.loc[:, ['China', 'Germany', 'Iceland','Israel',
                        'Italy', 'Japan']]

# Plot scatter matrix of country transposed dataframe
plt.figure (figsize = (10, 10))
pd.plotting.scatter_matrix(data_T, figsize=(10,10), s=5, alpha=0.8);

heatmap(data, data_t, ind_title)

# Use normalize function to normalize dataframe
df_scale = normalizer(data, '2017', '2019')
# Create kmeans
kmeans_cluster(df_scale)

df = pd.read_csv('API_19_DS2_en_csv_v2_4700503.csv', skiprows=4)
print(df)
indicator = 'CO2 emissions (metric tons per capita)'

df = df.loc[df['Indicator Name'] == indicator]
em_china = df[df['Country Name'] == 'China']
em_china = em_china.dropna(axis=1)
em_china = em_china.T
em_china.columns = [indicator]
em_china = em_china.iloc[4:]
em_china = em_china.iloc[:-1]
em_china = em_china.reset_index()
em_china = em_china.rename(columns={"index": "Year"})
em_china["Year"] = pd.to_numeric(em_china["Year"])

# Plot the graph of co2 x year
em_china.plot("Year", indicator)
plt.title('CO2 emissions (metric tons per capita) Vs. Year - China')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.show()

print(em_china.info())

param, covar = opt.curve_fit(exponential, em_china["Year"],
                             em_china[indicator],
                             p0=(em_china[indicator].min(), 0.03))

# Plot the graph of co2 x year and exponencial fitted co2 x year
em_china["fit"] = exponential(em_china["Year"], *param)
em_china.plot("Year", [indicator, "fit"])
plt.title('CO2 emissions (metric tons per capita) Vs. Year - China')
plt.ylabel(indicator)
plt.show()

param, covar = opt.curve_fit(logistic, em_china["Year"], em_china[indicator],
                             p0=(em_china[indicator].min(), 0.05,
                                 em_china["Year"].min()))

# Plot the graph of co2 x year and logistic fitted co2 x year
sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
em_china["fit"] = logistic(em_china["Year"], *param)
em_china.plot("Year", [indicator, "fit"])
plt.ylabel(indicator)
plt.title('CO2 emissions (metric tons per capita) Vs. Year - China')
plt.show()

# Create arrange of years
year = np.arange(em_china["Year"].min(), 2041)
print(year)
# Create forecast
forecast = logistic(year, *param)

# Plot the graph of co2 x year and logistic fitted co2 x year with prediction
plt.figure()
plt.plot(em_china["Year"], em_china[indicator], label=indicator)
plt.plot(year, forecast, label="forecast")

plt.title('CO2 emissions (metric tons per capita) Vs. Year - China')
plt.xlabel("year")
plt.ylabel(indicator)
plt.legend()
plt.show()

low, up = err.err_ranges(year, logistic, param, sigma)

# Plot the graph of co2 x year and logistic fitted co2 x year with prediction
plt.figure()
plt.plot(em_china["Year"], em_china[indicator], label=indicator)
plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel(indicator)
plt.legend()
plt.title('CO2 emissions (metric tons per capita) Vs. Year - China')
plt.show()