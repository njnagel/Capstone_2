import pandas as pd 
import csv
import random as rd 
from random import random
from random import sample
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import descartes
import geopandas as gpd 
from shapely.geometry import Point, Polygon
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.linalg import svd

#pd.options.display.max_colwidth = 50
##########pulling sample from large file and saving to csv
# filename = "data/Crimes_-_2001_to_present.csv"
# p = 0.20  # 20% of the lines
# # keep the header, then take only 20% of lines
# # if random from [0,1] interval is greater than 0.01 the row will be skipped
# df = pd.read_csv(
#          filename,
#          header=0, 
#          skiprows=lambda i: i>0 and rd.random() > p
# )

# df = df.dropna()
# df.to_csv('data/chicagocrimes.csv')

chicagocrimes = pd.read_csv('data/chicagocrimes.csv')


chicagocrimessub = chicagocrimes.where(chicagocrimes['Year'] >= 2010)
#pd.get_dummies(chicagocrimessub['District'])
####convert t/f to binary
     
# chicagocrimessub['Domestic'] = chicagocrimessub['Domestic'].astype(int)
# chicagocrimessub['Arrest'] = chicagocrimessub['Arrest'].astype(int)

#####prepare data matrix
Xmatrix = chicagocrimessub.drop(['ID', 'Case Number', 'Block', 'Date', 'IUCR', 'Primary Type', 'Description', 'Location Description', 'Police Districts', 'Police Beats','FBI Code', 'X Coordinate', 'Y Coordinate', 'Updated On', 'Location', 'Historical Wards 2003-2015', 'Boundaries - ZIP Codes'], axis = 1)
Xmatrix = Xmatrix.astype('float')
Xmatrixnona = Xmatrix.dropna()

######Perform EDA

#chicagocrimes.isnull().sum(axis=0)

# ptvaluesbyyear = chicagocrimessub.groupby("Year")['Primary Type'].value_counts() 
# ptvalues = chicagocrimessub['Primary Type'].value_counts()
# ptvaluesarray = np.array(ptvalues)

# ptdf = pd.DataFrame(ptvaluesbyyear)

years = chicagocrimessub['Year'].unique()
primtypes = chicagocrimessub['Primary Type'].unique()
# ##convert dates to a day of week field
datetodatetime = pd.to_datetime(chicagocrimessub['Date'])

chicagopops = {'2010': 2695598, '2011': 2707120, '2012': 2700800, '2013': 2710000, '2014': 2715000, '2015': 2723000, '2016': 2720546, '2017': 2722586, '2018': 2705994, '2019': 2695000}

weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
dayofweek = datetodatetime.dt.weekday
#dayofweekasstring = weekDays[dayofweek]

###########mapping
#street_map = gpd.read_file('data/geo_export_33ca7ae0-c469-46ed-84da-cc7587ccbfe6.shp')
# Xmatrixmap = Xmatrixnona.drop(['Ward', 'District', 'Beat', 'Community Area', 'Community Areas', 'Census Tracts', 'Zip Codes', 'Wards'], axis=1)
# Xmatrixmap2010=Xmatrixmap[Xmatrixmap['Year'] == 2010]
# Xmatrixmap2018=Xmatrixmap[Xmatrixmap['Year'] == 2018]
# crs = {"init": 'epsg:4326'}
# geometry = [Point(xy) for xy in zip(Xmatrixmap2010["Longitude"], Xmatrixmap2010["Latitude"])]
# geometry18 = [Point(xy) for xy in zip(Xmatrixmap2018["Longitude"], Xmatrixmap2018["Latitude"])]
# #Xmatrixmap = Xmatrixnona['Year','Arrest', 'Domestic', 'Latitude', 'Longitude']

# geo_df = gpd.GeoDataFrame(Xmatrixmap2010, crs=crs, geometry=geometry)
# geo_df18 = gpd.GeoDataFrame(Xmatrixmap2018, crs=crs, geometry = geometry18)


def geo_maps(df, measure, year):
    fig,ax = plt.subplots(figsize = (15, 15))
    street_map.plot(ax = ax, alpha = .4, color='grey')
    df[df[measure] == 0].plot(ax = ax, markersize = 20, color='blue', marker="o", label="Neg")
    df[df[measure] == 1].plot(ax=ax, markersize = 20, color = 'red', marker = "x", label='Pos')
    plt.title(measure, 'by Location', year)
    plt.legend(prop={'size': 15})
    plt.show()

#geo_maps(geo_df, Arrest, 2010)    


# fig,ax = plt.subplots(figsize = (15, 15))
# street_map.plot(ax = ax, alpha = .4, color='grey')
# geo_df[geo_df['Domestic'] == 1].plot(ax=ax, markersize = 20, color = 'red', marker = "x", label='DV')
# plt.title('Domestic Violence Calls by Location 2010')
# plt.legend(prop={'size': 15})
# plt.show()

# fig, ax = plt.subplots(figsize=(15,15))
# street_map.plot(ax = ax, alpha = .4, color='grey')
# geo_df18[geo_df18['Domestic'] == 1].plot(ax=ax, markersize = 20, color = 'red', marker = "x", label='DV')
# plt.title('Domestic Violence Calls by Location 2018')
# plt.legend(prop={'size': 15})
# plt.show()



####Some plots
def plot_eda_hist(measure, year):

    subchicagocrimes = chicagocrimessub[chicagocrimessub['Year'] == year]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.hist(subchicagocrimes[measure])
    plt.xticks(rotation = 45, fontsize = 6)
    #ax.set_xticklabels(measure.labels, rotation = 45)
    plt.title(measure)
    plt.title(measure, year)
    return

def plot_eda_lines(measure, y):

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(chicagocrimes[measure], y)
    plt.xticks(rotation = 45, fontsize = 6)
    ax.set_xticklabels(measure.labels, rotation = 45)
    plt.title(measure)
    plt.title(year)
    return

def plot_death_rates(data, measure):
    plt.rc('xtick', labelsize=6) 
    plt.rc('ytick', labelsize=6)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.barh(data['Location'], data[measure])
    plt.title(measure)
    return

# lists = sorted(chicagopops.items()) # sorted by key, return a list of tuples
# x, y = zip(*lists) # unpack a list of pairs into two tuples
# plt.plot(x, y)
# plt.xticks(fontsize=10)
# plt.title('City of Chicago Population by Year')
# plt.show()

fig,ax = plt.subplots(figsize = (15, 15))
ax.plot(arrestsbyyear.index,arrestsbyyear.values, color='red')
ax.plot(domesticbyyear.index,domesticbyyear.values, color = 'blue')
plt.title('Events by Year')

# fig,ax = plt.subplots()
# for primtype in primtypes:
#     ptdf2 = chicagocrimessub[chicagocrimessub['Primary Type'] == primtype]
#     y = ptdf2['Year'].value_counts(sort=False)
#     x = ptdf2['Year'].unique()
#     plt.xticks(rotation=45)
#     #x = np.argsort(x)
#     plt.title('Primary Type of Crime by Year')
#     ax.plot(x, y)
#     plt.legend()

# fig,ax = plt.subplots()
# for primtype in primtypes:
#     ptdf2 = chicagocrimessub[chicagocrimessub['Primary Type'] == primtype]
#     y = ptdf2['Arrest'].value_counts(sort=False)
#     x = ptdf2['Arrest'].unique()
#     plt.xticks(rotation=45)
#     #x = np.argsort(x)
#     plt.title('Primary Type of Crime by Arrest')
#     ax.plot(x, y)
#     plt.legend()


###########normalize the data
# beatmean=np.mean(Xmatrixnona['Beat'])
# beatstd=np.std(Xmatrixnona['Beat'])
# Xmatrixnona['beatnorm'] = (Xmatrixnona['Beat'] - beatmean)/beatstd
# districtmean=np.mean(Xmatrixnona['District'])
# districtstd=np.std(Xmatrixnona['District'])
# Xmatrixnona['districtnorm'] = (Xmatrixnona['District'] - districtmean)/districtstd
# wardmean=np.mean(Xmatrixnona['Ward'])
# wardstd=np.std(Xmatrixnona['Ward'])
# Xmatrixnona['wardnorm'] = (Xmatrixnona['Ward'] - wardmean)/wardstd
# Xmatrixnona['commareanorm'] = (Xmatrixnona['Community Area'] - np.mean(Xmatrixnona['Community Area']))/np.std(Xmatrixnona['Community Area'])
# Xmatrixnona['yearnorm'] = (Xmatrixnona['Year'] - np.mean(Xmatrixnona['Year']))/np.std(Xmatrixnona['Year'])
# Xmatrixnona['censusnorm'] = (Xmatrixnona['Census Tracts'] - np.mean(Xmatrixnona['Census Tracts']))/np.std(Xmatrixnona['Census Tracts'])
# Xmatrixnona['zipnorm'] = (Xmatrixnona['Zip Codes'] - np.mean(Xmatrixnona['Zip Codes']))/np.std(Xmatrixnona['Zip Codes'])
# Xmatrixnona['latnorm'] = (Xmatrixnona['Latitude'] - np.mean(Xmatrixnona['Latitude']))/np.std(Xmatrixnona['Latitude'])
# Xmatrixnona['longnorm'] = (Xmatrixnona['Longitude'] - np.mean(Xmatrixnona['Longitude']))/np.std(Xmatrixnona['Longitude'])

##########plot
##plot corr heatmap
# f = plt.figure(figsize=(19, 15))
# plt.matshow(Xmatrixnona.corr(), fignum=f.number)
# plt.xticks(range(Xmatrixnona.shape[1]), Xmatrixnona.columns, fontsize=14, rotation=45)
# plt.yticks(range(Xmatrixnona.shape[1]), Xmatrixnona.columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
#plt.title('Correlation Matrix', fontsize=16);

#pd.plotting.scatter_matrix(Xmatrixnona)
#plt.show()

######SVD

# U, sigma, VT = svd(Xmatrixnona, full_matrices=False, compute_uv=True)
# # Check if U and VT matrices are column orthogonal
# # If the matrix is orthogonal, it's dot product with itself should give
# # the identity matrix (inline with itself (1), orthogonal with all the rest (0))

# print("Checking U:")
# print(np.dot(U.T,U).round(1))
# print("\nChecking V:")
# print(np.dot(VT.T,VT).round(1))

# print("\nU matrix:")
# print("Relates Users (rows) to latent features (columns) based on magnitude of values in matrix.\n")
# print(U.round(2))

# print("S matrix")
# print(sigma.shape) # these are just the diagonal values of the singular values matrix
# print("The latent feature singular values. The (singular value)^2 is\n"
#       "is the eigenvalue.")
# print("\nThe singular values:")
# print(sigma.round(2)) # these are just the diagonal values of the singular values matrix
# print("\nThe singular values matrix, S:")
# sigma_m = sigma * np.eye(len(sigma))
# print(sigma_m.round(2))

###############VIF to reduce features
# vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(Xmatrixnona.to_numpy(), i) for i in range(Xmatrixnona.shape[1])]
# vif["features"] = Xmatrixnona.columns
# vif.round(1)

if __name__ == '__main__':  

    
    
    # plot_eda_hist('Arrest', 2010)
    # plot_eda_hist('Arrest', 2011)
    # plot_eda_hist('Arrest', 2012)
    # plot_eda_hist('Arrest', 2013)
    # plot_eda_hist('Arrest', 2014)
    # plot_eda_hist('Arrest', 2015)
    # plot_eda_hist('Arrest', 2016) 
    pass    