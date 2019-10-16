import pandas as pd 
import csv
import random as rd 
from random import random
from random import sample
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
from numpy.linalg import svd

#pd.options.display.max_colwidth = 50



#chicagocrimes = pd.read_csv('data/Crimes_-_2001_to_present.csv')
#chicagocrimes.rename(columns={0: "ID", 1: "Case", 2: "Date", 3: "Block", 4: "IUCR", 5 : "PrimaryType", 6 : "Description", 7: "LocationDesc", 8: "Arrest", 9: "Domestic", 10: "Beat", 11: "District", 12: "Ward", 13: "CommunityArea", 14: "FBICode", 15: "Xcoord", 16: "Ycoord", 17: "Year", 18: "Updatedon", 19: "Lat", 20: "Long", 21: "Location"})

filename = "data/Crimes_-_2001_to_present.csv"
p = 0.20  # 20% of the lines

# keep the header, then take only 20% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
df = pd.read_csv(
         filename,
         header=0, 
         skiprows=lambda i: i>0 and rd.random() > p
)


chicagocrimes = df.dropna()

chicagocrimessub = chicagocrimes.where(chicagocrimes['Year'] >= 2010)


#convert t/f to binary
     
# chicagocrimessub['Domestic'] = chicagocrimessub['Domestic'].astype(int)
# chicagocrimessub['Arrest'] = chicagocrimessub['Arrest'].astype(int)


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

####Some plots
def plot_eda_hist(measure, year):

    subchicagocrimes = chicagocrimessub[chicagocrimessub['Year'] == year]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.hist(subchicagocrimes[measure])
    plt.xticks(rotation = 45, fontsize = 6)
    #ax.set_xticklabels(measure.labels, rotation = 45)
    plt.title(measure)
    plt.title(measuasre, year)
    return

def plot_eda_lines(measure, y):

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(chicagocrimes[measure], y)
    plt.xticks(rotation = 45, fontsize = 6)
    #ax.set_xticklabels(measure.labels, rotation = 45)
    plt.title(measure)
    #plt.title(year)
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



# fig,ax = plt.subplots()
# for primtype in primtypes:
#     ptdf2 = chicagocrimessub[chicagocrimessub['Primary Type'] == primtype]
#     y = ptdf2['Arrest'].value_counts(sort=False)
#     x = ptdf2['Arrest'].unique()
#     plt.xticks(rotation=45)
#     #x = np.argsort(x)
#     plt.title('Primary Type of Crime by Arrest')
#     ax.plot(x, y)


#fig, ax = plt.subplots()

# sns.lineplot(x="Primary Type", y="values", hue="Year", data=ptvaluesarray)
# ax.flatten()
# plt.xticks(rotation=45, fontsize=6)

######SVD
Xmatrix = chicagocrimessub.drop(['ID', 'Case Number', 'Block', 'Date', 'IUCR', 'Primary Type', 'Description', 'Location Description', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Updated On', 'Location', 'Historical Wards 2003-2015', 'Boundaries - ZIP Codes', 'Domestic'], axis = 1)
Xmatrix = Xmatrix.astype('float')

Xmatrixnona = Xmatrix.dropna()
U, sigma, VT = svd(Xmatrixnona, full_matrices=False, compute_uv=True)
# Check if U and VT matrices are column orthogonal
# If the matrix is orthogonal, it's dot product with itself should give
# the identity matrix (inline with itself (1), orthogonal with all the rest (0))

print("Checking U:")
print(np.dot(U.T,U).round(1))
print("\nChecking V:")
print(np.dot(VT.T,VT).round(1))

print("\nU matrix:")
print("Relates Users (rows) to latent features (columns) based on magnitude of values in matrix.\n")
print(U.round(2))

print("S matrix")
print(sigma.shape) # these are just the diagonal values of the singular values matrix
print("The latent feature singular values. The (singular value)^2 is\n"
      "is the eigenvalue.")
print("\nThe singular values:")
print(sigma.round(2)) # these are just the diagonal values of the singular values matrix
print("\nThe singular values matrix, S:")
sigma_m = sigma * np.eye(len(sigma))
print(sigma_m.round(2))

if __name__ == '__main__':  

    
    
    # plot_eda_hist('Arrest', 2010)
    # plot_eda_hist('Arrest', 2011)
    # plot_eda_hist('Arrest', 2012)
    # plot_eda_hist('Arrest', 2013)
    # plot_eda_hist('Arrest', 2014)
    # plot_eda_hist('Arrest', 2015)
    # plot_eda_hist('Arrest', 2016) 
    pass    