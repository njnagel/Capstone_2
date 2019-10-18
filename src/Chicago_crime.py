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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

pd.options.display.max_colwidth = 100
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
Xmatrix = chicagocrimessub.drop(['ID','Primary Type', 'Community Areas', 'Description', 'Case Number', 'Date', 'Block', 'IUCR', 'Location Description', 'Police Districts', 'Police Beats','FBI Code', 'X Coordinate', 'Y Coordinate', 'Updated On', 'Location', 'Wards', 'Historical Wards 2003-2015', 'Boundaries - ZIP Codes'], axis = 1)
#Xmatrix = Xmatrix.astype('float')
Xmatrixnona = Xmatrix.dropna()

######Perform EDA

#chicagocrimes.isnull().sum(axis=0)

#ptvaluesbyyear = chicagocrimessub.groupby("Year")['Primary Type'].value_counts() 
#ptvalues = chicagocrimessub.groupby('Primary Type')['Arrest'].value_counts()
#ptvaluesarray = np.array(ptvalues)

#ptdf = pd.DataFrame(ptvaluesbyyear)
#ptdf2 = pd.DataFrame(ptvalues)

years = chicagocrimessub['Year'].unique()
#primtypes = chicagocrimessub['Primary Type'].unique()

# ##convert dates to a day of week field
#datetodatetime = pd.to_datetime(chicagocrimessub['Date'])

chicagopops = {'2010': 2695598, '2011': 2707120, '2012': 2700800, '2013': 2710000, '2014': 2715000, '2015': 2723000, '2016': 2720546, '2017': 2722586, '2018': 2705994, '2019': 2695000}

weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
#dayofweek = datetodatetime.dt.weekday
#dayofweekasstring = weekDays[dayofweek]

###########mapping
# street_map = gpd.read_file('data/geo_export_33ca7ae0-c469-46ed-84da-cc7587ccbfe6.shp')
# Xmatrixmap = Xmatrixnona.drop(['Ward', 'District', 'Beat', 'Community Area', 'Community Areas', 'Census Tracts', 'Zip Codes'], axis=1)
# Xmatrixmap2010=Xmatrixmap[Xmatrixmap['Year'] == 2010]
# Xmatrixmap2018=Xmatrixmap[Xmatrixmap['Year'] == 2018]
# crs = {"init": 'epsg:4326'}
# geometry = [Point(xy) for xy in zip(Xmatrixmap2010["Longitude"], Xmatrixmap2010["Latitude"])]
# geometry18 = [Point(xy) for xy in zip(Xmatrixmap2018["Longitude"], Xmatrixmap2018["Latitude"])]
# #Xmatrixmap = Xmatrixnona['Year','Arrest', 'Domestic', 'Latitude', 'Longitude']
# geo_df = gpd.GeoDataFrame(Xmatrixmap2010, crs=crs, geometry=geometry)
# geo_df18 = gpd.GeoDataFrame(Xmatrixmap2018, crs=crs, geometry = geometry18)

# ptdfarrest = geo_df.groupby('Primary Type')['Arrest']
# ptarrestdf = pd.DataFrame(ptdfarrest)

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
# geo_df['Primary Type'].values.plot(ax=ax)
# plt.title('Calls by Primary Type 2010')
# plt.legend(prop={'size': 15})
# plt.show()

###geomap of DV calls 
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
    plt.title(year)
    return

def plot_eda_lines(measure, y):

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(chicagocrimes[measure], y)
    plt.xticks(rotation = 45, fontsize = 6)
    ax.set_xticklabels(measure.labels, rotation = 45)
    plt.title(measure)
    plt.title(year)
    return


##############Plot City pop by year
# lists = sorted(chicagopops.items()) # sorted by key, return a list of tuples
# x, y = zip(*lists) # unpack a list of pairs into two tuples
# plt.plot(x, y)
# plt.xticks(fontsize=10)
# plt.title('City of Chicago Population by Year')
# plt.show()
###########plot arrests and domestic events by year
a=(2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018)
b = (73758, 69947, 67009, 60808, 54582, 51359, 53436, 52733, 52470)
arrestsbyyear = Xmatrixnona.groupby('Year')['Arrest'].sum()
domesticbyyear = Xmatrixnona.groupby('Year')['Domestic'].sum()

fig,ax = plt.subplots(figsize = (15, 15))
ax.plot(a,b, color='green')
ax.plot(arrestsbyyear.index,arrestsbyyear.values, color='red')
ax.plot(domesticbyyear.index,domesticbyyear.values, color = 'blue')
plt.title('Calls, Arrests and Domestic Events by Year')

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

###############VIF to reduce features
#Xmatrixnona=Xmatrixnona.drop('Domestic')
# vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(Xmatrixnona.to_numpy(), i) for i in range(Xmatrixnona.shape[1])]
# vif["features"] = Xmatrixnona.columns
# vif.round(1)

##########logistic regression
X = Xmatrixnona.drop(['Domestic', 'Beat', 'District', 'Community Area', 'Year', 'Latitude', 'Longitude', 'Zip Codes'], axis=1)
y = Xmatrixnona['Domestic'] 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0, stratify = y) 

logistic_regression= LogisticRegression(class_weight='balanced')
model = logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)
(model)

coeffs = pd.DataFrame(zip(X.columns, model.coef_))
print('Coeffs:', coeffs)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
f, ax = plt.subplots(1, figsize=(10, 8))

sns.set(font_scale=2)
plt.title('Confusion Matrix for Logistic Regression')
sns.heatmap(confusion_matrix, annot=True, fmt='d', linewidths=.5)

('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
# results = confusion_matrix(y_test, y_pred) 
# ('Confusion Matrix :')
# (results)  
# ('Report : ')
# (classification_report(y_test, y_pred)) 

FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
Sensitivity = TPR/(FNR + TPR)
Accuracy = (TPR + TNR)/(TPR+TNR+FPR+FNR)

if __name__ == '__main__':  
    arrestsbyyear = Xmatrixnona.groupby('Year')['Arrest'].sum()
    domesticbyyear = Xmatrixnona.groupby('Year')['Domestic'].sum()
    
    
    # plot_eda_hist('Arrest', 2010)
    # plot_eda_hist('Arrest', 2011)
    # plot_eda_hist('Arrest', 2012)
    # plot_eda_hist('Arrest', 2013)
    # plot_eda_hist('Arrest', 2014)
    # plot_eda_hist('Arrest', 2015)
    # plot_eda_hist('Arrest', 2016) 
    pass    