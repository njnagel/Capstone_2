import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
pd.options.display.max_colwidth = 50

chicagocrimes = pd.read_csv('data/Chicago_Crimes_2012_to_2017.csv')
chicagocrimes = chicagocrimes.dropna()
# fullcrimes = pd.read_csv('data/Crimes_-_2001_to_present.csv')
# fullcrimes = fullcrimes.dropna()
#crimesanal = fullcrimes[fullcrimes['Year'].isin(2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019)]



#Perform EDA

# chicagocrimes.head()
# chicagocrimes.info()
# chicagocrimes.columns


#chicagocrimes.isnull().sum(axis=0)


ptvaluesbyyear = chicagocrimes.groupby("Year").chicagocrimes['Primary Type'].value_counts() 
primtypearray = ptvalues.values 

# sns.lineplot(x="Primary Type", y="primtypearray[1]", hue="Year", data=chicagocrimes)
# plt.xticks(rotation=45, fontsize=6)
# plt.show()



def plot_eda_hist(measure, year):

    subchicagocrimes = fullcrimes[fullcrimes['Year'] == year]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.hist(subchicagocrimes[measure])
    plt.xticks(rotation = 45, fontsize = 6)
    #ax.set_xticklabels(measure.labels, rotation = 45)
    plt.title(measure)
    plt.title(year)
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
