

![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/CPDimage.png)




# Chicago Crimes 2010 - 2018

The Chicago Police Department (CPD) is the law enforcement agency of the U.S. city of Chicago, Illinois, under the jurisdiction of the City Council. The Police Superintendent serves as an apointee of the mayor. It is the second-largest municipal police department in the United States. It has approximately 13,500 officers. There are 22 police districts, 50 wards, 77 Community Areas, and 285 beats. There is considerable overlap in the geographic designations, though no two classifications are 100% nested.

In recent years, the United States Department of Justice has criticized the department for its poor training, lack of oversight and routine use of excessive force. The Wiki page provides a concise list of controversial events related to the CPD.


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/CPDcont.png)





2017 Department of Justice report and agreement for enforcement

Following the McDonald shooting, Illinois State Attorney General Lisa Madigan requested that the US Department of Justice conduct a civil rights investigation of the Chicago Police Department. They released their report in January 2017, announcing an agreement with the city to work on improvements under court supervision. They strongly criticized the police for a culture of excessive violence, especially against minority suspects and the community, and said there was insufficient and poor training, and lack of true oversight.

# Project Purpose

The intent of this project is to look at Chicago crime data and assess whether there are abherent patterns that may indicate inconsistent policing over time, crime or location.


# Dataset

The dataset was obtained from Kaggle.com. This dataset reflects reported incidents of crime (with the exception of murders where data exists for each victim) that occurred in the City of Chicago from 2010 to 2018. Data is extracted from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system. In order to protect the privacy of crime victims, addresses are shown at the block level only and specific locations are not identified. Should you have questions about this dataset, you may contact the Research & Development Division of the Chicago Police Department at 312.745.6071 or RandD@chicagopolice.org. 

Disclaimer: These crimes may be based upon preliminary information supplied to the Police Department by the reporting parties that have not been verified. The preliminary crime classifications may be changed at a later date based upon additional investigation and there is always the possibility of mechanical or human error. Therefore, the Chicago Police Department does not guarantee (either expressed or implied) the accuracy, completeness, timeliness, or correct sequencing of the information and the information should not be used for comparison purposes over time. The Chicago Police Department will not be responsible for any error or omission, or for the use of, or the results obtained from the use of this information.

# Data Processing

The columns in the original dataset included: CaseNumber, ID, Date, Block, IUCR, Primary Type (of Crime), Crime Description, Location Description, Beat, Ward, District, Community Area, Arrest (T/F), Domestic (T/F), X Coordinate, Y Coordinate, Latitude, Longitude, and Location. 

The original data consisted of reports from 2001 to present, over 6 million records. A 20% random sample of the data was read into a Pandas dataframe, then into a csv file in local storage for subsequent use.

Columns that were of no analytic utility were dropped.

Analysis was completed in Pandas and Matplotlib. Mapping was done using GeoPandas.


# EDA


Chicago crimes summarized by Primary Types for 2010, 2012, and 2018 are presented below:

2010 Crimes by Primary Type

Highest occurring crimes are Battery, Other Offence, Weapons, Theft, Criminal Damage, Narcotics


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/PrimaryType2010.png)


2012 Crimes by Primary Type

Highest include Burglary, Theft, Disturbing the Peace, Assault, Narcotics


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/PrimaryType2012.png)


2018 Crimes by Primary Type

Most common were Criminal Damage, Battery, Theft, Criminal Trespass, Weapons, Narcotics


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/PrimaryType2018.png)


# Events Over Time

Calls, Arrests and Domestic Violence calls are summarized over the time period in the chart below. Note that events from 2019 are included, althouogh the reporting year is still in process.


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/ArrestsDomesticbyYear.png)


Note that while calls have declined, arrests have declined as well over the years. DV calls have decreased only slightly. We can see from the plot of Chicago population that the numbers of residents have remained essentially the same from 2010 to 2018.

![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/Chicago%20Pop%20by%20Year.png)

A key question is whether the severity of crimes has changed over the years.  We can see that the nature of the crimes have changed in the first graphs.

# Events by Location

We can look for patterns related to location as well.  The map of Calls as a function of whether there was an arrest are below for the years of 2010 and 2018.


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/Arrests2010map.png)



![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/Arrests2018map.png)


Domestic Violence events can be seen for the two years below.


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/DV%20Calls%202010map.png)



![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/DV%20Calls%202018map.png)



# Modelling


The goal was to predict DV based on a predictive logistic regression.

  # Feature Reduction
  
  
  
To determine extent of relationships and identify redundant features, a Variance Inflation Factors analysis was done.  Any feature with a factor greater than 10 was removed.  This resulted in the final X matrix features to be Arrest, Ward, and Census Tract.  


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/VIFresults.png)


# Logistic Regression

Coefficients of Model - very small

Coefficients correspond to intercept, Arrest, Ward, and Census Tract

[3.417290285208584e-08, -0.0003784134375290321, -0.009620818975723106, 0.00041282222807018017]


The confusion matrix on the test set shows poor performance.


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/confmatrixreg.png)


The Accuracy score of .566 reflects the poor fit.

We are concerned with sensitivity because we would like to focus on false negatives (where we expect to see DV events and there are none). The Sensitivity (TP/(TP + FN)) also reflects a poor performance at .48.


# Next Steps

1. Answer the question - do arrest rates differ for DV calls by geographic area?

2. Hot Encode features and rerun analysis

3. Supplement original dataset with population counts to get rates 

4. Add demographics by Census Tract to relate to DV calls and Arrests

5. Incorporate pollution data to relate to violent crimes (Capstone 4)


![alt text](https://github.com/njnagel/Capstone_2/blob/master/img/dataguy.png)





















