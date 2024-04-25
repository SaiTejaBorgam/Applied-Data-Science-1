# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from scipy.optimize import curve_fit
import scipy.stats as ss
from pathlib import Path
import warnings

#Supressing any warnings
warnings.filterwarnings("ignore")

# Function to preprocess data
def Data_preprocess(df):
    """
    Function to clean a data frame with required columns 
    """
    df = df.drop(['Country Name','Country Code','Series Code'],axis =1)
    df.set_index('Series Name', inplace=True)
    df= df.T
    df.index = df.index.str.split().str.get(0)
    df.columns.name = ''
    df = df.rename(columns=
        {"Electricity production from coal sources (% of total)":
            "Electricity from Coal %",
         "Energy imports, net (% of energy use)":
            "Energy Imports %",
         "Energy use (kg of oil equivalent per capita)":
            "Energy use (Kg)",
         "Fossil fuel energy consumption (% of total)":
            "Electricity from Fossil fuel %",
         "GDP growth (annual %)":
            "GDP growth %",
         "Population density (people per sq. km of land area)":
            "Population Density",
         "Access to electricity (% of population)":
            "Access to electricity %"})
    df.index = df.index.astype(int)
    return df


# Function to plot a Lineplot
def Line_Plot(*df):
    """
    Defining a function to create a Line plot 
    to identify the relation between Energy Imports across countries
    """
    plt.figure(figsize=(7, 5))
    cmap = ['red','blue','orange','green']
    for i, df in enumerate(df):
        sns.lineplot( data = df['Energy Imports %'], 
                     color = cmap[i],marker ='o',label = x[i])
        
    #Set the title and lables
    plt.title('Relation between Energy Imports across countries')
    plt.xlabel('Years')
    plt.ylabel('Energy Imports %')
    plt.grid()
    # Saving and displaying plot
    plt.savefig('Linegraph.png')
    plt.show()
    return


# Function to plot a Histogram
def Histogram_Plot(*df):
    """
    Defining a function to create a histogram 
    to understand the frequency of GDP growth %
    for different countries across the years
    """
    plt.figure(figsize=(7, 5))
    for i, df in enumerate(df):
        sns.histplot(df['GDP growth %'], kde=True, stat="density"
                     ,bins=10,linewidth=0, label=x[i],alpha=0.5)
    
    #Set the titles, legend, labels and grid 
    plt.title('Distribution of GDP Growth %')
    plt.xlabel('Annual GDP Growth %')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.legend()
    # Saving and displaying plot
    plt.savefig('histogram.png')
    plt.show()
    return


# Function to plot a boxplot
def Box_Plot(df):
    """
    Plotting a box plot to observe the stats of various factors
    """  
    plt.figure(figsize=(6, 4))
    df.boxplot(grid = False)
    # Set title, legend and labels
    plt.title('Box plot for various factors')
    plt.ylabel('Values')
    plt.xticks(rotation=25)
    #Saving and display the plot
    plt.savefig('BoxPlot.png')
    plt.show()
    return


# Function to get K value using Elbow method
def Plot_Elbow_Method(min_k, max_k, wcss, best_n):
    """
    Plots the elbow method for finding K value
    """
    fig, ax = plt.subplots(dpi=144)
    ax.plot(range(min_k, max_k + 1), wcss, 'kx-')
    ax.scatter(best_n, wcss[best_n-min_k], marker='o', 
               color='red', facecolors='none', s=50)
    ax.set_xlabel('k')
    ax.set_xlim(min_k, max_k)
    ax.set_ylabel('WCSS')
    plt.show()
    return


# Function to get silhoutte score
def one_silhoutte_inertia(n, xy):
    """ 
    Calculates the silhoutte score and WCSS for n clusters 
    """
    # set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=n, n_init=20)
    # Fit the data
    kmeans.fit(xy)
    labels = kmeans.labels_
    
    # calculate the silhoutte score
    score = silhouette_score(xy, labels)
    inertia = kmeans.inertia_

    return score, inertia


# Function to plot clustering
def Cluster_Plot(labels, xy, xkmeans, ykmeans, centre_labels):
    """
    Plots clustered data as a scatter plot with determined centres shown
    """
    colours = plt.cm.Set1(np.linspace(0, 1, len(np.unique(labels))))
    cmap = ListedColormap(colours)
    
    fig, ax = plt.subplots(dpi=144)
    #Plot the data with different colors for clusters
    s = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap,
                   marker='o', label='Data')

    ax.scatter(xkmeans, ykmeans, c=centre_labels, cmap=cmap,
               marker='x', s=100, label='Estimated Centres')

    cbar = fig.colorbar(s, ax=ax)
    cbar.set_ticks(np.unique(labels))
    ax.legend()
    ax.set_xlabel('Electricity from Coal %')
    ax.set_ylabel('Energy use (Kg)')
    plt.show()
    return


# Function to calculate logistic fit value
def Logistic_Fit(t, n0, g, t0):
    """
    Calculates the logistic function with scale factor n0 and growth rate g
    """
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f

#Reading dataset file
project_dir = Path.cwd()
path = project_dir/'Electricity_data.csv'
df = pd.read_csv(path)

df_Brazil = df[df['Country Name'].isin(['Brazil'])]
df_Denmark = df[df['Country Name'].isin(['Denmark'])]
df_Finland = df[df['Country Name'].isin(['Finland'])]
df_Ethiopia = df[df['Country Name'].isin(['Ethiopia'])]

#list to store the coutries names
x = ['Brazil','Denmark','Finland','Ethiopia']

df_Brazil = Data_preprocess(df_Brazil)
df_Denmark = Data_preprocess(df_Denmark)
df_Finland = Data_preprocess(df_Finland)
df_Ethiopia = Data_preprocess(df_Ethiopia)

df= pd.concat([df_Brazil,df_Denmark,df_Finland,df_Ethiopia])

#Plotting Histogram
Histogram_Plot(df_Brazil,df_Denmark,df_Finland,df_Ethiopia)

#Plotting Line graph
Line_Plot(df_Brazil,df_Denmark,df_Finland,df_Ethiopia)

#Plotting a Box PLot
Box_Plot(df[['Energy Imports %','Electricity from Fossil fuel %','Population Density']])

#Using describe function for mean, stanadrd deviation, min and max value.
print('Stats of the data', end='\n')
print(df.describe())

#Printing statistics of data
print('Skewness of the data', end='\n')
print(df.skew() , end='\n\n')

print('Kurtosis of the data', end='\n')
print(df.kurtosis() , end='\n\n')

print('Correlation of the data', end='\n')
print(df.corr() , end='\n\n')


#Clustering the Electricity from coal % and Energy use (Kg)
df_clust = df[['Electricity from Coal %','Energy use (Kg)']].copy()
scaler = RobustScaler()
norm = scaler.fit_transform(df_clust)
colours = plt.cm.Set1(np.linspace(0, 1, 5))
cmap = ListedColormap(colours)


#Finding the best number of CLusters using silhoutte method
wcss = []
best_n, best_score = None, -np.inf
for n in range(2, 11):
    score, inertia = one_silhoutte_inertia(n, norm)
    wcss.append(inertia)
    if score > best_score:
        best_n = n
        best_score = score
print(f"Best number of clusters = {best_n:2g}")

#Finding the best number of CLusters using elbow method
Plot_Elbow_Method(2, 10, wcss, best_n)

#PLotting Clustering
inv_norm = scaler.inverse_transform(norm)  
for k in range(3, 5):
    kmeans = KMeans(n_clusters=k, n_init=20)
    kmeans.fit(norm)     # fit done on x,y pairs
    labels = kmeans.labels_
    cen = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]
    cenlabels = kmeans.predict(kmeans.cluster_centers_)
    Cluster_Plot(labels, inv_norm, xkmeans, ykmeans, cenlabels)
    

#Dataframe for Ethotia's logistic fit
df_Ethiopia_fit = df_Ethiopia[['Access to electricity %']]
df_Ethiopia_fit.index = df_Ethiopia_fit.index.astype(int)

numeric_index = (df_Ethiopia_fit.index - 2003).values
p, cov = curve_fit(Logistic_Fit, numeric_index, df_Ethiopia_fit['Access to electricity %'],
                  p0=(1.2e12, 0.03, 10))
sigma = np.sqrt(np.diag(cov))

fig, ax = plt.subplots(dpi=144)
df_Ethiopia_fit= df_Ethiopia_fit.assign(Logistic_Fit = 
                                        Logistic_Fit(numeric_index, *p))
df_Ethiopia_fit.plot(ax=ax, ylabel='Access to Electricity %',xlabel='Years')

#Plotting logistic fit till the year 2050
numeric_index = (df_Ethiopia_fit.index - 2003).values
p, cov = curve_fit(Logistic_Fit, numeric_index, df_Ethiopia_fit['Access to electricity %'],
                  p0=(1.2e12, 0.03, 10))
Access_2050 = Logistic_Fit(2050 - 2003, *p)
sample_params = ss.multivariate_normal.rvs(mean=p, cov=cov, size=1000)
Access_unc_2050 = np.std(Logistic_Fit(2050 - 2003, *sample_params.T))
fig, ax = plt.subplots(dpi=144)
time_predictions = np.arange(1990, 2050, 1)
gdp_predictions = Logistic_Fit(time_predictions - 2003, *p)
gdp_uncertainties = [np.std(Logistic_Fit(future_time - 2003, *sample_params.T)
                            ) for future_time in time_predictions]

#Plotting the data along with the logistic fit and the uncertainities
ax.plot(df_Ethiopia_fit.index, df_Ethiopia_fit['Access to electricity %'],
        'b-', label='Data')
ax.plot(time_predictions, gdp_predictions, 'k-', label='Logistic Fit')
ax.fill_between(time_predictions, gdp_predictions - gdp_uncertainties,
                gdp_predictions + gdp_uncertainties, 
                color='gray', alpha=0.3)
ax.set_xlabel('Years')
ax.set_ylabel('Access to Electricity %')
ax.grid(alpha=0.5)
ax.legend()
plt.show()
print(f"Access to Electricity % in 2050: {Access_2050:g}")
print(f"Access to Electricity % in 2050: {Access_2050:g} +/- {Access_unc_2050:g}")
