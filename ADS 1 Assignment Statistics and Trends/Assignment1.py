# Importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Function to preprocess data
def pre_process(df):
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Low'] = df['Low'].str[1:]
    df = df.astype({"Low": float})
    df['Year'] = df['Date'].dt.year
    df1 = df[['Year', 'Low']]
    df1 = df1.groupby(['Year']).mean()
    return df1


# Function to plot a Lineplot
def line_plot(df):
    """
    Creating a line plot to visualize trends over the years for companies 
    """
    # Plot a line graph
    df.plot(kind='line', figsize=(6, 4), marker='o')

    # Set title, legend, and labels
    plt.title('Share prices of the companies')
    plt.xlabel('Years')
    plt.ylabel('Price')
    plt.legend()

    # Saving and displaying the plot
    plt.savefig('LinePlot.png')
    plt.show()
    return


# Function to plot a Histogram
def histogram_plot(*dfs):
    """
    Creating a histogram to observe the frequency of 
    share prices across the years
    """
    # Variable to store labels
    x = ['Apple', 'Microsoft']
    
    plt.figure(figsize=(6, 4))
    
    # Plot an overlapped histogram to observe the frequency of prices
    for i, df in enumerate(dfs):
        sns.histplot(df, kde=True, stat="density", bins=10,
                     linewidth=0, label=x[i], alpha=0.5)
    
    # Set title, legend, and labels
    plt.title('Distribution of Apple and Microsoft share prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.legend()
    
    # Saving and displaying the plot
    plt.savefig('HistogramPlot.png')
    plt.show()
    return


# Function to plot a Boxplot
def box_plot(df):
    """
    Creating a box plot to observe the stats of the
    share prices between the companies
    """
    
    plt.figure(figsize=(6, 4))
    
    # Plot a box plot
    df.boxplot(grid=False)
    
    # Set title, legend, and labels
    plt.title('Box plot of share prices')
    plt.xlabel('Companies')
    plt.ylabel('Price')
    
    # Saving and displaying the plot
    plt.savefig('BoxPlot.png')
    plt.show()
    return

# Reading the data from the files
df_sony= pd.read_csv('C:/Users/saite/Sony.csv')
df_apple= pd.read_csv('C:/Users/saite/Apple.csv')
df_oracle= pd.read_csv('C:/Users/saite/Oracle.csv')
df_MS= pd.read_csv('C:/Users/saite/Microsoft.csv')

# Cleaning the data
df_sony = pre_process(df_sony)
df_apple = pre_process(df_apple)
df_oracle = pre_process(df_oracle)
df_MS = pre_process(df_MS)

# Renaming the columns
df_sony = df_sony.rename(columns={"Low": "Sony"})
df_apple = df_apple.rename(columns={"Low": "Apple"})
df_oracle = df_oracle.rename(columns={"Low": "Oracle"})
df_MS = df_MS.rename(columns={"Low": "Microsoft"})

# Merging the data 
df = df_sony.join(df_apple["Apple"])
df = df.join(df_MS["Microsoft"])
df = df.join(df_oracle["Oracle"])

# Printing statistics of the data
print('Basic statistics of Company Prices', end='\n')
print(df.describe(), end='\n\n')

print('Skewness of Company Prices', end='\n')
print(df.skew(), end='\n\n')

print('Kurtosis of Company Prices', end='\n')
print(df.kurtosis(), end='\n\n')

print('Correlation of Company Prices', end='\n')
print(df.corr(), end='\n\n')

# To show the Linegraph
line_plot(df)

# To show the Histogram
histogram_plot(df['Apple'], df['Microsoft'])

# To show the Boxplot
box_plot(df)