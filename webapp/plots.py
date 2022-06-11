#Import packages
import pandas as pd
import math
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import io


#Plot stock trend
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Price', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    #convert to byte image
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image

#Boxplot stock trend and season
def boxplot(df):
    fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
    sns.boxplot(x='year', y='Adj Close', data=df, ax=axes[0])
    sns.boxplot(x='month', y='Adj Close', data=df.loc[~df.year.isin([2007, 2019]), :])
    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
    axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
    #convert to byte image       
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image

#Mavg plot
def mavg(df):
    fig, axes = plt.subplots(1, 1, figsize=(20,7), dpi= 80)
    #calculate the mean of last 50/200 data points
    close_px = df['Adj Close']
    mavg_50 = close_px.rolling(window=50).mean()
    mavg_200 = close_px.rolling(window=200).mean()
    
    mpl.rc('figure', figsize=(16, 8))
    mpl.__version__

    # Adjusting the style of matplotlib
    style.use('ggplot')

    close_px.plot(label='Stock price')
    mavg_50.plot(label='mavg_50')
    mavg_200.plot(label='mavg_200')
    plt.legend()
    
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image

