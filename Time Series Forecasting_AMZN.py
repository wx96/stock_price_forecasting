#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Time-Series-Forecasting" data-toc-modified-id="Time-Series-Forecasting-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Time Series Forecasting</a></span><ul class="toc-item"><li><span><a href="#Visualizing-time-series-data" data-toc-modified-id="Visualizing-time-series-data-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Visualizing time series data</a></span><ul class="toc-item"><li><span><a href="#time-series-plot" data-toc-modified-id="time-series-plot-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>time series plot</a></span></li><li><span><a href="#2-side-view-plot-(to-visualize-pattern-reccurring-monthly/-annually)" data-toc-modified-id="2-side-view-plot-(to-visualize-pattern-reccurring-monthly/-annually)-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>2-side view plot (to visualize pattern reccurring monthly/ annually)</a></span></li><li><span><a href="#Seasonal-Plot-of-Time-Series" data-toc-modified-id="Seasonal-Plot-of-Time-Series-1.1.3"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Seasonal Plot of Time Series</a></span></li><li><span><a href="#Boxplot-of-Monthly-(seasonal)-and-Yearly-(trend)-distribution" data-toc-modified-id="Boxplot-of-Monthly-(seasonal)-and-Yearly-(trend)-distribution-1.1.4"><span class="toc-item-num">1.1.4&nbsp;&nbsp;</span>Boxplot of Monthly (seasonal) and Yearly (trend) distribution</a></span></li></ul></li><li><span><a href="#Patterns-in-a-time-series-[Sample-data]" data-toc-modified-id="Patterns-in-a-time-series-[Sample-data]-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Patterns in a time series [Sample data]</a></span><ul class="toc-item"><li><span><a href="#Decompose-a-time-series-into-its-components-[additive/-multiplicative]" data-toc-modified-id="Decompose-a-time-series-into-its-components-[additive/-multiplicative]-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Decompose a time series into its components [additive/ multiplicative]</a></span></li></ul></li><li><span><a href="#Stationary-Time-Series" data-toc-modified-id="Stationary-Time-Series-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Stationary Time Series</a></span><ul class="toc-item"><li><span><a href="#Why?" data-toc-modified-id="Why?-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Why?</a></span></li><li><span><a href="#How-to-test-for-stationary?" data-toc-modified-id="How-to-test-for-stationary?-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>How to test for stationary?</a></span></li><li><span><a href="#How-to-make-a-time-series-stationary?" data-toc-modified-id="How-to-make-a-time-series-stationary?-1.3.3"><span class="toc-item-num">1.3.3&nbsp;&nbsp;</span>How to make a time series stationary?</a></span></li></ul></li><li><span><a href="#Autocorrelation-and-partial-autocorrelation-functions" data-toc-modified-id="Autocorrelation-and-partial-autocorrelation-functions-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Autocorrelation and partial autocorrelation functions</a></span></li><li><span><a href="#Estimate-the-forecastability-of-a-time-series" data-toc-modified-id="Estimate-the-forecastability-of-a-time-series-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Estimate the forecastability of a time series</a></span></li><li><span><a href="#Smoothen-a-time-series" data-toc-modified-id="Smoothen-a-time-series-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Smoothen a time series</a></span><ul class="toc-item"><li><span><a href="#Why?" data-toc-modified-id="Why?-1.6.1"><span class="toc-item-num">1.6.1&nbsp;&nbsp;</span>Why?</a></span></li><li><span><a href="#How?" data-toc-modified-id="How?-1.6.2"><span class="toc-item-num">1.6.2&nbsp;&nbsp;</span>How?</a></span></li></ul></li><li><span><a href="#Granger-causality-test" data-toc-modified-id="Granger-causality-test-1.7"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Granger causality test</a></span></li><li><span><a href="#Moving-Average-/Rolling-Mean" data-toc-modified-id="Moving-Average-/Rolling-Mean-1.8"><span class="toc-item-num">1.8&nbsp;&nbsp;</span>Moving Average /Rolling Mean</a></span></li><li><span><a href="#Arima-(Auto-Regressive-Integrated-Moving-Average)" data-toc-modified-id="Arima-(Auto-Regressive-Integrated-Moving-Average)-1.9"><span class="toc-item-num">1.9&nbsp;&nbsp;</span>Arima (Auto Regressive Integrated Moving Average)</a></span><ul class="toc-item"><li><span><a href="#Order-of-differencing,-d" data-toc-modified-id="Order-of-differencing,-d-1.9.1"><span class="toc-item-num">1.9.1&nbsp;&nbsp;</span>Order of differencing, d</a></span></li><li><span><a href="#Order-of-the-auto-regressive-term,-p" data-toc-modified-id="Order-of-the-auto-regressive-term,-p-1.9.2"><span class="toc-item-num">1.9.2&nbsp;&nbsp;</span>Order of the auto regressive term, p</a></span></li><li><span><a href="#Moving-average-(MA)-term,-q" data-toc-modified-id="Moving-average-(MA)-term,-q-1.9.3"><span class="toc-item-num">1.9.3&nbsp;&nbsp;</span>Moving average (MA) term, q</a></span></li><li><span><a href="#Build-ARIMA-model" data-toc-modified-id="Build-ARIMA-model-1.9.4"><span class="toc-item-num">1.9.4&nbsp;&nbsp;</span>Build ARIMA model</a></span></li><li><span><a href="#Out-of-time-corss-validation-to-find-optimal-ARIMA-model" data-toc-modified-id="Out-of-time-corss-validation-to-find-optimal-ARIMA-model-1.9.5"><span class="toc-item-num">1.9.5&nbsp;&nbsp;</span>Out-of-time corss validation to find optimal ARIMA model</a></span></li><li><span><a href="#Accuracy-metrics-for-time-series-forecast" data-toc-modified-id="Accuracy-metrics-for-time-series-forecast-1.9.6"><span class="toc-item-num">1.9.6&nbsp;&nbsp;</span>Accuracy metrics for time series forecast</a></span></li><li><span><a href="#Auto-Arima-Forecast" data-toc-modified-id="Auto-Arima-Forecast-1.9.7"><span class="toc-item-num">1.9.7&nbsp;&nbsp;</span>Auto Arima Forecast</a></span></li><li><span><a href="#Automatically-build-SARIMA-model" data-toc-modified-id="Automatically-build-SARIMA-model-1.9.8"><span class="toc-item-num">1.9.8&nbsp;&nbsp;</span>Automatically build SARIMA model</a></span></li></ul></li><li><span><a href="#Stock-Price-Prediction" data-toc-modified-id="Stock-Price-Prediction-1.10"><span class="toc-item-num">1.10&nbsp;&nbsp;</span>Stock Price Prediction</a></span><ul class="toc-item"><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-1.10.1"><span class="toc-item-num">1.10.1&nbsp;&nbsp;</span>Feature Engineering</a></span></li><li><span><a href="#Pre-processing-&amp;-cross-validation" data-toc-modified-id="Pre-processing-&amp;-cross-validation-1.10.2"><span class="toc-item-num">1.10.2&nbsp;&nbsp;</span>Pre-processing &amp; cross-validation</a></span><ul class="toc-item"><li><span><a href="#Handling-missing-data" data-toc-modified-id="Handling-missing-data-1.10.2.1"><span class="toc-item-num">1.10.2.1&nbsp;&nbsp;</span>Handling missing data</a></span></li><li><span><a href="#Split-train-test-data" data-toc-modified-id="Split-train-test-data-1.10.2.2"><span class="toc-item-num">1.10.2.2&nbsp;&nbsp;</span>Split train-test data</a></span></li></ul></li><li><span><a href="#Simple-Linear-Analysis" data-toc-modified-id="Simple-Linear-Analysis-1.10.3"><span class="toc-item-num">1.10.3&nbsp;&nbsp;</span>Simple Linear Analysis</a></span></li><li><span><a href="#Quadratic-Regression" data-toc-modified-id="Quadratic-Regression-1.10.4"><span class="toc-item-num">1.10.4&nbsp;&nbsp;</span>Quadratic Regression</a></span></li></ul></li></ul></li><li><span><a href="#Web-Scrapping-HTML-Table" data-toc-modified-id="Web-Scrapping-HTML-Table-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Web Scrapping HTML Table</a></span><ul class="toc-item"><li><span><a href="#Get-table-content" data-toc-modified-id="Get-table-content-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Get table content</a></span></li><li><span><a href="#Parse-table-header" data-toc-modified-id="Parse-table-header-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Parse table header</a></span></li><li><span><a href="#Create-Pandas-Data-Frame" data-toc-modified-id="Create-Pandas-Data-Frame-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Create Pandas Data Frame</a></span></li></ul></li></ul></div>

# # Time Series Forecasting

# Types of time series forecasting: <br>
# I. Univariate time series forecasting <br>
# II. Multivariate time series forecasting
# 
# Ref: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

# In[1]:


#Import packages
import pandas as pd
import datetime
import pandas_datareader.data as web
import math
import numpy as np
import seaborn as sns

from sklearn import preprocessing

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl


# In[2]:


# #load data from lacal
# df= pd.read_csv('C:/Users/ooi.weixin/Documents/stock_prediction/data/AMZN.csv')
# df.head()

# #Set index as date
# df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
# df.index = df['Date']

# #plot historical prices trend
# plt.figure(figsize=(16,8))
# plt.plot(df['Close'], label='Close Price History')


# In[3]:


# load data from yahoo finance
start = datetime.datetime(2007, 1, 1)
end = datetime.date.today()

df = web.DataReader("AMZN", 'yahoo', start, end)
df.tail()
df.head()


# ## Visualizing time series data

# ### time series plot

# In[4]:


def plot_df(df, x, y, title="", xlabel='Date', ylabel='Price', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[5]:


plot_df(df, x=df.index, y=df['Adj Close'], title='Daily AMZN Stock Prices')   


# ### 2-side view plot (to visualize pattern reccurring monthly/ annually)

# In[6]:


x = df.index
y1 =df['Adj Close']

fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)
plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color='seagreen')
plt.ylim(-3000, 3000)
plt.title('Stock Price (Two Side View)', fontsize=16)
plt.hlines(y=0, xmin=np.min(x), xmax=np.max(x), linewidth=.5)
plt.show()


# ### Seasonal Plot of Time Series

# In[7]:


# Prepare data
df['year'] = [d.year for d in df.index]
df['month'] = [d.strftime('%b') for d in df.index]
years = df['year'].unique()

# Prepare Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)


# In[8]:


# Draw Plot
plt.figure(figsize=(20,12), dpi= 50)
for i, y in enumerate(years):
    if i > 0:
        plt.plot('month', 'Adj Close', data=df.loc[df.year==y, :], color=mycolors[i], label=y)
        plt.text(df.loc[df.year==y, :].shape[0]-.5, df.loc[df.year==y, 'Adj Close'][-1:].values[0], y, fontsize=10, color=mycolors[i], horizontalalignment='left') 
# Decoration
plt.gca().set(xlim=(-1.0, 12), ylim=(0, 2500), ylabel='$Price$', xlabel='$Month$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of DAYANG Stock Price Time Series", fontsize=20)
plt.show()


# ### Boxplot of Monthly (seasonal) and Yearly (trend) distribution

# In[9]:


# Prepare data
df['year'] = [d.year for d in df.index]
df['month'] = [d.strftime('%b') for d in df.index]
years = df['year'].unique()

# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='Adj Close', data=df, ax=axes[0])
sns.boxplot(x='month', y='Adj Close', data=df.loc[~df.year.isin([2007, 2019]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()


# ## Patterns in a time series [Sample data]

# = Base level + Trend + Seasonality + Error

# In[138]:


#Extra: difference of trend, sesonality and both from sample data
fig, axes = plt.subplots(1,3, figsize=(20,4), dpi=100)
pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/guinearice.csv', parse_dates=['date'], index_col='date').plot(title='Trend Only', legend=False, ax=axes[0])

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv', parse_dates=['date'], index_col='date').plot(title='Seasonality Only', legend=False, ax=axes[1])

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/AirPassengers.csv', parse_dates=['date'], index_col='date').plot(title='Trend and Seasonality', legend=False, ax=axes[2])

# additional: cyclic (not of fixed calendar based frequencies)


# ### Decompose a time series into its components [additive/ multiplicative]

# In[139]:


from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# Import Data
df_temp = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')

# Multiplicative Decomposition 
result_mul = seasonal_decompose(df_temp['value'], model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(df_temp['value'], model='additive', extrapolate_trend='freq')

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()

#residual in multiplicative looks random compared to additive; hence it is more preferred than additive


# In[12]:


# Extract the Components ----
# Actual Values = Product of (Seasonal * Trend * Resid)
df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
df_reconstructed.head()

# the multiplication of 3 components should be equal to the actual values


# ## Stationary Time Series

# ### Why?

# Much easy and more reliable
# 
# ARIMA is a linear regression model that uses its own lags as predictors. Linear regression models work best when the predictors are not correlated and are independent of each other.

# ### How to test for stationary?

# Options: <br>
# I. Look at the plot series. <br>
# II. Split the series into 2 or more contiguous parts and compute the summary stats (mean, variance, autocorrelation). For a stationaryseries, both stats should be similar. <br>
# III. Unit root tests (ADF test, KPSS test, PP test): determines how strongly a time series is defined by a trend

# In[10]:


# Summary stats

# 1. Histogram of the adjusted closing price
plt.hist(df['Adj Close'])
plt.show()

# 2. Split the data into two, and compare the stats summary between the two
split = round(len(df)/2)
X1, X2 = df['Adj Close'][0:split],df['Adj Close'][split:]

# 2.1 Mean
mean1, mean2 = X1.mean(),X2.mean()
print('mean1=%f, mean2=%f' % (mean1, mean2))

# 2.2 Variance
var1, var2 = X1.var(), X2.var()
print('variance1=%f, variance2=%f' % (var1, var2))


# ### How to make a time series stationary?

# Options: <br>
# I. Differencing the series (substract the previous value from the current value), d =>refer ARIMA model below <br>
# II. Take the log of the series <br>
# III. Take the nth root of the series <br>

# In[11]:


# Log of the series
plt.hist(np.log(df['Adj Close']))
plt.show()


# In[12]:


# Check the stats summary of the log series
# 2. Split the data into two, and compare the stats summary between the two
split = round(len(df)/2)
X = np.log(df['Adj Close'])
X1, X2 = X[0:split],X[split:]

# 2.1 Mean
mean1, mean2 = X1.mean(),X2.mean()
print('mean1=%f, mean2=%f' % (mean1, mean2))

#2.2 Variance
var1, var2 = X1.var(), X2.var()
print('variance1=%f, variance2=%f' % (var1, var2))


# ## Autocorrelation and partial autocorrelation functions

# Autocorrelation is simply the correlation of a series with its own lags. If a series is significantly autocorrelated, that means, the previous values of the series (lags) may be helpful in predicting the current value. <br>
# Lag scatter plot to check auto correlation (x:y variable; y: lag of y variable)

# Partial Autocorrelation also conveys similar information but it conveys the pure correlation of a series and its lag, excluding the correlation contributions from the intermediate lags.

# ## Estimate the forecastability of a time series

# Options: <br>
# I. Approximate entropy (higher, harder to forecast) <br>
# II. Sample entropy

# ## Smoothen a time series

# ### Why?

# 1. Reducing the effect of noise in a signal get a fair approximation of the noise-filtered series. <br>
# 2. The smoothed version of series can be used as a feature to explain the original series itself. <br>
# 3. Visualize the underlying trend better

# ### How?

# Options: <br>
# I. Take a moving average (large window-width might oversmooth the series) <br>
# II. Do a LOESS smoothing (Localized Regression) <br>
# III. Do a LOWESS smoothing (Locally Weighted Regression)

# ## Granger causality test 

# to determine if one time series will be useful to forecast another

# ## Moving Average /Rolling Mean

# to determine trend

# In[13]:


#calculate the mean of last 100 days data points
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()


# In[14]:


# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(16, 8))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()


# ## Arima (Auto Regressive Integrated Moving Average)

# a linear regression model that uses its own lags as predictors
# 
# <b>p</b> is the order of the Auto Regressive (AR) term = # of lags of Y to be used as predictor <br>
# <b>d</b> is the number of differencing required to make the time series stationary <br>
# <b>q</b> is the order of the Moving Average (MA) term = size of moving average window (lagged forecast errors) <br>
# 

# ![arima.PNG](attachment:arima.PNG)

# ### Order of differencing, d
# 
# To make the time series stationary. <br>
# The observation in a stationary time series are not dependent on time.(No trend, no seasonal effects) <br>
# 
# The right order of differencing is the minimum differencing required to get a near-stationary series which roams around a defined mean and the <b>ACF plot reaches to zero fairly quick. </b> <br>
# If the autocorrelations are <b>positive</b> for many number of lags (10 or more), then the series needs further differencing. <br>
# On the other hand, if the lag 1 autocorrelation itself is too <b>negative</b>, then the series is probably over-differenced.
# In the event, you can’t really decide between two orders of differencing, then go with the order that gives the <b>least standard deviation</b> in the differenced series.

# <b>Augmented Dickey Fuller test </b> [to check if the series is stationary] <br>
# <u>Null hypothesis (H0)</u>: the time series is non-stationary/ can be represented by a unit root (has some time dependent structure) <br>
# <u>Alternate hypothesis (H1)</u>: the time series is stationary/ does not have a unit root (no time-dependent structure) <br>
# 
# *The more negative the ADF statistic, the more likely to reject null hypothesis (the series is stationary)
# *If p-value of the test is less than the significance level (0.05), then reject the null hypothesis and infer that time series is indeed stationary

# In[15]:


# Original raw data
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Adj Close'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Number of lags used: %i' % result[2] )
print('Critical Values to determine ADF statistic:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[16]:


# log data ( attempt to make the series stationary)
from numpy import log
result = adfuller(log(df['Adj Close']).dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Number of lags used: %i' % result[2] )
print('Critical values to determine ADF statistic:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# Since p-value > significance level of 0.05 in both cases; hence the null hypothesis is failed to be rejected, where the series is not stationary. <br>
# 
# The ADF statistics in both cases are greater than the critical values, indicating the series are non-stationary

# In[17]:


# Autcorrelation plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(12,10), 'figure.dpi':120})

# reset in index as date as index will cause error
df1 = df.reset_index()

# Original Series
fig, axes = plt.subplots(4, 2, sharex=True)
axes[0, 0].plot(df1['Adj Close']); axes[0, 0].set_title('Original Series')
plot_acf(df1['Adj Close'], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df1['Adj Close'].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df1['Adj Close'].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df1['Adj Close'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df1['Adj Close'].diff().diff().dropna(), ax=axes[2, 1])

# 3rd Differencing
axes[3, 0].plot(df1['Adj Close'].diff().diff().diff()); axes[3, 0].set_title('3rd Order Differencing')
plot_acf(df1['Adj Close'].diff().diff().diff().dropna(), ax=axes[3, 1])

plt.show()


# The x-axis of autocorrelation chart is the number of lag; y-axis is the correlation value.
# 
# Not much improvements in 2nd order and 3rd order differencing compared to 1st order differencing. And the autocorrelation plot for 2nd differencing goes into the far negative zone fairly quickly, indicates the series might have been over difference. <b>Hence, the order of differencing is set to be 1 </b>

# <b> ndiff() </b> <br>
# Function to estimate the number of differences to make a given time series stationary

# In[18]:


from pmdarima.arima.utils import ndiffs
y = df['Adj Close']

## Adf Test
print("ADF",ndiffs(y, test='adf'))

# KPSS test
print("KPSS",ndiffs(y, test='kpss'))

# PP test:
print("PP",ndiffs(y, test='pp'))


# ### Order of the auto regressive term, p
# 
# The autoregression (AR) method models the next step in the sequence as a linear function of the observations at prior time steps. The method is suitable for univariate time series without trend and seasonal components.
# 
# By inspecting <b>Partial Autocorrelation (PACF) plot</b> to investigate the pure correlation between the series and its lag (after excluding the contributons from the intermediate lags)

# ![arima_AR.PNG](attachment:arima_AR.PNG) 

# In[19]:


# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df1['Adj Close'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,10))
plot_pacf(df1['Adj Close'].diff().dropna(), ax=axes[1])

plt.show()


# The x-axis of partial autocorrelation chart is the number of lag; y-axis is the correlation value.

# ### Moving average (MA) term, q
# 
# The error of the lagged forecast. <br>
# Can inspect the ACF plot for the number of MA terms. <br>
# <b>ACF tells how many MA terms are required to remove any autocorrelation in stationarized series. </b>

# ![arima_MA.PNG](attachment:arima_MA.PNG)

# In[20]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df1['Adj Close'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df1['Adj Close'].diff().dropna(), ax=axes[1])

plt.show()


# <b> If a series is slightly under-differenced, add one/ more additional AR terms. <br>
# If a series is slightly over-differenced, add an additional MA term. </b>

# ### Build ARIMA model
# 
# ARIMA(p,d,q)

# In[21]:


from statsmodels.tsa.arima_model import ARIMA

# 1,1,0 ARIMA Model
model = ARIMA(df['Adj Close'], order=(1,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# <b>AIC (Akaike Information Criteria) </b> <br>
# Quantifies 1) the goodness of fit, and 2) the simplicity/parsimony, of the model into a single statistic. <br>
# When comparing two models, the one with the lower AIC is generally “better”. 

# ![AIC.PNG](attachment:AIC.PNG)

# <b>Bayesian Information Criterion (BIC) </b> <br>
# BIC penalizes complex models more strongly than the AIC.

# ![BIC.PNG](attachment:BIC.PNG)

# In[22]:


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[23]:


# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()


# ### Out-of-time corss validation to find optimal ARIMA model
# 
# Take few steps back in time and forecast into the future to as many steps you took back. Then compare the forecast against the actuals.
# 
# <b> Why? </b> <br>
# To validate the prediction performance.
# 
# <b>How? </b> <br>
# Create the training and testing dataset by splitting the time series into 2 contiguous parts in approximately 75:25 ratio or a reasonable proportion based on time frequency of series.

# In[24]:


from statsmodels.tsa.stattools import acf

train_proportion = int(np.ceil(len(df)*0.75))

# Create Training and Test
train = df['Adj Close'][:train_proportion]
test = df['Adj Close'][train_proportion:]

len(test)


# In[31]:


# Build Model
model = ARIMA(train, order=(3, 2, 1))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(len(test), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# ### Accuracy metrics for time series forecast
# 
# <b> Mean Absolute Percentage Error (MAPE) </b> <br>
# Mean Error (ME) <br>
# Mean Absolute Error (MAE) <br>
# Mean Percentage Error (MPE) <br>
# Root Mean Squared Error (RMSE) <br>
# Lag 1 Autocorrelation of Error (ACF1) <br>
# <b> Correlation between the Actual and the Forecast (corr) </b> <br>
# <b> Min-Max Error (minmax) </b> <br>
# 
# *highlighted metrics are preferred in comparing performance among multiple models as they are percentage errors that vary between 0 and 1; hence easire to judge the forecast performance irrespective of the scale of the series.

# In[32]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)


# 15% MAPE implies the model is around 85% accurate in predidcting the next 25% observations

# ### Auto Arima Forecast

# In[33]:


from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

model = pm.auto_arima(df['Adj Close'], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# In[34]:


# Interpret residual plot in ARIMA
model.plot_diagnostics(figsize=(7,7))
plt.show()


# In[35]:


# forecast
n_periods = 817
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df['Adj Close']), len(df['Adj Close'])+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
# reset in index as date as index will cause error
df_noindex = df.reset_index()
plt.plot(df_noindex['Adj Close'])
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()


# # Web Scrapping HTML Table

# https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059

# Python packages: scrapy; BeautifulSoup

# In[36]:


# import packages
import requests
import lxml.html as lh
import pandas as pd


# ## Get table content

# In[37]:


url = 'https://finance.yahoo.com/quote/5176.KL/history?p=5176.KL'

# Create a handle, page, to handle the contents of the website
page = requests.get(url)

# Store the contents of the website under doc
doc = lh.fromstring(page.content)

# Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')


# In[38]:


# Check the length of the first 10 rows; should be of the same length
[len(T) for T in tr_elements[:10]]


# In[39]:


tr_elements


# ## Parse table header

# In[40]:


#Create empty list
col=[]
i=0

#For each row, store each first element (header) and an empty list
for t in tr_elements[0]:
    i+=1
    name=t.text_content()
    print ('%d:"%s"'%(i,name))
    col.append((name,[]))


# ## Create Pandas Data Frame

# In[41]:


for j in range(1,len(tr_elements)):
    #T is our j'th row
    T=tr_elements[j]
    
    #If row is not of size 10, the //tr data is not from our table 
    if len(T)!=7:
        break
    
    #i is the index of our column
    i=0
    
    #Iterate through each element of the row
    for t in T.iterchildren():
        data=t.text_content() 
        #Check if row is empty
        if i>0:
        #Convert any numerical value to integers
            try:
                data=int(data)
            except:
                pass
        #Append the data to the empty list of the i'th column
        col[i][1].append(data)
        #Increment i for the next column
        i+=1


# In[42]:


# Check the length of each column; all column should have the same # of rows
[len(C) for (title,C) in col]


# In[43]:


#
Dict={title:column for (title,column) in col}
df=pd.DataFrame(Dict)


# In[44]:


df.head()

