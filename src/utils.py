#!/usr/bin/env python
# coding: utf-8

# # utils
# This is the utils file for a set of scripts that produce hourly forecasts for Day-Ahead (DA) and Real-Time (RT) Local Marginal Prices (LMP) of electrical generation at the North Hub Bus.
# this file contains a collection of custom functions used in the "ERCOT_electricity_price_forecast" project
# 
# December 10, 2024
# Matt Chmielewski
# https://github.com/emskiphoto
#
# marginal_price_forecasting
# ERCOT_electricity_price_forecast

# import dependencies
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot, moment


def cols_w_lt_n_unique_vals(df, n=3):
    """Returns index of only column names whose
    series has no more than 'n' unique values.  This can 
    be used to identify columns suited for boolean dtype."""
    return df.columns[df.apply(pd.Series.nunique) <= n-1]


def region_from_feature(x):
    """Returns list of strings from x where the second term
    (ex. 'WZ_SouthCentral (RTLOAD)'  --> SouthCentral) is extracted
    from strings matching the example format.  If no value is found to
    be extracted, 'None' is returned"""
    return list([term.split("_")[1].split(" ")[0]\
            .lower().strip() 
            if len(term.split("_")) != 1
           else None for term in x])


def colors_from_dict(d, features, color_missing = 'grey'):
    """Returns list of RGBA colors of equal length to input features list based on
    feature to color assignments in input dictionary d"""
    rgba_missing = mpl.colors.to_rgba(color_missing)
    return [d[col] if col in d else rgba_missing for col in features]


def colors_from_colormap(n, colormap, alpha=1):
    """Returns list of 'n' RGBA tuples that each define a single color based on the 
    matplotlib 'colormap' object (ie., 'plt.cm.Blues') provided
    Example Usage:
    colors_from_colormap(3, plt.cm.Blues, alpha=0.5)"""
    return [colormap(i/n, alpha=alpha) for i in range(1,n+1)]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''Returns reduced/truncated version of input 'cmap' based on 
    minval/maxval inputs. 
    Example Usage:  plt.get_cmap('Blues')'''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def visualize_color_by_dictionary(d, figsize=(6, 12), title=''):
    """Returns plot of stack of bars with labels from keys in dictionary 'd', colored by
    RGBA color tuple in values of d."""
    fig, ax = plt.subplots(figsize = figsize)
    for idx, (f, c) in zip(range(len(d.keys())), d.items()):
    #     print(idx, f, c)
        ax.barh(idx, 1, color = c, height=1, align='edge')
        ax.text(0.5, idx, f, ha='center', va = 'bottom')
    ax.set_axis_off()
    plt.title(title)
    plt.show()

    
def min_max_std_symmetric(x, n_std, quantile=0.95):
    """Returns minimum and maximum values of series 'x' clipped to 
    'quantile' defined as +/- the
    standard deviation multiplied by n_std relative to mean.  Minimum
    and maximum values will have the same absolute distance from the mean """
    # clip outliers to quantile
    q_min = np.quantile(x, 1-quantile)
    q_max = np.quantile(x, quantile)
    x = np.clip(x, q_min, q_max)
    std_ = np.std(x) * n_std
    mean_ = np.mean(x)
    min_ = mean_ - std_
    max_ = mean_ + std_
    return (min_, max_)
    
    
def min_max_std_asymmetric(x, n_std, quantile=0.95):
    """Returns minimum and maximum values of series 'x' clipped to 
    'quantile' defined as the mean of the data plus 'n_std' multiplied by the standard
    deviation of the top or bottom (50th percentile) quantile values standard deviation
    multiplied by n_std relative to mean. Minimum
    and maximum values are likely to have absolute distance from the mean that are different (asymmetric)"""
    # clip outliers to quantile
    q_min = np.quantile(x, 1-quantile)
    q_max = np.quantile(x, quantile)
    x = np.clip(x, q_min, q_max)
    # split x by median value to create x_upper and x_lower   
    median_ = np.median(x)
    mean_ = np.mean(x)
    x_upper = x[x>median_]
    x_lower = x[x<median_]
    #      
    std_upper = np.std(x_upper) * n_std
    std_lower = np.std(x_lower) * n_std
#     std_ = np.std(x) * n_std
    min_ = mean_ - std_lower
    max_ = mean_ + std_upper
    return (min_, max_)


def periodic_spectral_density(x):
    """Return df containing power spectral density by frequency"""
    sampling_freq =  1 / pd.Timedelta(x.index.freq).seconds
    freqstr = x.index.freqstr
    df_period = pd.DataFrame(
            index= np.round(sampling_freq / periodogram(x.dropna(),
                               fs=sampling_freq)[0], 0),
             data = {'Power spectral density': periodogram(x.dropna(),
                   fs=sampling_freq)[1]}).rename_axis(freqstr)
    return df_period.sort_index()


def periodicity_hourly(x, n=3):
    """Returns top 'n' intervals where spectral density peaks occur to help
    identify periods in cyclical data"""
#     hourly sampling frequency
    sampling_freq = 0.0002777777777777778  
#     alternatively (not used)
#     sampling_freq =  1 / pd.Timedelta(x.index.freq).seconds
#     freqstr = x.index.freqstr
    temp = pd.DataFrame(
            index= np.round(sampling_freq / periodogram(x.dropna(),
                               fs=sampling_freq)[0], 0),
             data = {'Power spectral density': periodogram(x.dropna(),
                                           fs=sampling_freq)[1]}).rename_axis('Hours')
    return temp.nlargest(n, 'Power spectral density').index.tolist()


# Linear regression outputs:
# refine this step so that the linregress calculation does
# not have to be repeated for each function
# problem is that pd.Series.rolling only accepts functions that return a single value, not a tuple
def slope(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
    return slope

def intercept(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
    return intercept

def p_value(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
    return p_value


def clip_inf_values_to_quantile(x, q_min = 0.001, q_max=0.999):
    """Returns pandas series 'x' from input pandas series with any
    infinite values clipped to
    the quantiles set in q_min and q_max (ONLY if infinite values are present).  
    Consider q_min and q_max values that perserve the non-infinite
    minimum and maximum values intact"""
    if x.eq(-np.inf).any() or x.eq(np.inf).any():
#         print('Series has inf values')
    #     identify quantile values for clipping by removing inf and NaN
        x_min, x_max = x.replace(-np.inf, np.NaN).replace(np.inf, np.NaN)\
                        .dropna().quantile([q_min, q_max])
#         print(x_min, x_max)
        return x.clip(x_min, x_max)
    else:
        return x
    
    
def plot_correlogram(x, lags=None, title=None, color='black'):
    """Plots 2x2 array of Residuals, Probability Plot, Autocorrelation
    and Partial Auto-Correlation"""
    lags = min(10, int(len(x)/5)) if lags is None else lags
    with sns.axes_style('whitegrid'):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        x.plot(ax=axes[0][0], title='Residuals', color=color)
        x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
        q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
        stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
        axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
        probplot(x, plot=axes[0][1])
        mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
        s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
        axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
        plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
        plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
        axes[1][0].set_xlabel('Lag')
        axes[1][1].set_xlabel('Lag')
        fig.suptitle(title, fontsize=14)
        sns.despine()
        fig.tight_layout()
        fig.subplots_adjust(top=.9)
        

def plot_rolling_stats(x, window=48, figsize=(10,3)):
    x_rolling_mean = x.rolling(window).mean()
    x_rolling_std = x.rolling(window).std()
    x_std_of_std = round(x_rolling_std.std(),3)
    x_std_of_mean = round(x_rolling_mean.std(),3)
    x_mean_of_mean = round(x_rolling_mean.mean(),3)
    x_mean_of_std = round(x_rolling_std.mean(),3)
    adf_ = tsa.adfuller(x.dropna())[1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_rolling_mean, color='b', alpha=0.8,
            label=f'{window}-period mean\nmean:  {x_mean_of_mean}\nstd:  {x_std_of_mean}\nADF: {np.round(adf_,9)}')
#     ax.text(0.2,0.9, f'ADF: {np.round(adf_,9)}', transform=ax.transAxes)
    ax.legend(loc='upper left')
    ax_twinx = ax.twinx()
    ax_twinx.plot(x_rolling_std, color='orange',alpha=0.8,
              label=f'{window}-period std\nmean:  {x_mean_of_std}\nstd:  {x_std_of_std}')
    ax_twinx.legend(loc='upper right')
    ax.grid(axis='y')
#     plt.legend([f'{x.name} {window}-period mean', f'{x.name} {window}-period std'])
#     plt.show()
    return ax


def periodicity_hourly(x, n=3, sampling_freq_hz = 0.000277777777):
    """Returns top 'n' intervals where spectral density peaks occur to help
    identify periods in cyclical data"""
#     hourly sampling frequency
#     sampling_freq_hz = 0.0002777777777777778  
#     alternatively (not used)
#     sampling_freq_hz =  1 / pd.Timedelta(x.index.freq).seconds
#     freqstr = x.index.freqstr
    temp = pd.DataFrame(
            index= np.round(sampling_freq_hz / periodogram(x.dropna(),
                               fs=sampling_freq_hz)[0], 0),
             data = {'Power spectral density': periodogram(x.dropna(),
                                           fs=sampling_freq_hz)[1]}).rename_axis('Hours')
    return temp.nlargest(n, 'Power spectral density').index.tolist()