#!/usr/bin/env python
# coding: utf-8

# # Config
# This is the configuration file for a set of scripts that produce hourly forecasts for Day-Ahead (DA) and Real-Time (RT) Local Marginal Prices (LMP) of electrical generation at the North Hub Bus.
# 
# The data used in this analysis is from January 1, 2022 through August 19, 2022. The source data contains 5519 records with 47 columns and a datetime index, and includes null data points. The data are almost entirely numeric and require minimal cleaning.
# 
# December 10, 2024
# Matt Chmielewski

import pandas as pd
import pathlib
import matplotlib.pyplot as plt

#Analysis Parameters

timezone = 'US/Central'
max_fraction_nan = 0.05 # maximum fraction of all records
# containing any NaN value 
n_max_consec_nans_interp = 4  # maximum number of consecutive
# NaN values allowed to be imputed via interpolatation


# pandas configuration
pd.options.display.max_columns = 50


#plotting configuration
try: 
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        style_poster = [x_ for x_ in plt.style.available if x_.endswith('poster')][0]
        plt.style.use(style_poster)

plt.rcParams['figure.figsize'] = [11,5]
plt.rcParams["figure.titlesize"] = 'x-large'

#locations (expects that .ipynb code is stored in a folder on same level as following folders)
dir_cwd = pathlib.Path.cwd()
data_dir = dir_cwd.parent / 'data'
if not data_dir.exists():
    data_dir.mkdir(parents=True)
assert data_dir.is_dir()

model_dir = dir_cwd.parent / 'model'
if not model_dir.exists():
    model_dir.mkdir(parents=True)
assert model_dir.is_dir()

file_source = data_dir / "POC Sample Data.xlsx"
file_source_test = data_dir / "POC Sample Data_test.xlsx"
file_features = data_dir / 'features.csv'
file_features_select = data_dir / 'features_select.csv'
file_ts = data_dir / 'ts.csv'
file_ts_forecast = data_dir / 'ts_forecast.csv'
file_ts_stationary = data_dir / 'ts_select_stationary.csv'
file_ts_select = data_dir / 'ts_select.csv'
file_causality = data_dir / 'feature_causality.csv'


# Feature groups and colors
cols_target = ['HB_NORTH (DALMP)', 'HB_NORTH (RTLMP)']
cols_actual_solve_for = ['HB_NORTH (DALMP)', 'HB_NORTH (RTLMP)', 'WZ_Coast (RTLOAD)',
        'WZ_ERCOT (RTLOAD)', 'WZ_East (RTLOAD)', 'WZ_FarWest (RTLOAD)',
        'WZ_North (RTLOAD)', 'WZ_NorthCentral (RTLOAD)',
        'WZ_SouthCentral (RTLOAD)', 'WZ_Southern (RTLOAD)', 'WZ_West (RTLOAD)',
        'GR_COASTAL (WINDDATA)', 'GR_ERCOT (WINDDATA)', 'GR_NORTH (WINDDATA)',
        'GR_PANHANDLE (WINDDATA)', 'GR_SOUTH (WINDDATA)', 'GR_WEST (WINDDATA)',
        'ERCOT (GENERATION_SOLAR_RT)']

cmap_selection = ['Blues', 'Greens', 'Oranges', 'Greys',
      'coolwarm', 'Purples', 'Reds', 'Viridis', 'Viridis', 'Viridis']

feature_types = ['Demand', 'Wind', 'Solar', 'Outages', 'Fuel Price',
       'DayAhead Price', 'Rt Price']

cmaps = dict.fromkeys(feature_types)
feature_cmap = {k:  plt.get_cmap(cmap_selection[idx]) for idx, k in enumerate(cmaps.keys())}
feature_type_colors = {k: v(0.7) for k, v in feature_cmap.items()}

colors_features = {'HB_NORTH (DALMP)': (0.7137254901960784,
  0.7137254901960784,
  0.8470588235294118,
  1.0),
 'WZ_Coast (RTLOAD)': (0.5796078431372549,
  0.7701960784313725,
  0.8737254901960784,
  1.0),
 'WZ_ERCOT (RTLOAD)': (0.5231372549019608,
  0.73919261822376,
  0.8615455594002307,
  1.0),
 'WZ_East (RTLOAD)': (0.4666666666666667,
  0.7081891580161477,
  0.8493656286043829,
  1.0),
 'WZ_FarWest (RTLOAD)': (0.4120415224913495,
  0.6771856978085352,
  0.8362629757785467,
  1.0),
 'WZ_North (RTLOAD)': (0.37168781237985393,
  0.6496270665128797,
  0.8205151864667436,
  1.0),
 'WZ_NorthCentral (RTLOAD)': (0.32628988850442137,
  0.6186236063052672,
  0.802798923490965,
  1.0),
 'WZ_SouthCentral (RTLOAD)': (0.28089196462898885,
  0.5876201460976547,
  0.7850826605151865,
  1.0),
 'WZ_Southern (RTLOAD)': (0.24004613610149955,
  0.5537716262975779,
  0.7667973856209152,
  1.0),
 'WZ_West (RTLOAD)': (0.2035063437139562,
  0.5172318339100346,
  0.7479738562091504,
  1.0),
 'WZ_Coast (BIDCLOSE_LOAD_FORECAST)': (0.16696655132641292,
  0.48069204152249134,
  0.7291503267973857,
  1.0),
 'WZ_ERCOT (BIDCLOSE_LOAD_FORECAST)': (0.13042675893886968,
  0.4441522491349481,
  0.710326797385621,
  1.0),
 'WZ_East (BIDCLOSE_LOAD_FORECAST)': (0.10249903883121878,
  0.4086889657823914,
  0.6828911956939638,
  1.0),
 'WZ_FarWest (BIDCLOSE_LOAD_FORECAST)': (0.07481737793156479,
  0.3732564398308343,
  0.6552095347943099,
  1.0),
 'WZ_North (BIDCLOSE_LOAD_FORECAST)': (0.05021145713187236,
  0.341760861207228,
  0.6306036139946175,
  1.0),
 'WZ_NorthCentral (BIDCLOSE_LOAD_FORECAST)': (0.03137254901960784,
  0.3059746251441753,
  0.5944329104190696,
  1.0),
 'WZ_SouthCentral (BIDCLOSE_LOAD_FORECAST)': (0.03137254901960784,
  0.26943483275663205,
  0.5401768550557479,
  1.0),
 'WZ_Southern (BIDCLOSE_LOAD_FORECAST)': (0.03137254901960784,
  0.23289504036908892,
  0.4859207996924262,
  1.0),
 'WZ_West (BIDCLOSE_LOAD_FORECAST)': (0.03137254901960784,
  0.19635524798154555,
  0.4316647443291042,
  1.0),
 'Katy (GASPRICE)': (0.753610618, 0.830232851, 0.960871157, 1.0),
 'Henry (GASPRICE)': (0.717434544917647,
  0.05111754842352939,
  0.15873660770196077,
  1.0),
 'ERCOT (TOTAL_RESOURCE_CAP_OUT)': (0.7105882352941176,
  0.7105882352941176,
  0.7105882352941176,
  1.0),
 'HB_NORTH (RTLMP)': (0.9874509803921568,
  0.5411764705882353,
  0.41568627450980394,
  1.0),
 'ERCOT (GENERATION_SOLAR_RT)': (0.9921568627450981,
  0.6564705882352941,
  0.3827450980392157,
  1.0),
 'ERCOT (SOLAR_STPPF_BIDCLOSE)': (0.5076355247981545,
  0.1566320645905421,
  0.015440215301806996,
  1.0),
 'GR_COASTAL (WINDDATA)': (0.596078431372549,
  0.8345098039215686,
  0.5788235294117646,
  1.0),
 'GR_ERCOT (WINDDATA)': (0.5462514417531719,
  0.8112572087658593,
  0.5378546712802768,
  1.0),
 'GR_NORTH (WINDDATA)': (0.4908881199538639,
  0.7854209919261823,
  0.49233371780084584,
  1.0),
 'GR_PANHANDLE (WINDDATA)': (0.4392156862745098,
  0.7609381007304883,
  0.45505574778931185,
  1.0),
 'GR_SOUTH (WINDDATA)': (0.3764705882352941,
  0.7301806997308727,
  0.42429834678969625,
  1.0),
 'GR_WEST (WINDDATA)': (0.31999999999999995,
  0.7024990388312188,
  0.3966166858900423,
  1.0),
 'ERCOT (WIND_STWPF_BIDCLOSE)': (0.25725490196078427,
  0.6717416378316032,
  0.36585928489042674,
  1.0),
 'GR_COASTAL (WIND_STWPF_BIDCLOSE)': (0.22306805074971164,
  0.636632064590542,
  0.3392387543252595,
  1.0),
 'GR_ERCOT (WIND_STWPF_BIDCLOSE)': (0.18985005767012686,
  0.601199538638985,
  0.3126643598615917,
  1.0),
 'GR_NORTH (WIND_STWPF_BIDCLOSE)': (0.15294117647058825,
  0.5618300653594771,
  0.28313725490196073,
  1.0),
 'GR_PANHANDLE (WIND_STWPF_BIDCLOSE)': (0.11680123029604011,
  0.5275663206459055,
  0.25597846981930034,
  1.0),
 'GR_SOUTH (WIND_STWPF_BIDCLOSE)': (0.07374086889657824,
  0.49065743944636675,
  0.22522106881968473,
  1.0),
 'GR_WEST (WIND_STWPF_BIDCLOSE)': (0.034986543637062675,
  0.45743944636678197,
  0.19753940792003075,
  1.0),
 'NORTH (ERCOT) (WIND_STWPF_BIDCLOSE)': (0.0,
  0.41799307958477505,
  0.16862745098039217,
  1.0),
 'SOUTH_HOUSTON (WIND_STWPF_BIDCLOSE)': (0.0,
  0.37259515570934254,
  0.14980392156862746,
  1.0),
 'WEST (ERCOT) (WIND_STWPF_BIDCLOSE)': (0.0,
  0.3221530180699732,
  0.12888888888888894,
  1.0),
 'WEST_NORTH (WIND_STWPF_BIDCLOSE)': (0.0,
  0.2767550941945406,
  0.11006535947712418,
  1.0)}

colors_target = {col : color for col, color in zip(cols_target,
                   [colors_features[col] for col in cols_target])}

colors_target_light = {k : tuple(list(v[:3]) + [0.1]) for k, v in colors_target.items()}

colors_month_light = {1: (0.4843137254901961, 0.02463744919538197, 0.9999241101148306, 0.1),
 2: (0.303921568627451, 0.30315267411304353, 0.9881654720812594, 0.1),
 3: (0.12352941176470589, 0.5574894393428855, 0.9566044195004408, 0.1),
 4: (0.0490196078431373, 0.7594049166547072, 0.9084652718195236, 0.1),
 5: (0.22941176470588232, 0.9110226492460882, 0.8403440716378927, 0.1),
 6: (0.40980392156862744, 0.989980213280707, 0.7553827347189938, 0.1),
 7: (0.5901960784313725, 0.989980213280707, 0.6552838500134537, 0.1),
 8: (0.7705882352941176, 0.9110226492460884, 0.5420533564724495, 0.1),
 9: (0.9509803921568627, 0.7594049166547073, 0.4179603448867836, 0.1),
 10: (1.0, 0.5574894393428858, 0.29138974688932473, 0.1),
 11: (1.0, 0.30315267411304364, 0.15339165487868545, 0.1),
 12: (1.0, 0.024637449195382025, 0.012319659535238468, 0.1)}

colors_month_bold = {1: (0.4843137254901961, 0.02463744919538197, 0.9999241101148306, 1.0),
 2: (0.303921568627451, 0.30315267411304353, 0.9881654720812594, 1.0),
 3: (0.12352941176470589, 0.5574894393428855, 0.9566044195004408, 1.0),
 4: (0.0490196078431373, 0.7594049166547072, 0.9084652718195236, 1.0),
 5: (0.22941176470588232, 0.9110226492460882, 0.8403440716378927, 1.0),
 6: (0.40980392156862744, 0.989980213280707, 0.7553827347189938, 1.0),
 7: (0.5901960784313725, 0.989980213280707, 0.6552838500134537, 1.0),
 8: (0.7705882352941176, 0.9110226492460884, 0.5420533564724495, 1.0),
 9: (0.9509803921568627, 0.7594049166547073, 0.4179603448867836, 1.0),
 10: (1.0, 0.5574894393428858, 0.29138974688932473, 1.0),
 11: (1.0, 0.30315267411304364, 0.15339165487868545, 1.0),
 12: (1.0, 0.024637449195382025, 0.012319659535238468, 1.0)}
