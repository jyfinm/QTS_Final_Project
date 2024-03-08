"""
Source file containing all functions used in the notebook, imported directly.

"""
# ============================================

import sys
import math
import warnings

import psycopg2
import wrds
import gzip

import seaborn as sns
import os
import quandl
import json
import zipfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import functools
import requests
import io

from matplotlib.ticker import PercentFormatter
import urllib.request
from urllib.error import HTTPError
# from html_table_parser.parser import HTMLTableParser
#from bs4 import BeautifulSoup
import re

import plotnine as p9
from plotnine import ggplot, scale_x_date, guides, guide_legend, geom_bar, scale_y_continuous, \
    scale_color_identity, geom_line, geom_point, labs, theme_minimal, theme, element_blank, element_text, \
        geom_ribbon, geom_hline, aes, scale_size_manual, scale_color_manual, ggtitle

from datetime import datetime
import datetime

import pandas as pd
# import pandas_market_calendars as mcal
from pandas.plotting import autocorrelation_plot
import numpy as np
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import scipy as sp
from scipy.stats import norm
import scipy.stats as stats

from statsmodels.tsa.stattools import coint
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from collections import deque
from bisect import insort, bisect_left
from itertools import islice



# ============================================

# Quandl data retrieval functions


def grab_quandl_table(
    table_path,
    quandl_api_key,
    avoid_download=False,
    replace_existing=False,
    date_override=None,
    allow_old_file=False,
    **kwargs,
):
    root_data_dir = os.path.join(os.getcwd(), "quandl_data_table_downloads")
    data_symlink = os.path.join(root_data_dir, f"{table_path}_latest.zip")
    if avoid_download and os.path.exists(data_symlink):
        print(f"Skipping any possible download of {table_path}")
        return data_symlink
    
    table_dir = os.path.dirname(data_symlink)
    if not os.path.isdir(table_dir):
        print(f'Creating new data dir {table_dir}')
        os.makedirs(table_dir)

    if date_override is None:
        my_date = datetime.datetime.now().strftime("%Y%m%d")
    else:
        my_date = date_override
    data_file = os.path.join(root_data_dir, f"{table_path}_{my_date}.zip")

    if os.path.exists(data_file):
        file_size = os.stat(data_file).st_size
        if replace_existing or not file_size > 0:
            print(f"Removing old file {data_file} size {file_size}")
        else:
            print(
                f"Data file {data_file} size {file_size} exists already, no need to download"
            )
            return data_file

    dl = quandl.export_table(
        table_path, filename=data_file, api_key=quandl_api_key, **kwargs
    )
    file_size = os.stat(data_file).st_size
    if os.path.exists(data_file) and file_size > 0:
        print(f"Download finished: {file_size} bytes")
        if not date_override:
            if os.path.exists(data_symlink):
                print(f"Removing old symlink")
                os.unlink(data_symlink)
            print(f"Creating symlink: {data_file} -> {data_symlink}")
            os.symlink(
                data_file, data_symlink,
            )
    else:
        print(f"Data file {data_file} failed download")
        return
    return data_symlink if (date_override is None or allow_old_file) else "NoFileAvailable"

def fetch_quandl_table(table_path, api_key, avoid_download=True, **kwargs):
    return pd.read_csv(
        grab_quandl_table(table_path, api_key, avoid_download=avoid_download, **kwargs)
    )


# ============================================

# Data Restructuring

# Checkpoint2

def prepare_ticker_data(ticker_data, start_date, end_date):
    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
    filtered_data = ticker_data[(ticker_data['date'] >= start_date) & (ticker_data['date'] <= end_date)]
    sorted_data = filtered_data.sort_values(by='date')
    reduced_data = sorted_data[['date', 'close', 'adj_open', 'adj_close', 'adj_volume']]
    return reduced_data

def prepare_option_data(option_data):
    option_data['date'] = pd.to_datetime(option_data['date'])
    option_data['strike_price'] = option_data['strike_price'] / 1000.0
    return option_data

def enrich_option_data(option_data, ticker_data_reduced):
    enriched_data = pd.merge(option_data, ticker_data_reduced, on='date', how='left')
    return enriched_data

# Checkpoint3
def merge_dataframes(original_dfs, result_dfs):
    filtered_dfs = {}
    for key in original_dfs.keys():
        df = original_dfs[key]
        result_df = result_dfs[key]
        filtered_df = df.merge(result_df, on=['date', 'TTE'])
        filtered_dfs[key] = filtered_df
    return filtered_dfs

def filter_dfs_on_criteria(filtered_dfs, select_row_with_smallest_diff):
    final_dfs = {}
    for key, df in filtered_dfs.items():
        final_df = df.groupby('date', as_index=False).apply(select_row_with_smallest_diff).reset_index(drop=True)
        final_df = final_df.drop(columns=['abs_diff'])
        final_dfs[key] = final_df
    return final_dfs

def calculate_closest_dates(final_dfs):
    for key, df in final_dfs.items():
        df['date'] = pd.to_datetime(df['date'])
        dates_np = df['date'].values
        target_dates = dates_np + np.timedelta64(21, 'D')
        abs_diff_matrix = np.abs(dates_np[:, None] - target_dates)
        min_diff_indices = np.argmin(abs_diff_matrix, axis=0)
        closest_dates = dates_np[min_diff_indices]
        df['close_date'] = closest_dates
        final_dfs[key] = df
    return final_dfs

def merge_with_options(final_dfs, options_df1):
    options_df1['date'] = pd.to_datetime(options_df1['date'])
    options_df1['exdate'] = pd.to_datetime(options_df1['exdate'])
    
    for key, df in final_dfs.items():
        df['close_date'] = pd.to_datetime(df['close_date'])
        df['exdate'] = pd.to_datetime(df['exdate'])
        merged_df = pd.merge(df, options_df1, left_on=['close_date', 'cp_flag', 'strike_price', 'exdate'],
                             right_on=['date', 'cp_flag', 'strike_price', 'exdate'],
                             how='left', indicator=True)
        
        merged_df['is_present'] = merged_df['_merge'] == 'both'
        columns_to_drop = ['_merge'] + [col for col in merged_df.columns if col.endswith('_y')]
        merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        final_dfs[key] = merged_df
    return final_dfs

def find_closest_tte(group, days = 30):
    group['diff'] = (group['TTE'] - days).abs() - group['TTE'].lt(days) * 0.1
    return group.loc[group['diff'].idxmin()]

def select_row_with_smallest_diff(group):
    group['abs_diff'] = (group['close'] - group['strike_price']).abs()
    return group.loc[group['abs_diff'].idxmin()]

def check_dates_in_calls_and_puts(calls_key, puts_key, final_dfs):
    calls_df = final_dfs[calls_key]
    puts_df = final_dfs[puts_key]

    filtered_calls = calls_df[calls_df['is_present'] == False]
    filtered_puts = puts_df[puts_df['is_present'] == False]

    calls_dates = set(filtered_calls['date_x'].unique())
    puts_dates = set(filtered_puts['date_x'].unique())

    return calls_dates.issubset(puts_dates)

def count_dates_comparison(calls_key, puts_key, final_dfs):
    calls_df = final_dfs[calls_key][final_dfs[calls_key]['is_present'] == False]
    puts_df = final_dfs[puts_key][final_dfs[puts_key]['is_present'] == False]

    calls_dates = set(calls_df['date_x'].unique())
    puts_dates = set(puts_df['date_x'].unique())

    unique_in_calls = calls_dates.difference(puts_dates)
    unique_in_puts = puts_dates.difference(calls_dates)

    overlapping_dates = calls_dates.intersection(puts_dates)

    count_unique_in_calls = len(unique_in_calls)
    count_unique_in_puts = len(unique_in_puts)
    count_overlapping = len(overlapping_dates)

    return count_unique_in_calls, count_unique_in_puts, count_overlapping

# ============================================

# Further Data restructuring

def preprocess_options_data(calls_df, puts_df):
    calls_df.rename(columns={'date_x': 'date'}, inplace=True)
    puts_df.rename(columns={'date_x': 'date'}, inplace=True)

    calls_df['date'] = pd.to_datetime(calls_df['date'])
    puts_df['date'] = pd.to_datetime(puts_df['date'])

    calls_df['exdate'] = pd.to_datetime(calls_df['exdate'])
    puts_df['exdate'] = pd.to_datetime(puts_df['exdate'])

    return calls_df, puts_df


def compare_strike_prices(calls_df, puts_df):
    calls_grouped = calls_df.groupby('date')['strike_price'].apply(list).reset_index(name='calls_strike_prices')
    puts_grouped = puts_df.groupby('date')['strike_price'].apply(list).reset_index(name='puts_strike_prices')

    merged_df = pd.merge(calls_grouped, puts_grouped, on='date')
    merged_df['strike_prices_match'] = merged_df.apply(lambda row: set(row['calls_strike_prices']) == set(row['puts_strike_prices']), axis=1)

    return merged_df[merged_df['strike_prices_match'] == False]['date']

def compare_strike_prices_and_exdates(calls_df, puts_df):
    calls_grouped = calls_df.groupby('date').apply(lambda x: list(zip(x['strike_price'], x['exdate']))).reset_index(name='calls_data')
    puts_grouped = puts_df.groupby('date').apply(lambda x: list(zip(x['strike_price'], x['exdate']))).reset_index(name='puts_data')

    merged_df = pd.merge(calls_grouped, puts_grouped, on='date')
    merged_df['data_match'] = merged_df.apply(lambda row: set(row['calls_data']) == set(row['puts_data']), axis=1)

    return merged_df[merged_df['data_match'] == False]['date']




# ============================================

# Simulation Code

# Gives more metrics, but we will generally use the later, streamlined version of creating simulations in the actual implementation.

def create_simulations_original(options_subset, data, dropna_greeks=False):
    simulations = {}

    for index, row in options_subset.iterrows():
        strikeID = row['exdate'].strftime('%Y%m%d') + '_' + str(row['strike_price'])
        mask = (data['strikeID'] == strikeID) & (data['date'] >= row['date']) & (data['date'] <= row['close_date'])
        temp_df = data[mask].sort_values(by=['date', 'cp_flag'])

        shared_cols = ['date', 'exdate', 'strike_price', 'expiry_indicator', 'close', 'adj_open', 'adj_close', 'adj_volume', 'strikeID']
        greeks_cols = ['impl_volatility', 'delta', 'gamma', 'vega', 'theta']
        call_specific_cols = ['cp_flag', 'best_bid', 'best_offer', 'volume', 'open_interest'] + greeks_cols
        put_specific_cols = call_specific_cols

        calls = temp_df[temp_df['cp_flag'] == 'C'][shared_cols + call_specific_cols].rename(columns={col: col + '_c' for col in call_specific_cols})
        puts = temp_df[temp_df['cp_flag'] == 'P'][shared_cols + put_specific_cols].rename(columns={col: col + '_p' for col in put_specific_cols})

        merged_df = pd.merge(calls, puts, on=shared_cols, how='outer')

        if dropna_greeks:
            greeks_cols_c = [col + '_c' for col in greeks_cols]
            greeks_cols_p = [col + '_p' for col in greeks_cols]
            merged_df = merged_df.dropna(subset=greeks_cols_c + greeks_cols_p, how='any')

        merged_df['delta_sum'] = merged_df['delta_c'].fillna(0) + merged_df['delta_p'].fillna(0)
        merged_df['shares_held'] = -1 * merged_df['delta_sum']

        merged_df = merged_df.sort_values(by='date')
        merged_df['sharechange'] = merged_df['shares_held'].diff()

        simulations[row['date'].strftime('%Y-%m-%d')] = merged_df

    return simulations

# Current version of create_simulations

def create_simulations(options_subset, data, dropna_greeks=False):
    simulations = {}

    for index, row in options_subset.iterrows():
        strikeID = row['exdate'].strftime('%Y%m%d') + '_' + str(row['strike_price'])
        mask = (data['strikeID'] == strikeID) & (data['date'] >= row['date']) & (data['date'] <= row['close_date'])
        temp_df = data[mask].sort_values(by=['date', 'cp_flag'])

        shared_cols = ['date', 'exdate', 'strike_price', 'close', 'strikeID'] # 'expiry_indicator',  'adj_open', 'adj_close', 'adj_volume',
        greeks_cols = ['impl_volatility', 'delta'] # , 'gamma', 'vega', 'theta'
        call_specific_cols = ['cp_flag', 'best_bid', 'best_offer'] + greeks_cols # , 'volume', 'open_interest'
        put_specific_cols = call_specific_cols

        calls = temp_df[temp_df['cp_flag'] == 'C'][shared_cols + call_specific_cols].rename(columns={col: col + '_c' for col in call_specific_cols})
        puts = temp_df[temp_df['cp_flag'] == 'P'][shared_cols + put_specific_cols].rename(columns={col: col + '_p' for col in put_specific_cols})

        merged_df = pd.merge(calls, puts, on=shared_cols, how='outer')

        if dropna_greeks:
            greeks_cols_c = [col + '_c' for col in greeks_cols]
            greeks_cols_p = [col + '_p' for col in greeks_cols]
            merged_df = merged_df.dropna(subset=greeks_cols_c + greeks_cols_p, how='any')

        merged_df['delta_sum'] = merged_df['delta_c'].fillna(0) + merged_df['delta_p'].fillna(0)
        merged_df['shares_held'] = -1 * merged_df['delta_sum']

        merged_df = merged_df.sort_values(by='date')
        merged_df['sharechange'] = merged_df['shares_held'].diff()

        simulations[row['date'].strftime('%Y-%m-%d')] = merged_df

    return simulations

# ============================================

# IV calculations

def find_closest_index(val, col2):
    return np.abs(col2 - val).idxmin()

def calculate_iv_calls(df, delta_k, s_0, zcb_price):
    options_df = df.copy()
    options_df['adjusted_strike'] = options_df['strike_price'] / (zcb_price/100)

    options_df['closest_price_index'] = options_df['adjusted_strike'].apply(lambda x: find_closest_index(x, options_df['strike_price']))

    options_prices = options_df.loc[options_df['closest_price_index'].values, 'midpt']

    underlying_minus_strike = s_0 - (options_df['strike_price']/1000)
    underlying_minus_strike[underlying_minus_strike < 0] = 0
    options_prices.index = underlying_minus_strike.index
    options_df['g'] = (options_prices - underlying_minus_strike)/((options_df['strike_price']/1000)**2)
    
    options_df.iloc[1:-2, -1] = options_df.iloc[1:-2, -1] * 2
    return options_df.iloc[:, -1].sum() * delta_k

def calc_s0(dt, zcb_price, dte, spydata):
    unadjusted_price = spydata[spydata['date'] == dt]['close'].iloc[0]
    dividend_window_end = dt + pd.Timedelta(days=dte)

    dividends_to_be_paid = spydata[(spydata['date'] <= dividend_window_end) & (spydata['date'] >= dt)]['dividend'].sum()
    adjusted_price = unadjusted_price - (dividends_to_be_paid * zcb_price)
    return adjusted_price

def calculate_iv_for_calls(calls, option_data, tbills, spydata):
    our_ivs = pd.DataFrame(columns=['iv'])
    sizes = pd.Series()

    for ind, dt in enumerate(calls['date_x']):
        call_df = option_data[(option_data['date'] == dt) & (option_data['cp_flag'] == 'C')]
        call_df = call_df[call_df['dte'] == calls.loc[ind, 'dte']]
        call_df = call_df.sort_values('strike_price')

        call_df = call_df.reset_index(drop=True)

        call_df = call_df[call_df['midpt'] > 0.375]
        atm_strike = calls.loc[ind, 'strike_price']

        call_df = call_df[call_df['strike_price'] > 0.97 * atm_strike]

        call_df['increments'] = call_df['strike_price'].diff().bfill()

        mode = call_df['increments'].mode().iloc[0]
        inds_to_drop = pd.Series(call_df[call_df['increments'] > mode].index)
        midpoint = call_df.shape[0] / 2
        lower_inds_to_drop = inds_to_drop[inds_to_drop < midpoint]
        upper_inds_to_drop = inds_to_drop[inds_to_drop > midpoint]

        if not lower_inds_to_drop.empty:
            call_df = call_df.iloc[lower_inds_to_drop.max():]

        if not upper_inds_to_drop.empty:
            call_df = call_df.iloc[:upper_inds_to_drop.min()]

        valid_strikes = np.arange(start=int(call_df['strike_price'].min()), stop=call_df['strike_price'].max() + mode, step=mode)
        call_df = call_df[call_df['strike_price'].isin(valid_strikes)]

        # Look at how many options are being used to find IV
        sizes.loc[ind] = call_df.shape[0]

        delta_k = mode / 1000
        dte = call_df['dte'].iloc[0]

        tbills_today = tbills[(tbills['quote_date'] == dt)]
        days_back = -1
        while tbills_today.empty:
            tbills_today = tbills[tbills['quote_date'] == dt + pd.Timedelta(days=days_back)]
            days_back -= 1
        zcb_price = tbills_today[abs(tbills_today['dte'] - dte) == abs(tbills_today['dte'] - dte).min()]['price'].iloc[0]

        s_0 = calc_s0(dt, zcb_price, dte, spydata)

        iv = calculate_iv_calls(call_df, delta_k, s_0, zcb_price)

        our_ivs.loc[dt] = [iv]
    
    return our_ivs, sizes

# ============================================

# Strategy Simulation Part 1

def prepare_dataframes(data_file_path, options_file_path):
    data = pd.read_csv(data_file_path)
    options = pd.read_csv(options_file_path)

    data['exdate'] = pd.to_datetime(data['exdate'])
    options['exdate'] = pd.to_datetime(options['exdate'])

    data['exdate_str'] = data['exdate'].dt.strftime('%Y%m%d')
    data['strikeID'] = data['exdate_str'] + '_' + data['strike_price'].astype(str)
    data.drop(columns=['exdate_str'], inplace=True)

    options['exdate_str'] = options['exdate'].dt.strftime('%Y%m%d')
    options['strikeID'] = options['exdate_str'] + '_' + options['strike_price'].astype(str)
    options.drop(columns=['exdate_str'], inplace=True)

    options['date'] = pd.to_datetime(options['date'])
    data['date'] = pd.to_datetime(data['date'])

    return data, options

# dividend pay dates
def find_pay_date(end_of_month, trading_days):
    if end_of_month in trading_days:
        return end_of_month
    else:
        eligible_days = trading_days[trading_days <= end_of_month]
        return eligible_days.max() 
    
def process_spy_dividends(ticker_data_path, start_date, end_date):
    spy_divdata = pd.read_csv(ticker_data_path)[['date', 'dividend']].sort_values(by='date').reset_index(drop=True)
    spy_divdata = spy_divdata.loc[(spy_divdata['date'] >= start_date) & (spy_divdata['date'] <= end_date)].copy().reset_index(drop=True)
    spy_divdata['date'] = pd.to_datetime(spy_divdata['date'])
    
    trading_days = spy_divdata['date']
    
    spy_divdata = spy_divdata.loc[spy_divdata['dividend'] != 0]

    spy_divdata['end_of_next_month'] = spy_divdata['date'] + pd.offsets.MonthEnd(2)
    spy_divdata['pay_date'] = spy_divdata['end_of_next_month'].apply(lambda date: find_pay_date(date, trading_days))
    spy_divdata.drop(columns=['end_of_next_month'], inplace=True)

    return spy_divdata, trading_days
    
def filter_simulations(simulations, trading_days):
    filtered_simulations = {}

    for key, df in simulations.items():
        # Ensure 'date' column is in datetime64 dtype
        df['date'] = pd.to_datetime(df['date'])
        
        # Find the range of trading days for each simulation
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        # Filter the trading_days series to get the expected range of dates
        expected_trading_days = trading_days[(trading_days >= start_date) & (trading_days <= end_date)]
        
        # Get unique trading days from the simulation
        actual_trading_days = df['date'].unique()
        actual_trading_days = pd.to_datetime(actual_trading_days)
        
        # Check if all expected trading days are present in the actual trading days
        if expected_trading_days.isin(actual_trading_days).all():
            filtered_simulations[key] = df
    
    return filtered_simulations

# ============================================

# Strategy simulation part 2: PL/metrics

def calculate_realized_PL(df, long_op=True):
    df = df.reset_index(drop=True)
    
    # Vectorized initial operations for stock
    df['stock_pos'] = np.where(long_op, df['shares_held'], -df['shares_held'])
    df = df.drop(columns=['shares_held'])
    df['pos_change'] = np.where(long_op, df['sharechange'], -df['sharechange'])
    df = df.drop(columns=['sharechange'])
    df.loc[0, 'pos_change'] = df.loc[0, 'stock_pos']
    
    df['change_cost_basis'] = df['pos_change'] * df['close']
    df['stock_cost_basis'] = df['change_cost_basis'].cumsum()
    df['daily_stock_value'] = df['stock_pos'] * df['close']
    df['stock_PL'] = df['daily_stock_value'] - df['stock_cost_basis']

    # Initial option value and vectorized daily option value calculation
    df['option_cost_basis'] = df.loc[0, 'best_offer_c'] + df.loc[0, 'best_offer_p'] if long_op else -df.loc[0, 'best_bid_c'] - df.loc[0, 'best_bid_p']
    df['change_cost_basis_op'] = 0.0
    df.loc[0, 'change_cost_basis_op'] = df.loc[0, 'option_cost_basis']
    df['daily_option_value'] = np.where(long_op, df['best_bid_c'] + df['best_bid_p'], -(df['best_offer_c'] + df['best_offer_p']))
    df['option_PL'] = df['daily_option_value']- df['option_cost_basis']

    # Column to track total positions, PL, and cash flow after positions are closed
    df['total_cost_basis'] = df['stock_cost_basis'] + df['option_cost_basis']
    df['total_pos_value'] = df['daily_stock_value'] + df['daily_option_value']
    df['total_PL'] = df['stock_PL'] + df['option_PL']
    df['realized_stock_PL'] = 0.0
    df['realized_option_PL'] = 0.0
    df['realized_PL'] = 0.0

    # Misc
    df['UID'] = df['strikeID'] + '_' + str(df.loc[0, 'date'].date())
    df['to_open'] = 0
    df.loc[0, 'to_open'] = 1
#    df['gross_trades_value'] = abs(df['to_open'] * df['option_cost_basis']) + abs(df['change_cost_basis']) # Need to do this at end

    # Close positions on final day
    final_row_index = len(df) - 1
    df.loc[final_row_index, 'realized_stock_PL'] = df.loc[final_row_index, 'stock_PL'] if final_row_index > 0 else 0
    df.loc[final_row_index, 'realized_option_PL'] = df.loc[final_row_index, 'option_PL'] if final_row_index > 0 else 0
    df.loc[final_row_index, 'realized_PL'] = df.loc[final_row_index, 'total_PL'] if final_row_index > 0 else 0
#    df.loc[final_row_index, 'gross_trades_value'] = abs(df.loc[final_row_index, 'daily_option_value']) + abs(df.loc[final_row_index - 1, 'stock_pos']) * df.loc[final_row_index, 'close'] 

    final_close_price = df.loc[final_row_index, 'close']
    df.loc[final_row_index, 'stock_pos'] = 0
    df.loc[final_row_index, 'pos_change'] = - df.loc[final_row_index - 1, 'stock_pos'] if final_row_index > 0 else 0
    df.loc[final_row_index, 'change_cost_basis'] = df.loc[final_row_index, 'pos_change'] * final_close_price
    df.loc[final_row_index, 'stock_cost_basis'] = 0
    df.loc[final_row_index, 'daily_stock_value'] = 0
    df.loc[final_row_index, 'stock_PL'] = 0

    df.loc[final_row_index, 'option_cost_basis'] = 0
    df.loc[final_row_index, 'change_cost_basis_op'] = -df.loc[final_row_index, 'daily_option_value'] if final_row_index > 0 else 0
    df.loc[final_row_index, 'daily_option_value'] = 0
    df.loc[final_row_index, 'option_PL'] = 0

    df.loc[final_row_index, 'total_cost_basis'] = 0
    df.loc[final_row_index, 'total_pos_value'] = 0
    df.loc[final_row_index, 'total_PL'] = 0
    
    return df

def compare_iv(filtered_simulations, iv_data):
    temp_data = []

    for key, df in filtered_simulations.items():
        temp_data.append({'date': key, 'BS_Call_IV': df.loc[0, 'impl_volatility_c']})

    BS_Call_IV = pd.DataFrame(temp_data)

    iv_data['date'] = pd.to_datetime(iv_data['date'])
    BS_Call_IV['date'] = pd.to_datetime(BS_Call_IV['date'])

    IV_compare = pd.merge(BS_Call_IV, iv_data[['date', 'iv']], on='date', how='left')
    IV_compare.rename(columns={'iv': 'MF_Call_IV'}, inplace=True)
    IV_compare['IV_diff'] = IV_compare['MF_Call_IV'] - IV_compare['BS_Call_IV']

    return IV_compare

# ============================================

# Strategies

# Long-Short
def trade_strategy_1(x):
    if x > 0.25:
        return 1
    elif x < -0.10:
        return -1
    else:
        return 0

# Long Only
def trade_strategy_2(x):
    if x > 0.35:
        return 1
    else:
        return 0

# Short Only
def trade_strategy_3(x):
    if x < -0.08:
        return -1
    else:
        return 0
    

def generate_trades_dfs(strat_dict, initial_df, simulations_long, simulations_short):
    trades_dfs = {}
    
    for key in strat_dict.keys():
        dfs_to_combine = []
        
        for index, row in initial_df.iterrows():
            date = row['date']
            trade = row[key]
            iv_diff = row['IV_diff']
            
            if trade == 1 and date in simulations_long:
                df_to_add = simulations_long[date].copy()
            elif trade == -1 and date in simulations_short:
                df_to_add = simulations_short[date].copy()
            else:
                # Skip if 'trade' is 0 or the date is not in the dictionaries
                continue
            
            # Add 'trade' & 'IV_diff' column
            df_to_add['IV_diff'] = iv_diff  # Needed for position calculation
            df_to_add[key] = trade  # Include the 'trade' value
            dfs_to_combine.append(df_to_add)
        
        # Concatenate all collected DataFrames
        trades_dfs[key] = pd.concat(dfs_to_combine, ignore_index=True)
        trades_dfs[key] = trades_dfs[key].sort_values(by=['date', 'exdate', 'strike_price', 'to_open']).reset_index(drop=True)

    return trades_dfs

def preprocess_options(options_df):
    options_df['UID'] = options_df['strikeID'] + '_' + options_df['date'].dt.date.astype(str)
    volumes = options_df[['date', 'volume_c', 'volume_p', 'adj_volume', 'UID']].copy()
    volumes['date'] = volumes['date'].dt.strftime('%Y-%m-%d')
    volumes['volume_med'] = (volumes['volume_c'] + volumes['volume_p']) / 2
    return volumes

def pos_size(IV_diff, strike_price, option_cost_basis, UID, key, volumes_df, trades_dfs):
    volume = min(volumes_df.loc[volumes_df['UID'] == UID, 'volume_med'].item(), 50)
    factor = max(volume * strike_price / 10, 1)  # Ensure factor is not zero

    if option_cost_basis == 0:
        filtered_df = trades_dfs[key].loc[trades_dfs[key]['UID'] == UID, 'option_cost_basis']
        option_cost_basis = filtered_df.iloc[0] if not filtered_df.empty else 0

    return round(abs(IV_diff) / abs(option_cost_basis) * factor) if option_cost_basis != 0 else 0

def update_trades_with_pos_size(trades_dfs, volumes):
    for key, df in trades_dfs.items():
        df = df.drop(columns=[col for col in df.columns if col.endswith('_p') or col.endswith('_c')]).copy()

        df['pos_size'] = df.apply(lambda row: pos_size(row['IV_diff'], row['strike_price'], row['option_cost_basis'], row['UID'], key, volumes, trades_dfs), axis=1)
        lot_size = 100 * df['pos_size']

        for col in ['stock_pos', 'pos_change', 'change_cost_basis', 'stock_cost_basis', 'daily_stock_value', 'stock_PL', 'option_cost_basis',
                    'change_cost_basis_op', 'daily_option_value', 'option_PL', 'total_cost_basis', 'total_pos_value', 'total_PL', 'realized_stock_PL',
                    'realized_option_PL', 'realized_PL']:
            df['sized_' + col] = lot_size * df[col]
        
        trades_dfs[key] = df
    
    return trades_dfs


# ============================================

# dataframe metrics

def summarize_pl_by_date(trades_dfs, trading_days):
    PL_temp_dfs = {}
    for key, df in trades_dfs.items():
        columns_to_sum = ['sized_' + col for col in ['stock_pos', 'change_cost_basis', 'stock_cost_basis', 'daily_stock_value', 'stock_PL', 'option_cost_basis', 'change_cost_basis_op',
                        'daily_option_value', 'option_PL', 'total_cost_basis', 'total_pos_value', 'total_PL', 'realized_stock_PL', 'realized_option_PL', 'realized_PL']]

        grouped_df = df[['date'] + columns_to_sum].groupby('date').sum().reset_index()
        pl_df = grouped_df.set_index('date').reindex(trading_days).fillna(0).reset_index()
        pl_df.rename(columns={'index': 'date'}, inplace=True)
        PL_temp_dfs[key] = pl_df
    
    return PL_temp_dfs

def calculate_dividends(PL_temp_dfs, spy_divdata):
    divvies = {}
    for key, df in PL_temp_dfs.items():
        df['date'] = pd.to_datetime(df['date'])
        df['pay_date']  = pd.to_datetime(df['pay_date'])
        temp_merged = pd.merge(spy_divdata, df[['date', 'sized_stock_pos']], how='left', on='date')
        temp_merged['div'] = temp_merged['sized_stock_pos'] * temp_merged['dividend']
        divvies[key] = temp_merged
    
    return divvies

def merge_dividends_with_pl(PL_temp_dfs, divvies):
    for key, pl_df in PL_temp_dfs.items():
        div_df = divvies[key]
        pl_df['date'] = pd.to_datetime(pl_df['date'])
        merged_df = pd.merge(pl_df, div_df[['pay_date', 'div']], how='left', left_on='date', right_on='pay_date')
        merged_df.drop(columns=['pay_date'], inplace=True)
        merged_df['div'] = merged_df['div'].fillna(0)
        merged_df['sized_realized_stock_PL'] += merged_df['div']
        merged_df['sized_realized_PL'] += merged_df['div']
        PL_temp_dfs[key] = merged_df
    
    return PL_temp_dfs


def process_tbills_data(tbill_data_path, start_date, end_date, trading_days):
    tbills_data = pd.read_csv(tbill_data_path)[['CALDT', 'TDDURATN', 'TMATDT', 'TDNOMPRC']].sort_values(by=['CALDT', 'TDDURATN']).reset_index(drop=True)
    tbills_data = tbills_data.rename(columns={
        'TMATDT': 'maturity_date',
        'CALDT': 'date',
        'TDNOMPRC': 'price',
        'TDDURATN': 'dte'
    })
    tbills_data['maturity_date'] = pd.to_datetime(tbills_data['maturity_date'])
    tbills_data['date'] = pd.to_datetime(tbills_data['date'])
    tbills_data = tbills_data.loc[(tbills_data['date'] >= start_date) & (tbills_data['date'] <= end_date)].copy().reset_index(drop=True)
    tbills_data = tbills_data.drop_duplicates(subset='date', keep='first').reset_index(drop=True)
    tbills_data = tbills_data[tbills_data['date'].isin(trading_days)].copy().reset_index(drop=True)
    tbills_data['rate'] = (100 / tbills_data['price']) ** (1 / tbills_data['dte']) - 1
    tbills_data['leverage_rate'] = ((tbills_data['rate'] + 1) ** 365 + 25 / 100 / 100) ** (1 / 365) - 1  # 25 bps to loan anything (leverage)
    
    return tbills_data

def calculate_rfr(trading_days, tbills_data):
    if not isinstance(trading_days, pd.DataFrame):
        trading_days_df = pd.DataFrame({'date': trading_days})
    else:
        trading_days_df = trading_days.copy()

    trading_days_df['date'] = pd.to_datetime(trading_days_df['date'])
    tbills_data['date'] = pd.to_datetime(tbills_data['date'])

    rfr = trading_days_df.merge(tbills_data[['date', 'rate', 'leverage_rate']], on='date', how='left', sort=True)

    rfr['rate'] = rfr['rate'].ffill()
    rfr['leverage_rate'] = rfr['leverage_rate'].ffill()

    return rfr

# ============================================

# ALL METRICS - CORRESPONDING TO "Final Dataframes"

# Function calculates all relevant metrics for our trading analysis
# Due to the smaller data, we went for a less vectorized approach here to get more metrics easily.

def process_pl_dfs(PL_temp_dfs, rfr, INITIAL, KAPITAL):
    PL_dfs = {}
    
    for key, df in PL_temp_dfs.items():
        pl_df = pd.DataFrame(index=df.index)
        pl_df['date'] = df['date']

        # Trading costs and values
        pl_df['gross_stock_trades'] = abs(df['sized_change_cost_basis'])
        pl_df['gross_option_trades'] = abs(df['sized_change_cost_basis_op'])
        pl_df['gross_trades_value'] = pl_df['gross_stock_trades'] + pl_df['gross_option_trades']
        pl_df['stock_trading_costs'] = 1/100/100 * pl_df['gross_stock_trades']
        pl_df['option_trading_costs'] = 1/100/100 * pl_df['gross_option_trades']
        pl_df['net_trading_costs'] = 1/100/100 * pl_df['gross_trades_value']

        # Position values
        pl_df['stock_pos_value'] = df['sized_daily_stock_value']
        pl_df['option_pos_value'] = df['sized_daily_option_value']
        pl_df['gross_pos_value'] = pl_df['stock_pos_value'] + pl_df['option_pos_value']

        # Realized P&L calculations
        real_stock_PL = df['sized_realized_stock_PL'] - pl_df['stock_trading_costs']
        real_option_PL = df['sized_realized_option_PL'] - pl_df['option_trading_costs']
        real_net_PL = df['sized_realized_PL'] - pl_df['net_trading_costs']
        pl_df['stock_PL'] = real_stock_PL.cumsum()
        pl_df['option_PL'] = real_option_PL.cumsum()
        pl_df['net_PL'] = real_net_PL.cumsum()

        # Initial cash and capital calculations
        pl_df['start_cash'] = 0.0
        pl_df['initial_kapital'] = INITIAL
        pl_df['short_fee'] = 0.0
        pl_df['initial_cash'] = 0.0
        pl_df['interest'] = 0.0
        pl_df['lever_cash'] = 0.0
        pl_df['leverage_fee'] = 0.0
        pl_df['end_kapital'] = 0.0
        pl_df.loc[0, 'start_cash'] = INITIAL

        # Iterate over each day to calculate fees and interest
        for i in range(0, len(pl_df)):
            if i > 0:
                pl_df.loc[i, 'start_cash'] = pl_df.loc[i - 1, 'end_kapital']
            pl_df.loc[i, 'short_fee'] = - min(0, df.loc[i, 'sized_daily_stock_value']) * rfr.loc[i, 'leverage_rate']
            pl_df.loc[i, 'initial_kapital'] = pl_df.loc[i, 'start_cash'] + real_net_PL[i] - pl_df.loc[i, 'short_fee']
            pl_df.loc[i, 'initial_cash'] = max(pl_df.loc[i, 'initial_kapital'] - df.loc[i, 'sized_total_cost_basis'], 0)
            pl_df.loc[i, 'interest'] = pl_df.loc[i, 'initial_cash'] * rfr.loc[i, 'rate']
            pl_df.loc[i, 'lever_cash'] = max(df.loc[i, 'sized_total_cost_basis'] - pl_df.loc[i, 'initial_kapital'], 0)
            pl_df.loc[i, 'leverage_fee'] = pl_df.loc[i, 'lever_cash'] * rfr.loc[i, 'leverage_rate']
            pl_df.loc[i, 'end_kapital'] = pl_df.loc[i, 'initial_kapital'] + pl_df.loc[i, 'interest'] - pl_df.loc[i, 'leverage_fee']

        # Cumulative fees and interest
        pl_df['net_short_fees'] = pl_df['short_fee'].cumsum()
        pl_df['net_interest_paid'] = pl_df['interest'].cumsum()
        pl_df['net_interest_earned'] = pl_df['leverage_fee'].cumsum()

        # Final position value and total cash calculations
        pl_df['net_pos_value'] = pl_df['end_kapital'] - df['sized_total_cost_basis'] + pl_df['gross_pos_value']
        pl_df['tot_cash'] = KAPITAL - df['sized_total_cost_basis'] + pl_df['net_PL'] - pl_df['net_interest_paid'] + pl_df['net_interest_earned']

        PL_dfs[key] = pl_df

    return PL_dfs

# ============================================

# Original calculate PL function for mass simulation
# NOT USED LATER - for doc purposes only (in the appendix)

def calculate_realized_PL(df, long_op=True):
    df = df.reset_index(drop=True)  
    df['stock_pos'] = df['shares_held'] if long_op else -df['shares_held']
    df['avg_cost'] = np.nan
    df['realized_PL'] = 0.0
    df['option_PL'] = 0.0
    df['stock_PL'] = 0.0  

    initial_option_value = df.loc[0, 'best_offer_c'] + df.loc[0, 'best_offer_p'] if long_op else -df.loc[0, 'best_bid_c'] - df.loc[0, 'best_bid_p']

    for i in range(len(df)):
        close_price = df.loc[i, 'close']
        df.loc[i, 'option_PL'] = (df.loc[i, 'best_bid_c'] + df.loc[i, 'best_bid_p'] - initial_option_value) if long_op else (-df.loc[i, 'best_offer_c'] - df.loc[i, 'best_offer_p'] - initial_option_value)

        if i == 0:
            df.loc[i, 'avg_cost'] = close_price
            continue

        prev_pos = df.loc[i - 1, 'stock_pos']
        current_pos = df.loc[i, 'stock_pos']
        pos_change = current_pos - prev_pos

        if not pd.isna(df.loc[i - 1, 'avg_cost']):
            df.loc[i, 'stock_PL'] = df.loc[i - 1, 'realized_PL'] + (close_price - df.loc[i - 1, 'avg_cost']) * prev_pos

        daily_option_value = df.loc[i, 'best_bid_c'] + df.loc[i, 'best_bid_p'] if long_op else -df.loc[i, 'best_offer_c'] - df.loc[i, 'best_offer_p']
        df.loc[i, 'option_PL'] = daily_option_value - initial_option_value

        if pos_change != 0:
            if np.sign(pos_change) == np.sign(prev_pos) or prev_pos == 0:
                total_shares = abs(prev_pos) + abs(pos_change)
                total_cost = df.loc[i - 1, 'avg_cost'] * abs(prev_pos) + close_price * abs(pos_change)
                df.loc[i, 'avg_cost'] = total_cost / total_shares if total_shares != 0 else close_price
            else:
                closed_shares = min(abs(prev_pos), abs(pos_change))
                realized_PL_this_step = (close_price - df.loc[i - 1, 'avg_cost']) * closed_shares * np.sign(prev_pos)
                df.loc[i, 'realized_PL'] = df.loc[i - 1, 'realized_PL'] + realized_PL_this_step
                if abs(pos_change) > abs(prev_pos):
                    excess_shares = abs(pos_change) - abs(prev_pos)
                    df.loc[i, 'avg_cost'] = close_price
                    df.loc[i, 'stock_pos'] = excess_shares * np.sign(pos_change)
                else:
                    df.loc[i, 'avg_cost'] = np.nan
        else:
            df.loc[i, 'avg_cost'] = df.loc[i - 1, 'avg_cost']
            df.loc[i, 'stock_pos'] = prev_pos

        df['avg_cost'].ffill(inplace=True)

    final_row_index = len(df) - 1
    final_pos = df.loc[final_row_index, 'stock_pos']
    final_close_price = df.loc[final_row_index, 'close']
    final_avg_cost = df.loc[final_row_index, 'avg_cost']
    final_realized_PL = (final_close_price - final_avg_cost) * final_pos
    df.loc[final_row_index, 'realized_PL'] += final_realized_PL
    df.loc[final_row_index, 'stock_PL'] = df.loc[final_row_index, 'realized_PL']
    df.loc[final_row_index, 'stock_pos'] = 0

    return df


# ============================================
# ============================================
# ============================================
# ============================================

