# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 20:46:31 2018

@author: WJDT
"""
import numpy as np
import pandas as pd
import datetime as dt
import fix_yahoo_finance as yf

# DATA INPUT
start_date = dt.datetime(2016, 1, 1)
end_date = dt.datetime(2018, 12, 31)
codes = pd.read_csv('KSCode_for_python.csv', engine = 'python') # Tickers Importing
Index = yf.download('^KS11', start_date, end_date)
tickers = []

# Tickers Listing
for i in range(len(codes['0'])):
    tickers.append(codes['0'][i])

errors = []
ignored = 0
v_e = 0

print('1 of {} has been downloaded'.format(len(tickers)+1))
print('({}, {}, {})'.format(1, ignored, v_e))

# Market Data Download
for i in range(len(tickers)):
    if i == 0:
        w = yf.download(tickers[i], start_date, end_date)
        x = np.array(w['Close'])
        lengst = len(x)
        print('2 of {} has been downloaded'.format(len(tickers)+1))
        print('({}, {}, {})'.format(2, ignored, v_e))
    else:
        try:
            w = yf.download(tickers[i], start_date, end_date)
            if len(w) == lengst:
                x = np.vstack([x,w['Close']])
                print('{} of {} has been downloaded'.format(i+2,len(tickers)+1))
                print('({}, {}, {})'.format(i+2 - ignored - v_e, ignored, v_e))
            else:
                
                print('{} of {} has been ignored by constraints'.format(i+2,len(tickers)+1))
                errors.append(i)
                ignored += 1
                print('({}, {}, {})'.format(i+2 - ignored - v_e, ignored, v_e))
        except ValueError:
            print('{} of {} has been ignored by ValueError'.format(i+2,len(tickers)+1))
            errors.append(i)
            v_e += 1
            print('({}, {}, {})'.format(i+2 - ignored - v_e, ignored, v_e))
            pass

# Error Counting    
error_adjusted_tickers = np.delete(tickers, errors)
error_counted = len(errors) - ignored

# Data Exporting as 'Closed Price'
dff = pd.DataFrame(x.T)
dff.index = w.index
dff.columns = error_adjusted_tickers
dff.to_excel('KSData1601-1812.xlsx')

Index.to_excel('^KS11-1601-1812.xlsx')