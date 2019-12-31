# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:01:34 2018

@author: WJDT
"""

import numpy as np
import pandas as pd

# Data Importing & pre-Processing
dff = pd.read_excel('KSData1601-1812.xlsx')
Index = pd.read_excel('^KS11-1601-1812.xlsx')
dff.index = dff['Date']
del dff['Date']
dff.index = pd.to_datetime(dff.index)
Index.index = Index['Date']
del Index['Date']
Index.index = pd.to_datetime(Index.index)

# return weekday
def getDayName(a):
    daystring = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    return daystring[a]

# Data Processing... w/ continuous compounding...
df_daily_ret = np.log(dff/dff.shift(1)).dropna()

"""
df_weekly_ret = df_daily_ret.resample('W-{}'.format(getDayName(dff.index[0].weekday())), how = {np.sum})
df_weekly_ret = np.exp(df_weekly_ret) - 1
"""
df_monthly_ret = df_daily_ret.resample('M', how = {np.sum})
df_monthly_ret = np.exp(df_monthly_ret) - 1


df_Index_daily = np.log(Index['Adj Close']/Index['Adj Close'].shift(1)).dropna()
"""
df_Index_weekly = df_Index_daily.resample('W-{}'.format(getDayName(dff.index[0].weekday())), how = {np.sum})
df_Index_weekly = np.exp(df_Index_weekly) - 1
"""
df_Index_monthly = df_Index_daily.resample('M', how = {np.sum})
df_Index_monthly = np.exp(df_Index_monthly) - 1


# Market Premium Generator
#MktPre_wk = np.array(df_Index_weekly).reshape(-1)
MktPre = np.array(df_Index_monthly).reshape(-1) 


leng = 12 # 13 wk as quater / 26 wk as 2 quater / 6 mo as 2 quater / 12 mo as 1 yr
leng_st = leng * 2 # note, multiplier = (construction point - initial data point) / leng


for i in range(leng): 
    # using CAPM..
    """
    Features = np.vstack([np.ones(leng), MktPre_wk[leng_st-2*leng+i:leng_st-leng+i]])
    BaseMat = np.dot(Features,Features.T)
    BaseMat_1 = np.linalg.inv(BaseMat)
    BetaSet = np.dot(np.dot(BaseMat_1, Features), df_weekly_ret[leng_st-2*leng+i:leng_st-leng+i])
    err = df_weekly_ret[leng_st-2*leng+i:leng_st-leng+i] - np.dot(Features.T, BetaSet)
    err2s = np.diagonal(np.dot(err.T,err))
    TSS = np.array(np.sum((df_weekly_ret[leng_st-2*leng+i:leng_st-leng+i] - np.sum(df_weekly_ret[leng_st-2*leng+i:leng_st-leng+i])/leng)**2))
    varhat = err2s/(len(Features.T)-len(Features))
    """
    Features = np.vstack([np.ones(leng), MktPre[leng_st-2*leng+i:leng_st-leng+i]])
    BaseMat = np.dot(Features,Features.T)
    BaseMat_1 = np.linalg.inv(BaseMat)
    BetaSet = np.dot(np.dot(BaseMat_1, Features),df_monthly_ret[leng_st-2*leng+i:leng_st-leng+i])
    err = df_monthly_ret[leng_st-2*leng+i:leng_st-leng+i] - np.dot(Features.T, BetaSet)
    err2s = np.diagonal(np.dot(err.T,err))
    TSS = np.array(np.sum((df_monthly_ret[leng_st-2*leng+i:leng_st-leng+i] - np.sum(df_monthly_ret[leng_st-2*leng+i:leng_st-leng+i])/leng)**2))
    varhat = err2s/(len(Features.T)-len(Features))
    

    # STATs decription
    residual_CAPM = pd.DataFrame(err2s,  columns = ['residual score']).T
    JenAlphadf = pd.DataFrame(BetaSet[0], columns = ['JAlpha']).T
    MktBetadf = pd.DataFrame(BetaSet[1], columns = ['MktBeta']).T
    
    if i == 0:
        A = JenAlphadf
        B = MktBetadf
        R = residual_CAPM
    else:
        A = pd.merge(A, JenAlphadf, how = 'outer')
        B = pd.merge(B, MktBetadf, how = 'outer')
        R = pd.merge(R, residual_CAPM, how = 'outer')

A.columns, B.columns, R.columns = dff.columns, dff.columns, dff.columns

