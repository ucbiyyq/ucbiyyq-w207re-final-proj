import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.tseries.holiday import USFederalHolidayCalendar

"""
Module of helper classes and functions to build features.

Inspired by:
    
    Hands-On Machine Learning with Scikit-Learn and TensorFlow: 
    Concepts, Tools, and Techniques to Build Intelligent Systems
    
    by Géron, Aurélien, O'Reilly Media.
    
"""

class DataFrameSelector(BaseEstimator, TransformerMixin): 
    """
    Simple helper class, meant make it easier to use Pandas 
    along with sklearn Pipeline. Create and initate with a 
    list of features, then when the pipeline transform function
    is called, will return a Numpy array of the features.
    
    See Chap 2 transformation pipelines
    
    Example:
        train_pd = pd.read_csv("data.csv")
        num_features = ["X", "Y"]
        num_pipeline = Pipeline([
            ("selector", DataFrameSelector(num_features))
        ])
        train_prepared = num_pipeline.transform(train_pd)
        
    """
    def __init__(self, attribute_names): 
        self.attribute_names = attribute_names 
        
    def fit(self, X, y = None): 
        return self 
    
    def transform(self, X): 
        return X[self.attribute_names].values

class MyTransformer(BaseEstimator, TransformerMixin):
    """
    Silly test class
    
    """
    def __init__(self):
        pass # nothing to init
        
    def fit(self, X, y = None):
        return self # no fitting
    
    def transform(self, X, y = None):
        def silly(df):
            return df.X + df.Y
        X["Z"] = silly(X)
        return X
    
class SFCCTransformer(BaseEstimator, TransformerMixin):
    """
    Helper class for our SanFrancisco Crime Classification project.
    
    Centralizes transformation logic, and make it easier to use
    transformations with Pandas, Pipeline, and gscv. Note, meant to transform
    Pandas into Pandas.
    
    Should use in conjunction with DataFrameSelector and one hot encoders.
    
    See Chap 2 custom transformers
    
    """
    def __init__(self, holiday_calendar = USFederalHolidayCalendar(), latitude_outlier = 50):
        self.holiday_calendar = holiday_calendar
        self.latitude_outlier = latitude_outlier
        
    def fit(self, X, y = None):
        return self # no fitting
    
    def transform(self, X, y = None):
        
        def add_delta(dtt, delta):
            """
            helper funciton, given a Series of dates, 
            returns Series of delta since the mininum date
            
            see Linda's baseline code
            """
            res = (dtt - dtt.min()) / np.timedelta64(1, delta)
            res = np.floor(res).astype("int")
            return res
        
        def calc_is_holiday(dtt):
            """
            Helper function, given Series dates, 
            returns Series of 1 if date is holiday, 0 otherwise
            
            https://stackoverflow.com/questions/29688899/pandas-checking-if-a-date-is-a-holiday-and-assigning-boolean-value
            """
            dt = dtt.dt.date
            holidays = self.holiday_calendar.holidays(start = dt.min(), end = dt.max()).date
            res = dt.isin(holidays).astype("int")
            return res
        
        # creates a copy of the input dataframe
        X_out = X.copy()
        
        # extracts various date-related features
        dtt = pd.to_datetime(X_out.Dates)
        
        X_out["hour_delta"] = add_delta(dtt, "h") # hour since start, 0 to 108263
        X_out["day_delta"] = add_delta(dtt, "D") # day since start, 0 to 4510
        X_out["week_delta"] = add_delta(dtt, "W") # week since start, 0 to 644
        X_out["month_delta"] = add_delta(dtt, "M") # month since start, 0 to 148
        X_out["year_delta"] = add_delta(dtt, "Y") # year since start, 0 to 12
        
        X_out["hour_of_day"] = dtt.dt.hour # 0 to 23
        X_out["day_of_week"] = dtt.dt.dayofweek # 0 to 7, note day name is already DayOfWeek
        X_out["day_of_month"] = dtt.dt.day # 1 to 31
        X_out["day_of_year"] = dtt.dt.dayofyear # 1 to 365
        X_out["week_of_year"] = dtt.dt.week # 2 to 52
        X_out["month_of_year"] = dtt.dt.month # 1 to 12
        X_out["quarter_of_year"] = dtt.dt.quarter # 1 to 4
        X_out["year"] = dtt.dt.year # 2003 to 2015
        
        X_out["is_weekend"] = dtt.dt.dayofweek // 5 # 1 if sat or sun, 0 otherwise
        X_out["is_holiday"] = calc_is_holiday(dtt) # 1 if holiday, 0 otherwise
        
        # calculate cyclical values for hours, etc
        # http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
        X_out["hour_of_day_sin"] = np.round( np.sin(dtt.dt.hour * (2. * np.pi / 24)), 3)
        X_out["hour_of_day_cos"] = np.round( np.cos(dtt.dt.hour * (2. * np.pi / 24)), 3)
        
        X_out["day_of_week_sin"] = np.round( np.sin(dtt.dt.dayofweek * (2. * np.pi / 7)), 3)
        X_out["day_of_week_cos"] = np.round( np.cos(dtt.dt.dayofweek * (2. * np.pi / 7)), 3)
        
        X_out["month_of_year_sin"] = np.round( np.sin((dtt.dt.month - 1) * (2. * np.pi / 12)), 3)
        X_out["month_of_year_cos"] = np.round( np.cos((dtt.dt.month - 1) * (2. * np.pi / 12)), 3)
        
        # TODO calculate police shifts? apparently its not regularly-spaced shifts
        
        # TODO calculate address-based features, such as street, intersection, etc
        
        # TODO calculate distance from hotspots of crime
        
        return X_out

def print_summary(df):
    """
    Helper function to simply print summaries of all the variables in the dataset
    
    """
    # to summarize by counting
    cats = ["Category"
            , "hour_of_day"
            , "day_of_week"
            , "DayOfWeek"
            , "day_of_month"
            , "day_of_year"
            , "week_of_year"
            , "month_of_year"
            , "quarter_of_year"
            , "year"
            , "is_weekend"
            , "is_holiday"
            , "PdDistrict"
            ]
    
    # to summarize by describe
    nums = ["X", "Y"
            , "Dates"
            , "hour_delta"
            , "day_delta"
            , "week_delta"
            , "month_delta"
            , "year_delta"
            , "hour_of_day"
            , "day_of_week"
            , "day_of_month"
            , "day_of_year"
            , "week_of_year"
            , "month_of_year"
            , "quarter_of_year"
            , "year"
            , "hour_of_day_sin"
            , "hour_of_day_cos"
            , "day_of_week_sin"
            , "day_of_week_cos"
            , "month_of_year_sin"
            , "month_of_year_cos"
            ] 
    
    # messy, can't really summarize, so just print a sample
    miscs = ["Descript", "Resolution", "Address"]
    
    print("============")
    print(df.info())
    
    for c in cats:
        print("============")
        print(c)
        print("------------")
        try:
            print(df[c].value_counts())
        except KeyError:
            print(c, "not in dataframe")
    
    print("============")
    print(df[nums].describe())
    
    for m in miscs:
        print("============")
        print(m)
        print("------------")
        try:
            print(df[m].sample(3, random_state = 0))
        except KeyError:
            print(m, "not in dataframe")
        