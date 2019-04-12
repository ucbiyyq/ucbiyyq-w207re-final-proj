import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

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
    def __init__(self
                 , holiday_calendar = USFederalHolidayCalendar()
                 , not_latenight = (7, 20)
                 , geo_fence = (-122.51365, -122.25000, 37.70787, 50.00000)
                 ):
        """
        constructor for the SFCCTransformer
        
        can pass in some params:
            holiday_calendar, which holiday calendar to use for is_holiday; default US Federal
            not_latenight, hour of day that is NOT late night; default 7 am to 8 pm
            geo_fence, sf city limits, any record with X or Y outside limit has X and Y set to NA, so that they can be imputed later on; min long X, max long X, min lat Y, max lat Y; default (-122.51365, -122.25000, 37.70787, 50.00000)
        """
        self.holiday_calendar = holiday_calendar
        self.not_latenight = not_latenight
        self.geo_fence = geo_fence
        
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
        
        def calc_is_latenight(dtt):
            hrs = dtt.dt.hour
            res = np.ones(shape = hrs.shape)
            res[(hrs > self.not_latenight[0]) & (hrs < self.not_latenight[1])] = 0
            res = res.astype("int")
            return res
        
        # creates a copy of the input dataframe
        X_out = X.copy()
        
        # sets any X or Y that is outside geo fence limits to NaN, for later imputation
        X_out.loc[(X_out["X"] < self.geo_fence[0]) | (X_out["X"] > self.geo_fence[1]) | (X_out["Y"] < self.geo_fence[2]) | (X_out["Y"] > self.geo_fence[3]), ["X", "Y"]] = np.nan
        
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
        X_out["is_latenight"] = calc_is_latenight(dtt) # 1 if after 8 pm and before 6 am, 0 otherwise
        
        # calculate cyclical values for hours, etc
        # http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
        X_out["hour_of_day_sin"] = np.round( np.sin(dtt.dt.hour * (2. * np.pi / 24)), 3)
        X_out["hour_of_day_cos"] = np.round( np.cos(dtt.dt.hour * (2. * np.pi / 24)), 3)
        
        X_out["day_of_week_sin"] = np.round( np.sin(dtt.dt.dayofweek * (2. * np.pi / 7)), 3)
        X_out["day_of_week_cos"] = np.round( np.cos(dtt.dt.dayofweek * (2. * np.pi / 7)), 3)
        
        X_out["day_of_month_sin"] = np.round( np.sin(dtt.dt.dayofweek * (2. * np.pi / 31)), 3)
        X_out["day_of_month_cos"] = np.round( np.cos(dtt.dt.dayofweek * (2. * np.pi / 31)), 3)
        
        X_out["day_of_year_sin"] = np.round( np.sin(dtt.dt.dayofweek * (2. * np.pi / 366)), 3)
        X_out["day_of_year_cos"] = np.round( np.cos(dtt.dt.dayofweek * (2. * np.pi / 366)), 3)
        
        X_out["week_of_year_sin"] = np.round( np.sin(dtt.dt.dayofweek * (2. * np.pi / 53)), 3)
        X_out["week_of_year_cos"] = np.round( np.cos(dtt.dt.dayofweek * (2. * np.pi / 53)), 3)
        
        X_out["month_of_year_sin"] = np.round( np.sin((dtt.dt.month - 1) * (2. * np.pi / 12)), 3)
        X_out["month_of_year_cos"] = np.round( np.cos((dtt.dt.month - 1) * (2. * np.pi / 12)), 3)
        
        X_out["quarter_of_year_sin"] = np.round( np.sin((dtt.dt.month - 1) * (2. * np.pi / 4)), 3)
        X_out["quarter_of_year_cos"] = np.round( np.cos((dtt.dt.month - 1) * (2. * np.pi / 4)), 3)
        
        # TODO calculate address-based features, such as street, intersection, etc
        # might be better to do this in prep_data with CountVectorizer or something
        
        # TODO calculate distance from hotspots of crime
        
        return X_out

def shuffle_split_data(df, is_test = False):
    """
    Helper function to shuffle and split train data into a Dataframe of data, and a Series of labels
    
    Can also be used for the test data if the is_test flag is set to true
    
    i.e.
        1. given train_pd or test_pd
        2. shuffles dataframe
        3. splits into data (and labels)
    """
    shuffled = df.sample(frac = 1, random_state = 0)
    data = shuffled
    labels = pd.Series([])
    if is_test:
        pass
    else:
        labels = data["Category"]
        data = data.drop(["Category", "Descript", "Resolution"], axis = 1)
    return data, labels

def prep_data(data, feat_nums, feat_cat, feat_binary, feat_text):
    """
    Helper function to prepare the train and test data. 
    Uses the other classes in this module, along with pipelines, to convert features.
    
    params
        feat_nums, list of numeric feature names, i.e. X, Y, hour_of_day_sin, hour_of_day_cos ...
        feat_cat, list of categorical feature names, i.e. DayOfWeek, PdDistrict, ...
        feat_binary, list of binary feature names, i.e. is_weekend, is_latenight, ...
        feat_text, list of text feature names, i.e. Address
    
    i.e.
        1. extracts basic features from data, with SFCCTransformer
        2. splits data into numeric, categorical, binary, and text features
        3. for numerics, runs through imputer and scaler
        4. for categorical, runs through onehot encoding
        5. for binary, doesn't do anything extra
        6. for text, converts to term frequency (TODO)
        7. then unions everything into one dataframe
    """
    
    # extracts some new basic attributes from the existing attributes
    sfcc = SFCCTransformer()
    pipe = Pipeline([
        ("transformer", sfcc)
    ])
    data = pipe.transform(data)
    
    # splits into numeric, categorical, text dataframes, so we that can feed them through different pipelines
    
    # feeds numeric features into pipeline that has
    # SimpleImputer (median), to fill in any missing values (esp the X and Y)
    # and MinMaxScaler so that they will have similar scale to our other features
    imputer = SimpleImputer(missing_values = np.nan, strategy = "median")
    scaler = MinMaxScaler()
    pipe_num = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler)
    ])
    data_num = data[feat_nums]
    data_num_out = pipe_num.fit_transform(data_num)
    data_num_out = pd.DataFrame(data_num_out, columns = data_num.columns)
    
    # feeds categorical features into pipeline, which has
    # OneHotEncoder, which will turn the categorical features into 1 or 0 per level
    one_hot = OneHotEncoder(sparse = False)
    pipe_cat = Pipeline([
        ("encoder", one_hot)
    ])
    data_cat = data[feat_cat]
    data_cat_out = pipe_cat.fit_transform(data_cat)
    data_cat_cols = np.concatenate(one_hot.categories_).ravel().tolist()
    data_cat_out = pd.DataFrame(data_cat_out, columns = data_cat_cols)
    
    # don't need to do anything to prepare the binary features
    data_binary_out = data[feat_binary]
    
    # feeds text features into pipeline, which as
    # TODO find out how to use count vectorizer here
#     feat_text

    result = pd.concat([data_num_out, data_cat_out, data_binary_out], axis = 1, sort = False)
    
    return result

def prep_submissions(predsproba, categories):
    """
    Helper function to prepare the raw predsproba array into a panda with the correct column headers and an index
    """
    cols = np.sort(pd.unique(categories))
    submissions = pd.DataFrame(data = predsproba, columns = cols)
    
    # rounds any floats to less precision
    submissions= submissions[cols].round(2)
    
    # adds an Id column
    idx = np.arange(0, len(predsproba))
    submissions.insert(loc = 0, column = "Id", value = idx.tolist())
    return(submissions)

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
            , "is_latenight"
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
            , "day_of_month_sin"
            , "day_of_month_cos"
            , "day_of_year_sin"
            , "day_of_year_cos"
            , "week_of_year_sin"
            , "week_of_year_cos"
            , "month_of_year_sin"
            , "month_of_year_cos"
            , "quarter_of_year_sin"
            , "quarter_of_year_cos"
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
        