import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

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
                 , imputer = SimpleImputer(missing_values = np.nan, strategy = "median")
                 , scaler = MinMaxScaler()
                 , holiday_calendar = USFederalHolidayCalendar()
                 , not_latenight = (7, 20)
                 , geo_fence = (-122.51365, -122.25000, 37.70787, 50.00000)
                 , geo_strategy = "imputer"
                 
                 , pddistrict_drop = True
                 , dayofweek_drop = True
                 , dates_drop = True
                 , address_drop = True
                 
                 , pddistrict_onehot = True
                 , dayofweek_onehot = True
                 , dates_deltas = True
                 , dates_binary = True
                 , dates_ordinal = False
                 , dates_onehot = True
                 , dates_cyclical = True
                 , address_vectorize = True
                 ):
        """
        constructor for the SFCCTransformer
        
        params:
            imputer
                what kind of imputer to use for any missing values in numeric features
                default SimpleImputer(np.NaN, "median")
            scaler
                what kind of scaler to use for numeric features
                default MinMaxScaler
            holiday_calendar
                which holiday calendar to use for is_holiday
                default US Federal Holidays
            not_latenight
                hour of day that is NOT late night
                default 7 am to 8 pm
            geo_fence
                sf city limits 
                min long X, max long X, min lat Y, max lat Y
                default (-122.51365, -122.25000, 37.70787, 50.00000)
            geo_strategy
                how to handle records with X or Y outside of the geo_fence
                identity = keep the X or Y values the same
                imputer = uses the imputer to fill in, i.e. median
                centroid = sets as centroid of all crime within fence
                default imputer
                
            pddistrict_drop
                if true, drops the PdDistrict feature after it has been converted
                default true
            dayofweek_drop
                if true, drops the DayOfWeek feature after it has been converted
                default true
            dates_drop
                if true, drops the Dates feature after it has been converted
                default true
            address_drop
                if true, dops the Address feature after it has been converted
                default true
                
            pddistrict_onehot
                if true, makes the PdDistrict onehot encoded, otherwise PdDistrict is not transformed 
                default true
            dayofweek_onehot
                if true, makes the DayOfWeek onehot encoded, otherwise DayOfWeek is not transformed
                default true
            dates_deltas
                if true, includes various date-as-deltas features
                default true
            dates_binary
                if true, includes various date-as-binary features
                default true
            dates_ordinal
                if true, includes various date-as-ordinal features
                default false
            dates_onehot
                if true, include various date-related features as one-hot encoded
                default true
            dates_cyclical
                if true, include various date-related features as cyclical features, e.g. sin and cos
                default true

        """
        self.imputer = imputer
        self.scaler = scaler
        self.holiday_calendar = holiday_calendar
        self.not_latenight = not_latenight
        self.geo_fence = geo_fence
        self.geo_strategy = geo_strategy
        
        self.pddistrict_drop = pddistrict_drop
        self.dayofweek_drop = dayofweek_drop
        self.dates_drop = dates_drop
        self.address_drop = address_drop
        
        self.pddistrict_onehot = pddistrict_onehot
        self.dayofweek_onehot = dayofweek_onehot
        self.dates_deltas = dates_deltas
        self.dates_binary = dates_binary
        self.dates_ordinal = dates_ordinal
        self.dates_onehot = dates_onehot
        self.dates_cyclical = dates_cyclical
        self.address_vectorize = address_vectorize
        
        
    def fit(self, X, y = None):
        return self # no fitting
    
    def transform(self, X, y = None):
        """
        Creates new features
        
            1. extracts basic features from data
            2. for numerics, runs through imputer and scaler, depending on flags
            3. for categorical, runs through onehot encoding, depending on flags
            4. for binary, doesn't do anything extra
            5. for text, converts to term frequency (TODO)
        
        output is one combined dataframe, dense format
        """
        
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
            """
            Helper function to calculate if crime happened late at night
            returns Series of 1 if is late at night, 0 otherwise
            
            see Arthur's code
            """
            hrs = dtt.dt.hour
            res = np.ones(shape = hrs.shape)
            res[(hrs > self.not_latenight[0]) & (hrs < self.not_latenight[1])] = 0
            res = res.astype("int")
            return res
        
        def process_numerics(sr, imputer, scaler, rd = 3):
            """
            Helper function to impute any missing values on numeric features, and scale them using the scaler
            expects a Series, imputer, scaler, and optional rounding
            also rounds floats to 3 decimal places by default
            returns Series or Dataframe of numeric features
            """
            out = sr.copy()
            pipe = Pipeline([
                ("imputer", imputer),
                ("scaler", scaler)
            ])
            out = pipe.fit_transform(out.to_frame())
            out = np.round(out, rd)
            return out
        
        def process_onehot(sr, prefix = ""):
            """
            Helper function to one-hot encode a categorical feature
            expects a Series, and optional column prefix
            returns dataframe of one-hot encoded feature, will need to be concatenated with the original dataframe
            """
            out = sr.copy()
            one_hot = OneHotEncoder(sparse = False, categories = "auto")
            pipe = Pipeline([
                ("encoder", one_hot)
            ])
            oh = pipe.fit_transform(out.to_frame())
            oh = oh.astype("int")
            cols = np.concatenate(one_hot.categories_).ravel().tolist()
            cols = list(map(str, cols))
            cols = [prefix + c for c in cols]
            out = pd.DataFrame(oh, columns = cols)
            return out
        
        def process_geo(coord, mn, mx, geo_strategy):
            """
            Helper function to process any longitude X or latitude Y that is outside the geo_fence
            """
            coord_out = coord.copy()
            if (geo_strategy == "identity"):
                pass
            elif (geo_strategy == "imputer"):
                # sets any X that is outside geo fence limits to NaN, for imputation
                coord_out[(coord_out < mn) | (coord_out > mx)] = np.nan
            elif (geo_strategy == "centroid"):
                # TODO implement a centroid geo_strategy
                pass
            else:
                raise Exception("unexpected geo_strategy!", geo_strategy)
            return coord_out
        

        
        # creates a copy of the input dataframe
        X_out = X.copy()
        # resets index or concatenations don't work later ...
        X_out.reset_index(drop=True, inplace=True)
        
        # handles X and Y
        if ("X" in X_out.columns):
            X_out["X"] = process_numerics(process_geo(X_out["X"], self.geo_fence[0], self.geo_fence[1], self.geo_strategy), self.imputer, self.scaler)
        
        if ("Y" in X_out.columns):
            X_out["Y"] = process_numerics(process_geo(X_out["Y"], self.geo_fence[2], self.geo_fence[3], self.geo_strategy), self.imputer, self.scaler)
        
        if ("X" in X_out.columns) & ("Y" in X_out.columns):
            # TODO calculate distance from hotspots of crime
            # somehow we need to have a list of hotspots for each category of crime ...
            pass
        
        if ("PdDistrict" in X_out.columns):
            if (self.pddistrict_onehot):
                oh = process_onehot(X_out["PdDistrict"], "pdd_")
                X_out = pd.concat([X_out, oh], axis = 1, join_axes=[X_out.index])
                if (self.pddistrict_drop):
                    X_out = X_out.drop(["PdDistrict"], axis = 1)
        
        if ("DayOfWeek" in X_out.columns):
            if (self.dayofweek_onehot):
                oh = process_onehot(X_out["DayOfWeek"], "dow_")
                X_out = pd.concat([X_out, oh], axis = 1, join_axes=[X_out.index])
                if (self.dayofweek_drop):
                    X_out = X_out.drop(["DayOfWeek"], axis = 1)
        
        # extracts various date-related features
        if ("Dates" in X_out.columns):
            dtt = pd.to_datetime(X_out["Dates"])
            if (self.dates_drop):
                X_out = X_out.drop(["Dates"], axis = 1)
            
            if (self.dates_deltas):
                X_out["hour_delta"] = process_numerics(add_delta(dtt, "h"), self.imputer, self.scaler, rd = 6) # hour since start, 0 to 108263
                X_out["day_delta"] = process_numerics(add_delta(dtt, "D"), self.imputer, self.scaler, rd = 4) # day since start, 0 to 4510
                X_out["week_delta"] = process_numerics(add_delta(dtt, "W"), self.imputer, self.scaler) # week since start, 0 to 644
                X_out["month_delta"] = process_numerics(add_delta(dtt, "M"), self.imputer, self.scaler) # month since start, 0 to 148
                X_out["year_delta"] = process_numerics(add_delta(dtt, "Y"), self.imputer, self.scaler) # year since start, 0 to 12
            
            if (self.dates_binary):
                X_out["is_weekend"] = dtt.dt.dayofweek // 5 # 1 if sat or sun, 0 otherwise
                X_out["is_holiday"] = calc_is_holiday(dtt) # 1 if holiday, 0 otherwise
                X_out["is_latenight"] = calc_is_latenight(dtt) # 1 if after 8 pm and before 6 am, 0 otherwise
            
            if (self.dates_ordinal):
                X_out["hour_of_day"] = dtt.dt.hour # 0 to 23
                X_out["day_of_week"] = dtt.dt.dayofweek # 0 to 7, note day name is already DayOfWeek
                X_out["day_of_month"] = dtt.dt.day # 1 to 31
                X_out["day_of_year"] = dtt.dt.dayofyear # 1 to 365
                X_out["week_of_year"] = dtt.dt.week # 2 to 52
                X_out["month_of_year"] = dtt.dt.month # 1 to 12
                X_out["quarter_of_year"] = dtt.dt.quarter # 1 to 4
                X_out["year"] = dtt.dt.year # 2003 to 2015
            
            if (self.dates_onehot):
                # hour of day
                oh = process_onehot(dtt.dt.hour.astype(int), "hod_")
                X_out = pd.concat([X_out, oh], axis = 1, join_axes=[X_out.index])
                # day of week
                # DayOfWeek as onehot is already covered, so no need to repeat here
                # day of month
                oh = process_onehot(dtt.dt.day.astype(int), "dom_")
                X_out = pd.concat([X_out, oh], axis = 1, join_axes=[X_out.index])
                # day of year
                # DO NOT USE, appears to crash laptop
                #oh = process_onehot(dtt.dt.dayofyear.astype(int), "doy_")
                #X_out = pd.concat([X_out, oh], axis = 1, join_axes=[X_out.index])
                # week of year
                oh = process_onehot(dtt.dt.weekofyear.astype(int), "woy_")
                X_out = pd.concat([X_out, oh], axis = 1, join_axes=[X_out.index])
                # month of year
                oh = process_onehot(dtt.dt.quarter.astype(int), "qoy_")
                X_out = pd.concat([X_out, oh], axis = 1, join_axes=[X_out.index])
                # quarter of year
                oh = process_onehot(dtt.dt.year.astype(int), "y_")
                X_out = pd.concat([X_out, oh], axis = 1, join_axes=[X_out.index])
                
            
            if (self.dates_cyclical):
                # calculate cyclical values for hours, etc
                # http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
                X_out["hour_of_day_sin"] = process_numerics(np.sin(dtt.dt.hour * (2. * np.pi / 24)), self.imputer, self.scaler)
                X_out["hour_of_day_cos"] = process_numerics(np.cos(dtt.dt.hour * (2. * np.pi / 24)), self.imputer, self.scaler)
                X_out["day_of_week_sin"] = process_numerics(np.sin((dtt.dt.dayofweek - 1) * (2. * np.pi / 7)), self.imputer, self.scaler)
                X_out["day_of_week_cos"] = process_numerics(np.cos((dtt.dt.dayofweek - 1) * (2. * np.pi / 7)), self.imputer, self.scaler)
                X_out["day_of_month_sin"] = process_numerics(np.sin((dtt.dt.day - 1) * (2. * np.pi / 31)), self.imputer, self.scaler)
                X_out["day_of_month_cos"] = process_numerics(np.cos((dtt.dt.day - 1) * (2. * np.pi / 31)), self.imputer, self.scaler)
                X_out["day_of_year_sin"] = process_numerics(np.sin((dtt.dt.dayofyear - 1) * (2. * np.pi / 366)), self.imputer, self.scaler)
                X_out["day_of_year_cos"] = process_numerics(np.cos((dtt.dt.dayofyear - 1) * (2. * np.pi / 366)), self.imputer, self.scaler)
                X_out["week_of_year_sin"] = process_numerics(np.sin((dtt.dt.weekofyear - 1) * (2. * np.pi / 53)), self.imputer, self.scaler)
                X_out["week_of_year_cos"] = process_numerics(np.cos((dtt.dt.weekofyear - 1) * (2. * np.pi / 53)), self.imputer, self.scaler)
                X_out["month_of_year_sin"] = process_numerics(np.sin((dtt.dt.month - 1) * (2. * np.pi / 12)), self.imputer, self.scaler)
                X_out["month_of_year_cos"] = process_numerics(np.cos((dtt.dt.month - 1) * (2. * np.pi / 12)), self.imputer, self.scaler)
                X_out["quarter_of_year_sin"] = process_numerics(np.sin((dtt.dt.quarter - 1) * (2. * np.pi / 4)), self.imputer, self.scaler)
                X_out["quarter_of_year_cos"] = process_numerics(np.cos((dtt.dt.quarter - 1) * (2. * np.pi / 4)), self.imputer, self.scaler)
                # does not make sense for year to be cyclical
        
        if ("Address" in X_out.columns):
            if (self.address_vectorize):
                # TODO calculate address-based features, such as street, intersection, etc
                # might be better to do this in prep_data with CountVectorizer or something
                pass
                if (self.address_drop):
                    X_out = X_out.drop(["Address"], axis = 1)
        
        return X_out

def prep_data(train_pd, test_pd, dev_size = 0, rs = 0):
    """
    Helper function to shuffle and separate the train, labels, test data, and test ids
    
    params
        train_pd, the train data set as a dataframe
        test_pd, the test data set as a dataframe
        dev_size, the fraction of the train data to split into a dev set, default 0
        rs, the random seed used to reproduce any shuffles
    """
    # note, we don't need a dev set since we will be using cross validation
    train_data, dev_data = train_test_split(train_pd, test_size = dev_size, shuffle = True, random_state = rs)
    train_labels = train_data.Category
    dev_labels = dev_data.Category
    train_data = train_data.drop(["Category", "Descript", "Resolution"], axis = 1)
    dev_data = dev_data.drop(["Category", "Descript", "Resolution"], axis = 1)
    
    test_data, _ = train_test_split(test_pd, test_size = 0, shuffle = True, random_state = rs)
    test_ids = test_data.Id
    test_data = test_data.drop(["Id"], axis = 1)
    
    return train_data, train_labels, dev_data, dev_lables, test_data, test_ids

def prep_submissions(predsproba, categories, ids):
    """
    Helper function to prepare the raw predsproba array into a panda with the correct column headers and an index
    """
    cols = np.sort(pd.unique(categories))
    submissions = pd.DataFrame(data = predsproba, columns = cols)
    
    # rounds any floats to less precision
    submissions= submissions[cols].round(2)
    
    # adds an Id column
    submissions.insert(loc = 0, column = "Id", value = ids.tolist())
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
        