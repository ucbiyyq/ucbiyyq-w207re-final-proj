from sklearn.base import BaseEstimator, TransformerMixin

"""
Module of helper classes and functions to build features.
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