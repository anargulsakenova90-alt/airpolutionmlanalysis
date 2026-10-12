import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

class PollutionPreprocessor(BaseEstimator, TransformerMixin):
    """Training-only statistics and donors; causal extrapolation uses past rows only.

    Inference accepts chronologically sorted feature rows. Supply recent context
    before the requested prediction window when using causal_linear.
    """
    def __init__(self, method='mean', donor_limit=2000):
        self.method=method
        self.donor_limit=donor_limit

    def fit(self,X,y=None):
        a=np.asarray(X,dtype=float)
        self.n_features_in_=a.shape[1]
        self.fallback_=np.nanmedian(a,axis=0)
        self.fallback_=np.nan_to_num(self.fallback_,nan=0.)
        if self.method in ['mean','median']:
            self.imputer_=SimpleImputer(strategy=self.method,keep_empty_features=True).fit(a)
        elif self.method=='KNN':
            self.distance_scaler_=StandardScaler().fit(a)
            z=self.distance_scaler_.transform(a)
            idx=np.random.default_rng(42).choice(len(z),min(self.donor_limit,len(z)),replace=False)
            self.imputer_=KNNImputer(n_neighbors=5,weights='distance',keep_empty_features=True).fit(z[idx])
        self.output_scaler_=StandardScaler().fit(self._fill(X))
        return self

    def _fill(self,a):
        breaks=np.zeros(len(a),dtype=bool)
        if hasattr(a,'index') and isinstance(a.index,pd.DatetimeIndex):
            breaks[1:]=(a.index[1:]-a.index[:-1])!=pd.Timedelta(hours=1)
        a=np.asarray(a,dtype=float)
        if self.method in ['mean','median']:
            return self.imputer_.transform(a)
        if self.method=='KNN':
            z=self.distance_scaler_.transform(a)
            out=np.vstack([self.imputer_.transform(z[i:i+1000]) for i in range(0,len(z),1000)])
            # Empty training features have mean/scale NaN; no future observations
            # may introduce a feature absent during training.
            out=out*self.distance_scaler_.scale_+self.distance_scaler_.mean_
            return np.nan_to_num(out,nan=0.)
        out=a.copy()
        for j in range(a.shape[1]):
            history=[]; gap=0
            for i in range(len(a)):
                if breaks[i]: history=[]; gap=0
                value=a[i,j]
                if np.isfinite(value):
                    history.append(value); history=history[-2:]; gap=0
                else:
                    gap+=1
                    if gap<=3 and history:
                        value=history[-1]+(history[-1]-history[-2] if len(history)==2 else 0)
                        out[i,j]=max(0.,value)
                        history.append(out[i,j]); history=history[-2:]
                    else:
                        out[i,j]=self.fallback_[j]
        return out

    def transform(self,X):
        return self.output_scaler_.transform(self._fill(X))
