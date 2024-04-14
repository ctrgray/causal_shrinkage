# Horse-racing examples of causal shrinkage in python
# Colin Gray, April 2024

import numpy as np
import pandas as pd
import statsmodels.api as sm


np.random.seed(530)

N = 10000
mu, sigma = 2, 1
num_features = 10
ate = 100

def load_data(*, n=10000, mu=2, sigma=1, num_features=10, ate=10):
    # Base data
    d = (np.random.uniform(size=N) > 0.5).astype(int)
    y0 = np.random.lognormal(mu, sigma, size=N)
    y = y0 + d*ate
    data = pd.DataFrame({'y': y, 'd': d})
    data['cons'] = 1
    # Add coefsficients & interactionsactions
    coefs, interactions, features = [], [], ['cons']
    for j in range(num_features):
        data[f"c{j}"] = np.random.uniform()*np.random.lognormal(mu/2, sigma/2, size=N)
        features += [f"c{j}"]
        coefs += [np.round(np.random.normal(0, 1), 2)]
        data['y'] += coefs[j]*data[f"c{j}"] 
        interactions += [np.round(np.random.normal(0, 1), 2)]
        data['y'] += interactions[j]*(data[f"c{j}"] - data[f"c{j}"].mean())*data['d']
    return data, {'features': features, 'coefs': coefs, 'interactions': interactions}


def residualize_features(data, features):
    ymodel = sm.OLS(endog=data['y'], exog=data[features]).fit()
    data['yresid'] = data['y'] - ymodel.predict(data[features])
    dmodel = sm.OLS(endog=data['d'], exog=data[features]).fit()
    data['dresid'] = data['d'] - dmodel.predict(data[features])
    return data[['yresid', 'dresid']]


data, dgp = load_data()
resids = residualize_features(data, dgp['features'])
reg = sm.OLS.from_formula("yresid ~ 1 + dresid", data=data).fit(cov_type='HC3')
reg.summary()

# option 1: standard r-learner shrinkage

# option 2: bayesian hierarchical shrinkage

# option 3: "hacky" shrinkage
