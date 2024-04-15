"""
Horse-racing examples of causal shrinkage in python
Colin Gray, April 2024

conda activate pymc_env
ipython
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats
import statsmodels.api as sm

np.random.seed(530)

def load_data(*, N=10000, mu=2, sigma=1, num_features=10, ate=10):
    "Create base data."
    # Groundwork
    d = (np.random.uniform(size=N) > 0.5).astype(int)
    y0 = np.random.lognormal(mu, sigma, size=N) # TODO: convert to lognormal
    y = y0 + d*ate
    data = pd.DataFrame({'y0': y0, 'y': y, 'd': d})
    data['cons'] = 1
    # Add coefficients & interactions
    coefs, interactions, features = [], [], ['cons']
    for j in range(num_features):
        data[f"c{j}"] = np.random.uniform()*np.random.lognormal(mu/2, sigma/2, size=N) # TODO: convert to lognormal
        features += [f"c{j}"]
        coefs += [np.round(np.random.lognormal(0, 1), 2)] # TODO: convert to lognormal
        data['y'] += coefs[j]*data[f"c{j}"] 
        interactions += [np.round(np.random.lognormal(0, 1), 2)] # TODO: convert to lognormal
        data['y'] += interactions[j]*(data[f"c{j}"] - data[f"c{j}"].mean())*data['d']
    return data, {'features': features, 'coefs': coefs, 'interactions': interactions}


def residualize_features(data, features):
    "Residualize features to mimic first stage of a meta-learner."
    ymodel = sm.OLS(endog=data['y'], exog=data[features]).fit()
    data['yresid'] = data['y'] - ymodel.predict(data[features])
    dmodel = sm.OLS(endog=data['d'], exog=data[features]).fit()
    data['dresid'] = data['d'] - dmodel.predict(data[features])
    return data[['yresid', 'dresid']]

def gaussian_qq_plot(yvar, figpath='figures/qq_residuals'):
    stats.probplot(yvar, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.savefig(figpath)
    plt.close()


# LOAD DATA #
data, dgp = load_data(num_features=10)
resids = residualize_features(data, dgp['features'])
reg = sm.OLS.from_formula("yresid ~ 1 + dresid", data=data).fit(cov_type='HC3')
print(f"Simple Regression ATE: {reg.summary()}")


# BAYESIAN HIERARCHICAL MODELS #

# compare residuals to a lognormal
resids = resids.copy()
resids['logy'] = np.log(resids['yresid'] - resids['yresid'].min() + 1)
gaussian_qq_plot(resids.loc[data['d']==0, 'logy'])

# run pymc3 model
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=100) # very weak priors
    beta = pm.Normal('beta', mu=0, sigma=100)
    mu = alpha + beta*data['d']  # model
    y_obs = pm.Normal('y_obs', mu=mu, sigma=100, observed=data['y'])
    # TODO: replace with yresid

with model:
    trace = pm.sample(5000, return_inferencedata=True)

# summarize
print(f"Posterior Summary: \n {az.summary(trace)}")
az.plot_posterior(trace, var_names=["beta"], hdi_prob=0.94)
plt.savefig('figures/beta_posterior')
plt.close()

# TODO: why is stdev of bayesian method so much wider than stdev of ols?
# TODO: how would I adapt this to handle a lognormal distribution with a gaussian lift (avoiding <0 errors)?




# R-LEARNER SHRINKAGE #


# HACKY SHRINKAGE #
