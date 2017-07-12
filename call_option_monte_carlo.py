# -*- coding: utf-8 -*-
"""
Data Analytics models Quantititive Finance

Python implementation of pricing a call option on a simple underlying like stock using Black-Scholes and Monte Carlo simulation of terminal price
"""
import numpy as np
from scipy.stats import norm

#initialize call option parameters
mju=0.01 #risk-free rate
sigma=0.05 #constant volatility
K=18.0 #strike
S0=18.0 #spot at time 0
T=10 #number of discrete time stpes until maturity
disc=np.exp(-mju*T) #discounting
vol = sigma*np.sqrt(T) #time-scaled volatility

# calculate price using Black-Scholes
d1=((np.log(S0/K)+(mju+0.5*sigma*sigma)*T))/vol
d2=d1-vol
BSprice = S0*norm.cdf(d1)-disc*K*norm.cdf(d2)

# simulate N ternimal prices of the underlying
N=1000000
np.random.seed(0)
rands=np.random.normal(size=N)

ST=S0*np.exp((mju-0.5*sigma*sigma)*T+vol*rands)
payoff=np.multiply([p if p > 0 else 0 for p in ST-K],disc)
MCprice=np.mean(payoff,axis=0)

#prints 2.145200 2.148449
print ("Option price with BS and MC are: %f %f" % (BSprice, MCprice))
