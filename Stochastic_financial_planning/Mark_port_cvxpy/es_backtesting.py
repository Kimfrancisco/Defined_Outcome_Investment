import os
os.chdir('/Users/kimfr/OneDrive - unist.ac.kr/Optimization/OR1')
os.chdir('/Users/kimfr/OneDrive - unist.ac.kr/2019-fall/OR1/Groupwork')

import numpy as np
price = np.loadtxt('test1.csv',delimiter=',',skiprows=1)
y = np.diff(np.log(price), n=1, axis=0)                  # get returns

import numpy as np
T = len(y)                            # number of obs for y
WE = 1000                             # estimation window length
p = 0.01                              # probability
l1 = int(WE * p)                      # HS observation
value = 1                             # portfolio value
VaR = np.full([T,4], np.nan)          # matrix for forecasts
## EWMA setup
lmbda = 0.94
s11 = np.var(y[1:174])
for t in range(1,WE):
    s11=lmbda*s11+(1-lmbda)*y[t-1]**2
		

from scipy import stats
from arch import arch_model
for t in range(WE, T):
    t1 = t - WE                                                # start of data window
    t2 = t - 1                                                 # end of data window
    window = y[t1:t2+1]                                        # data for estimation
    s11 = lmbda * s11 + (1-lmbda) * y[t-1]**2
    VaR[t,0]=-stats.norm.ppf(p)*np.sqrt(s11)*value             # EWMA
    VaR[t,1]=-np.std(window,ddof=1)*stats.norm.ppf(p)*value    # MA
    ys = np.sort(window)
    VaR[t,2] = -ys[l1 - 1] * value                             # HS
    am = arch_model(window, mean = 'Zero',vol='Garch',
                    p=1, o=0, q=1, dist='Normal')
    res = am.fit(update_freq=0, disp='off',show_warning=False)
    par = [res.params[0], res.params[1], res.params[2]]
    s4 = par[0] + par[1] * window[WE - 1]**2 + par[
        2] * res.conditional_volatility[-1]**2
    VaR[t,3] = -np.sqrt(s4) * stats.norm.ppf(p) * value        # GARCH(1,1)
## GARCH optimization in Python has some convergence issues
		

W1 = WE                                         # Python index starts at 0
m = ["EWMA", "MA", "HS", "GARCH"]
for i in range(4):
    VR = sum(y[W1:T] < -VaR[W1:T,i])/(p*(T-WE))
    s = np.std(VaR[W1:T, i], ddof=1)
    print ([i, m[i], VR, s])
plt.plot(y[W1:T])
plt.plot(VaR[W1:T])
plt.show()
plt.close()
		

def bern_test(p,v):
    lv = len(v)
    sv = sum(v)
    al = np.log(p)*sv + np.log(1-p)*(lv-sv)
    bl = np.log(sv/lv)*sv + np.log(1-sv/lv)*(lv-sv)
    return (-2*(al-bl))
		
def ind_test(V):
    J = np.full([T,4], 0)
    for i in range(1,len(V)-1):
        J[i,0] = (V[i-1] == 0) & (V[i] == 0)
        J[i,1] = (V[i-1] == 0) & (V[i] == 1)
        J[i,2] = (V[i-1] == 1) & (V[i] == 0)
        J[i,3] = (V[i-1] == 1) & (V[i] == 1)
    V_00 = sum(J[:,0])
    V_01 = sum(J[:,1])
    V_10 = sum(J[:,2])
    V_11 = sum(J[:,3])
    p_00=V_00/(V_00+V_01)
    p_01=V_01/(V_00+V_01)
    p_10=V_10/(V_10+V_11)
    p_11=V_11/(V_10+V_11)
    hat_p = (V_01+V_11)/(V_00+V_01+V_10+V_11)
    al = np.log(1-hat_p)*(V_00+V_10) + np.log(hat_p)*(V_01+V_11)
    bl = np.log(p_00)*V_00 + np.log(p_01)*V_01 + np.log(p_10)*V_10 + np.log(p_11)*V_11
    return (-2*(al-bl))
		

W1 = WE
ya = y[W1:T]
VaRa = VaR[W1:T,]
m = ['EWMA', 'MA', 'HS', 'GARCH']
for i in range(4):
    q = y[W1:T] < -VaR[W1:T,i]
    v = VaRa*0
    v[q,i] = 1
    ber = bern_test(p, v[:,i])
    ind = ind_test(v[:,i])
    print ([i, m[i], ber, 1 - stats.chi2.cdf(ber, 1), ind, 1 - stats.chi2.cdf(ind, 1)])
		
VaR = np.full([T,2], np.nan)                                 # VaR forecasts
ES = np.full([T,2], np.nan)                                  # ES forecasts
for t in range(WE, T):
    t1 = t - WE
    t2 = t - 1
    window = y[t1:t2+1]
    s11 = lmbda * s11 + (1-lmbda) * y[t-1]**2
    VaR[t,0] = -stats.norm.ppf(p) * np.sqrt(s11)*value       # EWMA
    ES[t,0]=np.sqrt(s11)*stats.norm.pdf(stats.norm.ppf(p))/p
    ys = np.sort(window)
    VaR[t,1] = -ys[l1 - 1] * value                           # HS
    ES[t,1] = -np.mean(ys[0:l1]) * value


ESa = ES[W1:T,:]
VaRa = VaR[W1:T,:]
m = ["EWMA", "HS"]
for i in range(2):
    q = ya <= -VaRa[:,i]
    nES = np.mean(ya[q] / -ESa[q,i])
    print ([i, m[i], 'nES', nES])