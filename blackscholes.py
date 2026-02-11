import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
d1 = lambda S, K, T, r, sigma: (
    (np.log(S / K) + (r + 0.5 * sigma**2) * T)
    / (sigma * np.sqrt(T))
)

d2 = lambda S, K, T, r, sigma: (
    d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
)
call_price = lambda S, K, T, r, sigma: (S*norm.cdf(d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma)))
put_price = lambda S, K, T, r, sigma: (K * np.exp(-r*T)*norm.cdf(d2(S, K, T, r, sigma)) - S*norm.cdf(d1(S, K, T, r, sigma)))
call_delta = lambda S, K, T, r, sigma: norm.cdf(d1(S, K, T, r, sigma))
put_delta = lambda S, K, T, r, sigma: norm.cdf(d1(S, K, T, r, sigma)) - 1
gamma = lambda S, K, T, r, sigma: norm.pdf(d1(S, K, T, r, sigma))/(S*sigma*np.sqrt(T))
vega = lambda S, K, T, r, sigma: S * norm.pdf(d1(S, K, T, r, sigma)) * np.sqrt(T)
theta_call = lambda S, K, T, r, sigma: -S*norm.pdf(d1(S,K,T,r,sigma))*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
theta_put = lambda S, K, T, r, sigma: -S*norm.pdf(d1(S,K,T,r,sigma))*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma))
rho_call = lambda S, K, T, r, sigma: K*T*np.exp(-r*T)*norm.cdf(d2(S, K, T, r, sigma))
rho_put = lambda S, K, T, r, sigma: -K*T*np.exp(-r*T)*norm.cdf(-d2(S, K, T, r, sigma))  

S0 = 100
K  = 100
T  = 0.25
r  = 0.05
sigma = 0.20

# 2) Call price vs underlying price S
S_grid = np.linspace(50, 150, 500)          # x-axis variable
y_call = call_price(S_grid, K, T, r, sigma) # y-axis output

plt.figure()
plt.plot(S_grid, y_call)
plt.xlabel("Underlying price S")
plt.ylabel("Call price C")
plt.title("Black–Scholes Call Price vs Underlying Price")
plt.grid(True)


# 3) Put delta vs underlying price S
y_put_delta = put_delta(S_grid, K, T, r, sigma)

plt.figure()
plt.plot(S_grid, y_put_delta)
plt.xlabel("Underlying price S")
plt.ylabel("Put delta Δp")
plt.title("Black–Scholes Put Delta vs Underlying Price")
plt.grid(True)



# 4) Gamma vs underlying price S
y_gamma = gamma(S_grid, K, T, r, sigma)

plt.figure()
plt.plot(S_grid, y_gamma)
plt.xlabel("Underlying price S")
plt.ylabel("Gamma Γ")
plt.title("Black–Scholes Gamma vs Underlying Price")
plt.grid(True)



# 5)  Vega vs volatility sigma
sigma_grid = np.linspace(0.05, 0.80, 500)
y_vega = vega(S0, K, T, r, sigma_grid)

plt.figure()
plt.plot(sigma_grid, y_vega)
plt.xlabel("Volatility σ")
plt.ylabel("Vega")
plt.title("Black–Scholes Vega vs Volatility")
plt.grid(True)


# 6)  Call theta vs time to maturity T
T_grid = np.linspace(1/365, 0.25, 500)   
y_theta_call = theta_call(S0, K, T_grid, r, sigma)

plt.figure()
plt.plot(T_grid, y_theta_call)
plt.xlabel("Time to maturity T (years)")
plt.ylabel("Call theta Θc (per year)")
plt.title("Black–Scholes Call Theta vs Time to Maturity")
plt.grid(True)

plt.show()