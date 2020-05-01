import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd


def optimal_portfolio(returns):

    n = len(returns)
    returns = np.asmatrix(returns)
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

def rebalance(T, return_vec):

    start = 0
    w = []
    total_return = []

    portfolio_val = 100000

    total_return.append(portfolio_val)

    delta = len(return_vec[0])//T

    while(start < len(return_vec[0])):

        if T != 2:

            weights, returns, risks = optimal_portfolio([return_vec[0][0:start+delta], return_vec[1][0:start+delta],
                                                         return_vec[2][0:start+delta]])

            start = start + delta

        else:

            weights, returns, risks = optimal_portfolio([return_vec[0][start:start + delta], return_vec[1][start:start + delta],
                                                         return_vec[2][start:start + delta]])

            start = start + delta

        # Calculate returns
        r1 = weights[0] * (sum(return_vec[0][0:start+delta]) / len(return_vec[2][0:start+delta]))
        r2 = weights[1] * (sum(return_vec[1][0:start+delta]) / len(return_vec[2][0:start+delta]))
        r3 = weights[2] * (sum(return_vec[2][0:start+delta]) / len(return_vec[2][0:start+delta]))
        total_return.append((r1+r2+r3+1)*portfolio_val)
        w.append(weights)

    return w, total_return


dataset1_pred = pd.read_csv('aapl_pred.csv')
dataset2_pred = pd.read_csv('ko_pred.csv')
dataset3_pred = pd.read_csv('duk_pred.csv')
dataset1_pred = dataset1_pred[['Predicted Return']]
dataset2_pred = dataset2_pred[['Predicted Return']]
dataset3_pred = dataset3_pred[['Predicted Return']]

dataset1_test = pd.read_csv('aapl_test.csv')
dataset2_test = pd.read_csv('ko_test.csv')
dataset3_test = pd.read_csv('duk_test.csv')
dataset1_test = dataset1_test[['Return']]
dataset2_test = dataset2_test[['Return']]
dataset3_test = dataset3_test[['Return']]

return_vec_pred = [np.flip(dataset1_pred.values.T[0]), np.flip(dataset2_pred.values.T[0]), np.flip(dataset3_pred.values.T[0])]
return_vec_actual = [np.flip(dataset1_test.values.T[0]), np.flip(dataset2_test.values.T[0]), np.flip(dataset3_test.values.T[0])]

# Multi-period optimization
w_pred, total_return_pred = rebalance(4, return_vec_pred)
w_actual, total_return_actual = rebalance(4, return_vec_actual)

periods2 = [0, 1, 2]
periods4 = [0, 1, 2, 3, 4]

# Generate graphs
plt.plot(periods4, total_return_pred, label='Predicted Portfolio')
plt.plot(periods4, total_return_actual, label='Actual Portfolio')
plt.xlabel('Period')
plt.ylabel('Portfolio Value ($)')
plt.title('Predicted Portfolio Value v.s. Actual Portfolio Value')
plt.legend()
plt.show()
