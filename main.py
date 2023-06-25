
import numpy as np

#Q1 a
import numpy as np

# The returns of each assets class
returns = np.array([0.031, 0.076, 0.012])
# Assets allocation in Fund Y
weights_Y = np.array([0.6, 0.2, 0.2])
# The expected return of Fund Y
expected_return_Y = np.dot(returns, weights)
print(expected_return_Y) #0.036199999999999996

#Q1b
#Assets allocation
weights_X = np.array([0.3, 0.6, 0.1])
weights_Z = np.array([0.3, 0.3, 0.4])

# ER
expected_return_X = np.dot(returns, weights_X)
print(expected_return_X)# 0.05609999999999999
expected_return_Z = np.dot(returns, weights_Z)
print(expected_return_Z)# 0.036899999999999995

#the most appropriate
print(max(expected_return_X, expected_return_Y, expected_return_Z))

#Q1c
#The variance-covariance matrix
cov_matrix = np.array([
    [0.0024, -0.0010, 0.0005],
    [-0.0010, 0.0297, 0.0056],
    [0.0005, 0.0056, 0.0259]
])

var_Z = np.dot(weights_Z.T, np.dot(cov_matrix, weights_Z))
std_Z = np.sqrt(var_Z)
print(var_Z, std_Z) #0.008317000000000001 0.09119758768739446

#Q1d
# Fund allocation(TableB)
A = np.array([
    [0.3, 0.6, 0.3],  # Bonds in X, Y, Z
    [0.6, 0.2, 0.3],  # Equities
    [0.1, 0.2, 0.4]   # Properties
])

# Desired proportions of assets(Given)
b = np.array([0.5, 0.3, 0.2])
#portfolio
p = np.linalg.solve(A, b)
print(p) #[0.22222222 0.66666667 0.11111111]

#check
result = np.dot(A, p)
print(result)


#Q2a
#Creat rhe payoff matrix
P_matrix = np.array([
    [40, 120, 140],
    [40, 60, 80],
    [40, 10, 30]
])

det = np.linalg.det(P_matrix) #-8.526512829121214e-12
print(det == 0)#True

#Q2 b
#Test security D
P_matrix_D = np.array([
    [40, 120, 60],
    [40, 60, 30],
    [40, 10, 5]
])
det_D = np.linalg.det(P_matrix_D)
print(det_D == 0) #F

P_matrix_E = np.array([
    [40, 120, 80],
    [40, 60, 20],
    [40, 10, 0]
])
det_E = np.linalg.det(P_matrix_E)
print(det_E == 0) #F

#Calculate
Q = np.array([
    [40, 120, 60],
    [40, 60, 30],
    [40, 10, 5]
])
Ps = np.array([32, 44, 22])
Patom = np.matmul(Ps, np.linalg.inv(Q))
print(Patom)

Q = np.array([
    [40, 120, 80],
    [40, 60, 20],
    [40, 10, 0]
])
Ps = np.array([32, 44, 20])
Patom = np.matmul(Ps, np.linalg.inv(Q))
print(Patom) #[0.15555556 0.37777778 0.26666667]

#Q2d
c = np.array([160, 100, 50])
p_free = np.dot(c, Patom)
print(p_free) #75.99999999999997

#Q2h
cf = np.array([70, 40, 15])
EPf = np.dot(cf, Patom)
print(EPf) #29.999999999999993

#Q3a
# Bond parameters
FV = 100
price = 95
C_R = 0.03
T = 4
#calculate
from scipy.optimize import fsolve
y0 = 0.06
def f(y):
    x = 3/(1+y) + 3/(1+y)**2 + 3/(1+y)**3 +103/(1+y)**4 - 95
    return x
x = fsolve(f,y0)
print(x) #[0.04390137]

#Q3b



