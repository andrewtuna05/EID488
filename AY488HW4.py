from scipy.optimize import linprog
import numpy as np

c = [155, 135, 120] # c Vector

#A Matrix but we multiply by -1 to create -Ax <= -b for the solver which is equivalent 
#to Ax >= b

A = np.array([ 
    [0.2, 0.1, 0.1],   
    [0.2, 0.2, 0.1],   
    [0.3, 0.25, 0.2],  
    [0.2, 0.25, 0.28], 
    [0.1, 0.2, 0.32]   
])

b = np.array([400000, 600000, 500000, 1000000, 300000])  #b vector (we multiply by -1)

#Part C
A_ub = -A#A Matrix but we multiply by -1 to create -Ax <= -b for the solver which is equivalent to Ax >= b
b_ub = -b #b vector (we multiply by -1)

#Decision variables must be non-negative and by default, linprog assumes non-negative constraints on the decision variables
res = linprog(c = c, A_ub = A_ub, b_ub = b_ub, method = "highs")

#Display Results
print("Part C - Optimal Amount of Barrels:")
print(f"Light bio-oil : {res.x[0]:,.2f}")
print(f"Medium bio-oil: {res.x[1]:,.2f}")
print(f"Heavy bio-oil: {res.x[2]:,.2f}")
print(f"Minimum total cost: {res.fun:,.2f}")

#Part D

#Now that we have equality, use A_eq and b_eq

#-np.eye(5) creates a negative 5x5 identity matrix and then np.hstack combines identity matrix to the right of A
A_eq = np.hstack((A, -np.eye(5)))

#We introduced 5 new slack variables into our "decision variables" but since they do not affect anything in the objective function, we multiply them by 0
c_std = np.concatenate([c, np.zeros(5)])

res_std = linprog(c=c_std, A_eq=A_eq, b_eq=b, method="highs")

#Display Results
print()
print("Part D - Optimal Amount of Barrels:")
print(f"Light bio-oil: {res_std.x[0]:,.2f}")
print(f"Medium bio-oil: {res_std.x[1]:,.2f}")
print(f"Heavy bio-oil: {res_std.x[2]:,.2f}")
print(f"Minimum total cost: ${res_std.fun:,.2f}")


#Part E
A_t = np.transpose(A)

#To maximize b.tranpose*y, we do minimize -b.transpose*y 
#c_dual is b from primal
c_dual = -b 

b_dual = c #b_dual is the our c vector from primal

res_dual = linprog(c=c_dual, A_ub=A_t, b_ub=b_dual, method="highs")

print()
print("Part E - Dual:")
print(f"Lambda_1: {res_dual.x[0]:,.2f}")
print(f"Lambda_2: {res_dual.x[1]:,.2f}")
print(f"Lambda_3: {res_dual.x[2]:,.2f}")
print(f"Lambda_4: {res_dual.x[3]:,.2f}")
print(f"Lambda_5: {res_dual.x[4]:,.2f}")
print(f"Maximum Value: ${-1*res_dual.fun:,.2f}") #multiply by negative 1 to get maximum value since we solved for minimize -b.transpose*y in the solver