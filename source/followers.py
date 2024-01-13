import cvxpy as cp
import numpy as np

def MVO(r, Q, Z, ub):
    # r (nx1), Q(nxn), Z (1x1) ub(nx1)

    # Find the total number of assets
    n = len(r)

    # Disallow short sales
    lb = np.zeros(n)

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(r.T @ x),
                      [cp.quad_form(x, Q) <= Z**2,
                       Aeq @ x == beq,
                       x >= lb,
                       x <= ub])
    prob.solve(verbose=False, solver=cp.ECOS)
    return x.value