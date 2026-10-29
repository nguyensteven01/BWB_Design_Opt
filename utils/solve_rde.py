import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import inv
import matplotlib.pyplot as plt
from functools import partial

def solve_rde(A, B, Q, R, T_f, P_f=None, ref_traj = None, t_eval=None, method='RK45'):
    """
    Solves the matrix Riccati differential equation backward in time:
        -dot P = A^T P + P A - P B R^{-1} B^T P + Q,   P(T_f)=P_f
    If n-dimensional reference trajectory provided, also solves:
        -dot eta = (A^T - P B R^{-1} B^T) eta - Q x_ref(t), eta(T_f) = -P_f * ref_traj(T_f)
    Returns:
        t_grid: increasing times from 0..T_f (numpy array)
        P_t: array shape (len(t_grid), n, n) with P(t) on that grid
    Inputs:
        A, B, Q, R: numpy arrays with compatible dimensions
        T_f: final time (float)
        P_f: terminal matrix (if None, defaults to zero matrix)
        ref_traj: Function of time that returns a STATE (n-dim) reference trajectory for all t in [0,T_f]. If not included,
                  it will be assumed that there is no pre-specified trajectory. If included, this function will also
                  solve for the value of eta that yields optimal control for tracking...
        t_eval: optional increasing time grid (0..T_f). If provided, will be used (must be increasing).
                Internally we integrate backward, so we create the reversed grid for solve_ivp.
        method: ODE solver to use (default 'RK45'; use 'Radau' or 'BDF' if stiff)

    Outputs:
        t_grid: increasing times from 0..T_f (numpy array)
        P_t: array shape (len(t_grid), n, n) with P_t(i,:) = P(t_grid(i))
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)
    n = A.shape[0]

    if P_f is None:
        P_f = np.zeros((n, n))
    Rinv = inv(R)

    def backward_ode_full_args(t, y_flat, ref_traj):
        if not ref_traj:
            P = y_flat.reshape((n, n))
            term = A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q
            dotP = -term
            return dotP.reshape(-1)
        else:
            P_flat = y_flat[:n*n]
            eta_flat = y_flat[n*n:]
            P = P_flat.reshape((n,n))
            eta = eta_flat.reshape((n,1))

            term = A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q
            dotP = -term
            xref = np.asarray(ref_traj(t)).reshape((n,1))
            RHS_eta = (A.T - P @ B @ Rinv @ B.T) @ eta - Q @ xref
            doteta = -RHS_eta

            dy = np.concatenate([dotP.reshape(-1), doteta.reshape(-1)])
            return dy
        
    backward_ode = partial(backward_ode_full_args,ref_traj=ref_traj)

    # prepare t_eval for solve_ivp (it wants the mesh in the direction we integrate)
    if t_eval is None:
        t_eval_back = np.linspace(T_f, 0.0, 1000)  # backward mesh
    else:
        # ensure it's increasing 0..T_f, then reverse for backward integration
        t_eval = np.array(t_eval)
        if np.min(t_eval) < 0 or np.max(t_eval) > T_f:
            raise ValueError("t_eval must be within [0, T_f].")
        if not np.all(np.diff(t_eval) >= 0):
            raise ValueError("t_eval must be increasing.")
        t_eval_back = t_eval[::-1]  # solve from T_f->0

    if ref_traj is not None:
        xref_f = np.asarray(ref_traj(T_f)).reshape((n,1))
        eta_f = -P_f @ xref_f
        yT = np.concatenate([P_f.reshape(-1), eta_f.reshape(-1)])
        do_eta = True
    else:
        yT = P_f.reshape(-1)
        do_eta = False

    sol_back = solve_ivp(backward_ode, (T_f, 0.0), yT, t_eval=t_eval_back, method=method, rtol=1e-7, atol=1e-9)

    if not sol_back.success:
        raise RuntimeError("Riccati backward integration failed: " + str(sol_back.message))

    # reverse to increasing time
    t_grid = sol_back.t[::-1].copy()
    y_rev = sol_back.y[:, ::-1].copy()

    nt = t_grid.size
    P_t = np.zeros((nt, n, n))
    if ref_traj is not None:
        eta_t = np.zeros((nt,n,1))
        for i in range(nt):
            P_t[i] = y_rev[:n*n, i].reshape((n,n))
            eta_t[i] = y_rev[n*n:,i].reshape((n,1))
        return t_grid, P_t, eta_t
    else:
        for i in range(nt):
            P_t[i] = y_rev[:, i].reshape((n, n))
        return t_grid, P_t

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    
    A = np.array([[0., 1.],
                  [0., 0.]])
    B = np.array([[0.],
                  [1.]])
    Q = np.diag([50., 1.])
    R = np.array([[0.1]])
    T_f = 1

    def xref(t):
        return [np.sin(t),np.cos(t)]

    t_grid, P_t, eta_t= solve_rde(A, B, Q, R, T_f, method='RK45', ref_traj=xref)

    # plot one element of P(t) as a sanity check
    plt.plot(t_grid, P_t[:,0,0])
    plt.plot(t_grid,eta_t[:,1])
    plt.plot(t_grid,eta_t[:,0])
    plt.xlabel('time [s]')
    plt.ylabel('P11(t)')
    plt.title('Riccati solution P_{11}(t)')
    plt.grid(True)
    plt.show()
