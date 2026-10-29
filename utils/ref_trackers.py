import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.linalg import inv
from scipy.interpolate import RegularGridInterpolator
from utils.solve_rde import solve_rde
import control as ct

''' This file will contain a function that takes an A,B matrix, an instruction on the type
of reference tracking to do, and then just return the states, reference, and control.'''

def lqr_ref_track(A,B,case,Tf=5.0, u_ub = [1, 1], u_lb = [-1, -1], do_meas_noise=False, do_proc_noise=False):

    if case == 'velocity':
        C = np.array([[1,0,0,0]])
        x1 = 4
    elif case == 'pitch':
        C = np.array([[0,0,0,1]])
        x1 = 0.5
    elif case == 'aoa':
        C = np.array([[0,1,0,0]])
        x1 = 0.5

    n = A.shape[0]
    m = B.shape[1]

    # LQ Tracking Cost weights
    Q = np.diag(1000 * C.flatten())
    if case == 'velocity':
        R = 1000 * np.identity(m)
    elif case == 'pitch':
        R = 100 * np.identity(m)
    elif case == 'aoa':
        R = 10 * np.identity(m)
        
    Rinv = inv(R)

    # Horizon
    T_f = Tf

    # Terminal cost
    P_f = np.zeros((n, n))

    #%% Reference trajectory (state-level reference x_ref(t))

    def x_reference(t):
        # Is a constant reference.
        L = C.T @ np.linalg.inv(C @ C.T)

        yref = np.array([[x1]])
        xref = L @ yref.T
        return xref

    #%% Solve Riccati equations backward in time

    # Backward-time ODE for P(t) and eta(t)
    # equations:
    #   -dot P = A^T P + P A - P B R^{-1} B^T P + Q
    #   -dot eta = (A^T - P B R^{-1} B^T) eta - Q x_ref(t)
    # Terminal: P(T_f) = P_f, eta(T_f) = -P_f x_ref(T_f)

    t_P_rev, P_t, eta_t = solve_rde(A,B,Q,R,T_f,P_f,x_reference)

    def interp_P_eta(t_query):
        # simple linear interpolation (clamped)
        if t_query <= 0.0:
            return P_t[0], eta_t[0]
        if t_query >= T_f:
            return P_t[-1], eta_t[-1]
        idx = np.searchsorted(t_P_rev, t_query) - 1
        if idx < 0:
            idx = 0
        t0, t1 = t_P_rev[idx], t_P_rev[idx+1]
        alpha = (t_query - t0) / (t1 - t0)
        P_interp = (1-alpha) * P_t[idx] + alpha * P_t[idx+1]
        eta_interp = (1-alpha) * eta_t[idx] + alpha * eta_t[idx+1]
        return P_interp, eta_interp

    #%% Simulate forward in time with the optimal control law
    def control_signal(Rinv,B,P,eta,x):
        u = -Rinv @ B.T @ (P @ x + eta)
        # u[0] = np.clip(u[0],0,1)
        # u[1] = np.clip(u[1],-1,1)
        u[0] = np.clip(u[0],u_lb[0],u_ub[0])
        u[1] = np.clip(u[1],u_lb[1],u_ub[1])
        # u = np.clip(u,-1,1)
        return u

    t_eval_fwd = np.linspace(0.0, T_f, 1001)
    np.random.seed(30)
    w = np.random.normal(0,2,size=(len(t_eval_fwd)))
    w_interp = RegularGridInterpolator([t_eval_fwd],w)


    def closed_loop_forward(t, x_flat):
        x = x_flat.reshape((n,1))
        P_now, eta_now = interp_P_eta(t)
        noise = np.asarray([[1],[1],[1],[1]])*w_interp([t])     # Noise affecting measurement

        if do_meas_noise:
            u = control_signal(Rinv,B,P_now,eta_now,x + noise)
        else:
            u = control_signal(Rinv,B,P_now,eta_now,x)

        # u = min_energy_control(A,B,t)

        if do_proc_noise:
            dx = A@x+ B@u + noise
        else:
            dx = A @ x + B @ u
        return dx.flatten()

    x0 = np.zeros((n,1)).flatten()
    sol_fwd = solve_ivp(closed_loop_forward, (0.0, T_f), x0, t_eval=t_eval_fwd, rtol=1e-7, atol=1e-9)

    if not sol_fwd.success:
        raise RuntimeError("Forward integration failed: " + str(sol_fwd.message))

    t = sol_fwd.t
    x = sol_fwd.y

    # compute reference and control histories
    xref_hist = np.zeros_like(x)
    u_hist = np.zeros((m,t.size))
    for i, ti in enumerate(t):
        # xr = ref_traj(ti)
        xr = x_reference(ti)

        P_now, eta_now = interp_P_eta(ti)
        noise = np.asarray([[1],[1],[1],[1]]) * w_interp([ti])
        if do_meas_noise:
            u = control_signal(Rinv,B,P_now,eta_now,x[:, i].reshape((n,1)) + noise)
        else:
            u = control_signal(Rinv,B,P_now,eta_now,x[:, i].reshape((n,1)))

        xref_hist[:, i] = xr.flatten()
        u_hist[:,i] = u.flatten()

    error = x - xref_hist

    if case == 'pitch':
        error_norm = np.sqrt(trapezoid(error[3,:]**2,t))
    elif case == 'velocity':
        error_norm = np.sqrt(trapezoid(error[0,:]**2,t))
    return t, x, xref_hist, u_hist, error_norm

def integral_ref_track(A,B,case,Tf=5.0, u_ub = [1,1], u_lb=[-1,-1], do_meas_noise = False, do_proc_noise = False):
    n = A.shape[0]
    m = B.shape[1]
    
    if case == 'velocity':
        C = np.array([[1,0,0,0]])
        x1 = 4
    elif case == 'pitch':
        C = np.array([[0,0,0,1]])
        x1 = 0.5

    k = C.shape[0] # outputs
    T_f = Tf
    # Add integrator state:
    Abar = np.concatenate((np.concatenate((A,np.zeros((n,k))),axis=1),np.concatenate((C,np.zeros((k,k))),axis=1)),axis=0)
    Bbar = np.concatenate((B,np.zeros((k,m))),axis=0)

    # print(f'Abar: {Abar}')
    # print(f'Bbar: {Bbar}')

    # LQR Cost weights
    if case == 'pitch':
        Q = np.diag([0,1,0,1,100])
        R = np.diag([0.1,10]) # penalize elevator usage more to avoid too much variation in AOA.
        # Q = np.diag([1, 100, 1, 100, 100])
        # R = np.diag([0.1, 1000])
    elif case == 'velocity':
        Q = np.diag([1, 1, 1, 1, 1])
        R = np.diag([0.1,0.1])
        # Q = np.diag([1, 1, 1, 100, 1])
        # R = np.diag([1, 100])

    #%% Reference trajectory (state-level reference x_ref(t))

    def x_reference(t):
        # x1 = 4
        return x1

    #%% Simulate forward in time with the control

    K = ct.lqr(Abar,Bbar,Q,R)[0]
    def control_signal(x):
        x = x.reshape((n+1,1))
        u = -K@x
        u[0] = np.clip(u[0],u_lb[0],u_ub[0])
        u[1] = np.clip(u[1],u_lb[1],u_ub[1])
        # u = np.clip(u,-1,1)
        return u

    t_eval_fwd = np.linspace(0.0, T_f, 1001)
    np.random.seed(30)
    w = np.random.normal(0,0.5,size=(len(t_eval_fwd)))
    w_interp = RegularGridInterpolator([t_eval_fwd],w)

    def closed_loop_forward(t, x_flat):
        x = x_flat.reshape((n+1,1))
        noise = np.asarray([[1],[1],[1],[1],[1]])*w_interp([t])     # Noise affecting measurement

        if do_meas_noise:
            u = control_signal(x + noise)
        else:
            u = control_signal(x)

        if do_proc_noise:
            dx = Abar @ x + Bbar @ u - np.concatenate((np.zeros((4,1)),np.asarray([[x_reference(t)]])),axis=0) + noise
        else:
            dx = Abar @ x + Bbar @ u - np.concatenate((np.zeros((4,1)),np.asarray([[x_reference(t)]])),axis=0)

        if case=='velocity':
            threshold=5
        elif case=='pitch':
            threshold=1
        if x[4] > threshold:      # prevent wind up, if integrator state too large, keep constant...
            dx[4] = min(dx[4],0)
        elif x[4] < -threshold:
            dx[4] = max(dx[4],0)

        return dx.flatten()

    x0 = np.zeros((n+1,1)).flatten()
    sol_fwd = solve_ivp(closed_loop_forward, (0.0, T_f), x0, t_eval=t_eval_fwd, rtol=1e-7, atol=1e-9)

    if not sol_fwd.success:
        raise RuntimeError("Forward integration failed: " + str(sol_fwd.message))

    t = sol_fwd.t
    y = sol_fwd.y
    x = y[:n,:]
    integrator = y[n:,:]

    # compute reference and control histories
    xref_hist = np.zeros(t.size) # Reference is only one state...
    u_hist = np.zeros((m,t.size))
    for i, ti in enumerate(t):
        xr = x_reference(ti)
        noise = np.asarray([[1],[1],[1],[1],[1]])*w_interp([ti])
        if do_meas_noise:
            u = control_signal(y[:,i].reshape((n+1,1)) + noise)
        else:
            u = control_signal(y[:,i].reshape((n+1,1)))
        xref_hist[i] = xr
        u_hist[:,i] = u.flatten()

    error = C@x - xref_hist
    error_norm = np.sqrt(trapezoid(error**2,t))

    return t, y, xref_hist, u_hist, error_norm[0]