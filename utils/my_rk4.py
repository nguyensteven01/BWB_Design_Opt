import numpy as np
import matplotlib.pyplot as plt

def rk4_step(f, t, y, h):
    """One step of classical RK4 for y' = f(t, y).

    Parameters
    ----------
    f : callable
        RHS f(t, y) -> ndarray of same shape as y.
    t : float
        Current time.
    y : ndarray
        Current state.
    h : float
        Step size.

    Returns
    -------
    y_next : ndarray
        State at t + h.
    """
    k1 = f(t,           y)
    k2 = f(t + 0.5*h,   y + 0.5*h*k1)
    k3 = f(t + 0.5*h,   y + 0.5*h*k2)
    k4 = f(t + h,       y + h*k3)
    return y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def rk4_integrate(f, t_span, y0, h):
    """Integrate y' = f(t, y) on t_span = (t0, tf) with fixed step h.

    The final step is shortened if (tf - t0) is not a multiple of h, so the
    last grid point lands exactly at tf.

    Returns
    -------
    t : (N,) ndarray
        Time grid.
    Y : (N, d) ndarray
        State trajectory, Y[k] = y(t[k]).
    """
    t0, tf = t_span
    if h <= 0 or tf <= t0:
        raise ValueError("Require h > 0 and tf > t0.")

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n_full = int(np.floor((tf - t0) / h))
    h_last = (tf - t0) - n_full * h
    n_steps = n_full + (1 if h_last > 0 else 0)

    t = np.empty(n_steps + 1)
    Y = np.empty((n_steps + 1, y0.size))
    t[0], Y[0] = t0, y0

    for k in range(n_full):
        Y[k+1] = rk4_step(f, t[k], Y[k], h)
        t[k+1] = t[k] + h
    if h_last > 0:
        Y[-1] = rk4_step(f, t[-2], Y[-2], h_last)
        t[-1] = tf

    return t, Y

if __name__ == "__main__":
    def myode(t,x):
        A = np.array([[0, 1],[-5, -2]])
        return A@x
    tf = 5
    x0 = np.array([2, 5])
    t2, X = rk4_integrate(myode, (0.0, tf), x0, h=1e-2)

    plt.plot(X[:,0],X[:,1])
    plt.grid()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()