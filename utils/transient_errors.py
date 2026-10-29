import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

def max_peak(t, x, prominence=None):
    maxima, _ = find_peaks(x, prominence=prominence)
    minima, _ = find_peaks(-x, prominence=prominence)
    ext = np.concatenate((maxima, minima))
    if ext.size == 0:
        print('issue computing rise/settle/os')
        return None, None, None
    i = ext[np.argmax(np.abs(x[ext]))]
    return t[i], x[i], i

def rise_settle(t,state_signal,trim_signal,initial_time,threshold=0.1,state_track=3):
    # This will be a specialized function to compute rise time for my nonlinear reference tracking sims.
    # It will be calculated by checking all the times at which a given signal crosses 10% of the initial value
    # This will yield 

    '''
    Inputs:
    t - time signal
    x - state signal
    initial_time - Time that we treat as the "initial" time for computing rise time. We want to know when the state reaches (threshold * 100)% of its value at the initial time.
    threshold - How close the signal needs to be to equilibrium before we determine it has finished rising
    
    Outputs:
    rise time'''

    new_t = t[t > initial_time]
    x = state_signal[:-1,:] - trim_signal
    x = x[:,t > initial_time]

    t = new_t

    x_init = x[state_track,0]

    bar = np.abs(threshold * x_init)
    time_below = t[np.abs(x[state_track,:]) <= bar]
    rise = time_below[0] - initial_time

    # settle
    time_above = t[np.abs(x[state_track,:]) >= bar]
    settle = time_above[-1] - initial_time

    overshoot_time, overshoot_peak, _idx = max_peak(t,x[state_track,:])
    overshoot_pct = np.abs(overshoot_peak) / np.abs(x[state_track,0])
    # print(f'Rise time: {rise}, Settle time: {settle}, Overshoot: (Peak) {overshoot_peak}, (PCT) {overshoot_pct}%')

    return rise, settle, overshoot_pct

def transient_errors(sim_dictionary, switch_time = 20):
    print("TRACKING ERROR ANALYSIS")
    print("="*70)
    
    for model, data in sim_dictionary.items():
        if model == 'ref':
            continue
        t = data['t']
        x = data['x']
        tp = data['tp']
        u = data['u']

        # print('here again')
        # print(u[:,599:605])

        rise, settle, os = rise_settle(t,x,tp,switch_time,threshold=0.05)

        
        # Compute errors
        error = x[0:4, :] - tp
        vel_err =   np.sqrt(trapezoid(error[0, :]**2, t))
        aoa_err =   np.sqrt(trapezoid(error[1, :]**2, t))
        pr_err =    np.sqrt(trapezoid(error[2, :]**2, t))
        pitch_err = np.sqrt(trapezoid(error[3, :]**2, t))
        error_norm = np.sqrt(trapezoid(np.sum(error**2, 0), t))
        u_norm = np.sqrt(trapezoid(np.sum(u**2,axis=0),t))
        elevator_norm = np.sqrt(trapezoid(u[1,:]**2,t))

        data['error'] = {
                'vel_err': vel_err,
                'aoa_err': aoa_err,
                'pr_err': pr_err,
                'pitch_err': pitch_err,
                'error_norm': error_norm,
                'u_norm':u_norm,
                'elev_norm':elevator_norm
        }
        print(f"\n{model.upper()} Model:")
        print(f"  Total error (L2 norm): {error_norm:.6f}")
        print(f"  Velocity error:        {vel_err:.6f}")
        print(f"  AOA error:             {aoa_err:.6f}")
        print(f"  Pitch rate error:      {pr_err:.6f}")
        print(f"  Pitch angle error:     {pitch_err:.6f}")
        print(f"  Input L2 norm:         {u_norm:.6f}")
        print(f"  Elevator L2 norm:      {elevator_norm:.6f}")

        print(f"  Rise time:             {rise}")
        print(f"  Settle time:           {settle}")
        print(f"  Overshoot:             {100*os}%")

        if model == 'initial':
            init_vel = vel_err
            init_aoa = aoa_err
            init_pr = pr_err
            init_pitch = pitch_err
            init_total = error_norm
            init_u = u_norm
            
            init_rise = rise
            init_settle = settle
            init_overshoot = os

            init_elev = elevator_norm
            continue   # This skips the rest of the for loop and starts again.

        vel_rel_err = (init_vel - vel_err)/init_vel
        aoa_rel_err = (init_aoa - aoa_err)/init_aoa
        pr_rel_err = (init_pr - pr_err)/init_pr
        pitch_rel_err = (init_pitch - pitch_err)/init_pitch
        total_rel_err = (init_total - error_norm)/init_total
        u_rel_norm = (init_u - u_norm)/init_u
        elev_rel_norm = (init_elev - elevator_norm)/init_elev

        rise_rel_norm = (rise - init_rise)/init_rise
        settle_rel_norm = (init_settle - settle)/init_settle
        os_rel_norm = (init_overshoot - os)/init_overshoot

        print(f"  Velocity relative error:        {vel_rel_err*100:.2f}")
        print(f"  AOA relative error:             {aoa_rel_err*100:.2f}")
        print(f"  Pitch rate relative error:      {pr_rel_err*100:.2f}")
        print(f"  Pitch angle relative error:     {pitch_rel_err*100:.2f}")
        print(f"  Elevator relative L2 norm:         {elev_rel_norm*100:.2f}")

        print(f"  Rise relative:                  {rise_rel_norm*100:.2f}")
        print(f"  Settle relative:                {settle_rel_norm*100:.2f}")
        print(f"  OS relative:                    {os_rel_norm*100:.2f}")


        data['error']['relative'] = {
            'vel':vel_rel_err     * 100 ,
            'aoa':aoa_rel_err     * 100 ,
            'pr':pr_rel_err       * 100 ,
            'pitch':pitch_rel_err * 100 ,
            'total':total_rel_err * 100 ,
            'u':u_rel_norm
        }

    print("="*70 + "\n")
    pass
