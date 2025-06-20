"""
Qubit Tuning Module

This module provides functions for tuning and calibrating qubits in a quantum system.
It includes functionality for:
- Tuning up qubits (frequency, pulse parameters, coherence measurements)
- Measuring qubit parameters (T1, T2, frequency, fidelity)
- Tracking qubit parameters over time
- Finding qubit spectroscopy
- Optimizing readout parameters

The module is designed to work with the QICK (Quantum Instrumentation Control Kit) framework.
"""

# Standard library imports
import os
import time
from datetime import datetime

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
import experiments as meas
import slab_qick_calib.config as config
from calib import tuneup

# Configure matplotlib
plt.rcParams['legend.handlelength'] = 0.5

# Default parameters
MAX_T1 = 500  # Maximum T1 time in microseconds
MAX_ERR = 1   # Maximum acceptable error in fits
MIN_R2 = 0.35  # Minimum acceptable R² value for fits
TOL = 0.3     # Tolerance for parameter convergence


def tune_up_qubit(qi, cfg_dict, update=True, first_time=False, readout=True, 
                 single=False, start_coarse=False, max_t1=MAX_T1, 
                 max_err=MAX_ERR, min_r2=MIN_R2, tol=TOL):
    """
    Comprehensive tuneup procedure for a single qubit.
    
    This function performs a complete calibration sequence for a qubit, including:
    - Resonator spectroscopy
    - Qubit spectroscopy
    - Amplitude Rabi oscillations
    - T1 measurement
    - Ramsey (T2*) measurement
    - Echo (T2E) measurement
    - Single-shot optimization
    
    Parameters
    ----------
    qi : int
        Qubit index
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    update : bool, optional
        Whether to update the configuration file with new parameters
    first_time : bool, optional
        Whether this is the first tuneup for this qubit
    readout : bool, optional
        Whether to update readout frequency
    single : bool, optional
        Whether to perform single-shot optimization
    start_coarse : bool, optional
        Whether to start with coarse optimization for single-shot
    max_t1 : float, optional
        Maximum T1 time in microseconds
    max_err : float, optional
        Maximum acceptable error in fits
    min_r2 : float, optional
        Minimum acceptable R² value for fits
    tol : float, optional
        Tolerance for parameter convergence
        
    Returns
    -------
    None
    """
    cfg_path = cfg_dict['cfg_file']
    auto_cfg = config.load(cfg_path)
    
    # Step 1: Resonator spectroscopy to find/verify resonator frequency
    rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span':'kappa'})
    if update: 
        rspec.update(cfg_path, freq=readout)

    # Step 2: If not first time, check histogram to verify readout
    if not first_time: 
        shot = meas.HistogramExperiment(cfg_dict, qi=qi, params={'shots':20000})
        if update: 
            shot.update(cfg_path)
    
    # Step 3: Fine qubit spectroscopy to find/verify qubit frequency
    qspec = meas.QubitSpec(cfg_dict, qi=qi, style='fine', 
                          params={'span':3,'expts':85,'soft_avgs':2, 'length':'t1'})     
    if not qspec.status:
        # If spectroscopy fails, try to find it with broader search
        find_spec(qi, cfg_dict, start="medium")

    # Update qubit frequency if it has drifted significantly
    if np.abs(qspec.data['new_freq'] - auto_cfg.device.qubit.f_ge[qi]) > 0.25 and qspec.status:
        print('Qubit frequency is off spectroscopy by more than 250 kHz, recentering')
        auto_cfg = config.update_qubit(cfg_path, 'f_ge', qspec.data['new_freq'], qi)
    
    # Step 4: Amplitude Rabi to calibrate pi pulse
    amp_rabi = meas.RabiExperiment(cfg_dict, qi=qi, params={'start':0.003})
    if update and amp_rabi.status:
        config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), amp_rabi.data['pi_length'], qi)               
    
    # Step 5: For first-time tuneup, measure T1 and optimize readout
    if first_time:
        # Initial T1 measurement to get coherence time estimate
        t1 = meas.T1Experiment(cfg_dict, qi=qi)
        if update: 
            t1.update(cfg_path, first_time=True)

        # Run single shot optimization to improve readout
        shot = meas.HistogramExperiment(cfg_dict, qi=qi, params={'shots':20000})
        if update: 
            shot.update(cfg_path)

    # Step 6: Run Ramsey to center qubit frequency
    if first_time: 
        recenter(qi, cfg_dict, style='coarse')
    else:
        recenter(qi, cfg_dict, style='fine')

    # Step 7: Refine pi pulse calibration after frequency centering
    amp_rabi = meas.RabiExperiment(cfg_dict, qi=qi, params={'start':0.003})
    if update and amp_rabi.status:
        config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), amp_rabi.data['pi_length'], qi)

    # Step 8: Optimize single-shot readout if requested
    if single: 
        params = {'expts_f':1, 'expts_gain':7, 'expts_len':5}
        meas_opt(cfg_dict, [qi], params, do_res=True, start_coarse=start_coarse)

    # Step 9: Verify readout with histogram
    shot = meas.HistogramExperiment(cfg_dict, qi=qi, params={'shots':20000})
    if update: 
        shot.update(cfg_path)

    # Step 10: Measure coherence times
    # Ramsey (T2*)
    t2r = get_coherence(meas.T2Experiment, qi, cfg_dict, par='T2r')

    # T1
    t1 = get_coherence(meas.T1Experiment, qi, cfg_dict, par='T1')
    if t1.status: 
        # Set final delay based on T1 (typically 6*T1 to ensure full relaxation)
        auto_cfg = config.update_readout(cfg_path, 'final_delay', 6*t1.data['new_t1'], 
                                        qi, sig=2, rng_vals=[10, 5*max_t1])

    # Echo (T2E)
    t2e = get_coherence(meas.T2Experiment, qi, cfg_dict, par='T2e')

    # Step 11: Measure dispersive shift (chi)
    chid, chi_val = tuneup.check_chi(cfg_dict, qi=qi)
    auto_cfg = config.update_readout(cfg_path, 'chi', float(chi_val), qi)
    
    # Step 12: Create summary figure with all measurements
    progs = {'amp_rabi':amp_rabi, 't1':t1, 't2r':t2r, 't2e':t2e, 
             'shot':shot, 'rspec':rspec, 'qspec':qspec, 'chid':chid}
    make_summary_figure(cfg_dict, progs, qi)


def measure_params(qi, cfg_dict, update=True, readout=True, display=False, max_t1=MAX_T1):
    """
    Measure and return key parameters for a qubit.
    
    This function performs a series of measurements to characterize a qubit's properties,
    including coherence times, frequencies, and readout fidelity.
    
    Parameters
    ----------
    qi : int
        Qubit index
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    update : bool, optional
        Whether to update the configuration file with new parameters
    readout : bool, optional
        Whether to update readout frequency
    display : bool, optional
        Whether to display plots during measurements
    max_t1 : float, optional
        Maximum T1 time in microseconds
        
    Returns
    -------
    dict
        Dictionary containing measured qubit parameters
    """
    cfg_path = cfg_dict['cfg_file']
    err_dict = {}
    
    # Step 1: Resonator spectroscopy
    rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span':'kappa'}, 
                         display=display, progress=False)
    if update: 
        rspec.update(cfg_path, freq=readout, fast=True, verbose=False)        
    
    if not rspec.status:
        # Handle failed resonator spectroscopy
        rspec.data['kappa'] = np.nan
        rspec.data['fit'] = [np.nan, np.nan, np.nan]
        rspec.display(debug=True)
        print('Resonator spectroscopy failed')
    
    # Step 2: Single shot measurement
    shot = meas.HistogramExperiment(cfg_dict, qi=qi, params={'shots':10000}, 
                                   display=display, progress=False)
    if update: 
        shot.update(cfg_path, fast=True, verbose=False)

    # Step 3: Amplitude Rabi
    amp_rabi = meas.RabiExperiment(cfg_dict, qi=qi, params={'start':0.003}, 
                                  display=display, progress=False, style='fast')
    if update and amp_rabi.status:
        config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), 
                           amp_rabi.data['pi_length'], qi, verbose=False)
    
    if not amp_rabi.status:
        amp_rabi.data['pi_length'] = np.nan
        print('Amp Rabi failed')
    
    err_dict['rabi_err'] = np.sqrt(amp_rabi.data['fit_err_avgi'][1][1])

    # Step 4: T2 Ramsey
    t2r = meas.T2Experiment(cfg_dict, qi=qi, display=display, progress=False, style='fast')
    if t2r.status and update:
        # Update qubit frequency and T2r time
        config.update_qubit(cfg_path, 'f_ge', t2r.data['new_freq'], qi, verbose=False)
        config.update_qubit(cfg_path, 'T2r', t2r.data['best_fit'][3], qi, 
                           rng_vals=[1, max_t1], sig=2, verbose=False)
    
    if not t2r.status:
        # Try to recover from failed T2 measurement
        t2r.display(debug=True, refit=True)
        print(t2r.data['r2'])
        print(t2r.data['fit_err_par'])

        # Try to find qubit frequency with spectroscopy
        find_spec(qi, cfg_dict, start="fine")
        t2r = meas.T2Experiment(cfg_dict, qi=qi, display=display, progress=False)
        if t2r.status and update:
            config.update_qubit(cfg_path, 'f_ge', t2r.data['new_freq'], qi, verbose=False)
            config.update_qubit(cfg_path, 'T2r', t2r.data['best_fit'][3], qi, 
                               rng_vals=[1, max_t1], sig=2, verbose=False)
            print('Recentered qubit frequency')
        
        if not t2r.status:
            # Handle persistently failed T2 measurement
            t2r.display(debug=True)
            t2r.data['best_fit'] = [np.nan, np.nan, np.nan, np.nan]
            t2r.data['new_freq'] = np.nan
            print('T2 Ramsey failed')
    
    err_dict['t2r_err'] = np.sqrt(t2r.data['fit_err_avgi'][3][3])
    err_dict['fge_err'] = np.sqrt(t2r.data['fit_err_avgi'][1][1])

    # Step 5: T1 measurement
    t1 = meas.T1Experiment(cfg_dict, qi=qi, display=display, progress=False, style='fast')
    if update: 
        t1.update(cfg_path, rng_vals=[1, max_t1], verbose=False)
    
    if not t1.status:
        t1.data['new_t1_i'] = np.nan
        t1.display(debug=True)
        print('T1 failed')
    
    err_dict['t1_err'] = np.sqrt(t1.data['fit_err_avgi'][2][2])

    # Step 6: T2 Echo measurement
    t2e = meas.T2Experiment(cfg_dict, qi=qi, display=display, progress=False, 
                           params={'experiment_type':'echo'}, style='fast')
    if update and t2e.status: 
        config.update_qubit(cfg_path, 'T2e', t2e.data['best_fit'][3], qi, 
                           rng_vals=[1, max_t1], sig=2, verbose=False)
    
    if not t2e.status:
        # Try to recover from failed T2E measurement
        print('Refitting')
        t2e.analyze(refit=True, verbose=True)
        if not t2e.status:
            t2e.data['best_fit'] = [np.nan, np.nan, np.nan, np.nan]
            t2e.display(debug=True)
            print('T2 Echo failed')
    
    err_dict['t2e_err'] = np.sqrt(t2e.data['fit_err_avgi'][3][3])

    # Compile all measured parameters into a dictionary
    qubit_dict = {
        't1': t1.data['new_t1_i'], 
        't2r': t2r.data['best_fit'][3], 
        't2e': t2e.data['best_fit'][3], 
        'f_ge': t2r.data['new_freq'], 
        'fidelity': shot.data['fids'][0],
        'phase': shot.data['angle'], 
        'kappa': rspec.data['kappa'], 
        'frequency': rspec.data['freq_min'],
        'pi_length': amp_rabi.data['pi_length']
    }
    
    # Add error values
    qubit_dict.update(err_dict)
    
    # Add R² values for fit quality assessment
    r2_dict = {
        't1_r2': t1.data['r2'], 
        't2r_r2': t2r.data['r2'], 
        't2e_r2': t2e.data['r2'], 
        'rspec_r2': rspec.data['r2'], 
        'amp_rabi_r2': amp_rabi.data['r2']
    }
    qubit_dict.update(r2_dict)
    
    # Round all values to 7 significant figures
    for key in qubit_dict:
        if isinstance(qubit_dict[key], (int, float)) and not np.isnan(qubit_dict[key]):
            qubit_dict[key] = round(qubit_dict[key], 7)
    
    return qubit_dict


def time_tracking(qubit_list, cfg_dict, total_time=12, display=False):
    """
    Track qubit parameters over time.
    
    This function repeatedly measures qubit parameters over a specified time period
    and saves the results for tracking parameter drift.
    
    Parameters
    ----------
    qubit_list : list
        List of qubit indices to track
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    total_time : float, optional
        Total tracking time in hours
    display : bool, optional
        Whether to display plots during measurements
        
    Returns
    -------
    tuple
        (tracking_data, tracking_path) where tracking_data is a list of dictionaries
        containing the measured parameters for each qubit, and tracking_path is the
        path where the data is saved
    """
    # Create directory for tracking data
    base_path = '\\'.join(cfg_dict['expt_path'].split('\\')[:-2]) + '\\Tracking\\'
    tracking_id = f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}_{total_time:.1f}hrs'
    tracking_path = os.path.join(base_path, tracking_id)
    os.mkdir(tracking_path)
    os.mkdir(os.path.join(tracking_path, 'images'))
    cfg_dict['expt_path'] = tracking_path
    
    # Initialize timing variables
    start_time = time.time()
    elapsed = 0
    i = 0
    
    # Run measurements until total_time is reached
    while elapsed < total_time:
        for j, qi in enumerate(qubit_list):
            # Measure current time
            tm = time.time()
            elapsed = (tm - start_time) / 3600
            print(f"Starting run {i}, for qubit {qi}. Time elapsed {elapsed:.2f} hrs")
            
            # Measure qubit parameters
            d = measure_params(qi, cfg_dict, display=display)
            d['time'] = tm 
            d['elapsed'] = elapsed
            
            # Store data for this iteration
            if i == 0:
                # Initialize storage dictionary for each qubit on first iteration
                tracking_data = [{key: [] for key in d.keys()} for _ in range(len(qubit_list))]
            
            # Append values for each parameter
            for key, val in d.items():
                tracking_data[j][key].append(val)
        
        i += 1
        
        # Save tracking data to CSV files
        for j, qi in enumerate(qubit_list):
            csv_path = os.path.join(base_path, 'csv', f'{tracking_id}_qubit_{qi}_tracking.csv')
            
            # Convert tracking data dict to numpy arrays for saving
            data_arrays = {}
            for key in tracking_data[j].keys():
                data_arrays[key] = np.array(tracking_data[j][key])
            
            # Create header and data rows
            header = ','.join(data_arrays.keys())
            rows = np.vstack(list(data_arrays.values())).T
            
            # Save to CSV
            np.savetxt(csv_path, rows, delimiter=',', header=header, comments='')

    return tracking_data, tracking_path


def make_summary_figure(cfg_dict, progs, qi):
    """
    Create a summary figure showing all measurement results for a qubit.
    
    Parameters
    ----------
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    progs : dict
        Dictionary of measurement program objects
    qi : int
        Qubit index
        
    Returns
    -------
    None
    """
    auto_cfg = config.load(cfg_dict['cfg_file'])
    fig, ax = plt.subplots(3, 3, figsize=(14, 9))
    
    # Display all measurement results
    progs['amp_rabi'].display(ax=[ax[0, 0]])
    progs['t1'].display(ax=[ax[0, 1]])
    progs['t2r'].display(ax=[ax[1, 0]])
    progs['t2e'].display(ax=[ax[1, 1]])
    progs['shot'].display(ax=[ax[0, 2], ax[1, 2]])
    progs['rspec'].display(ax=[ax[2, 0]])

    # Add readout parameters caption
    cap = f'Length: {auto_cfg.device.readout.readout_length[qi]:0.2f} $\mu$s'
    cap += f'\nGain: {auto_cfg.device.readout.gain[qi]}'
    ax[0, 2].text(0.02, 0.05, cap, transform=ax[0, 2].transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='left', 
                 bbox=dict(facecolor='white', alpha=0.8))

    # Display chi measurement
    chi_fig(ax[2, 1], qi, progs)
    
    # Display qubit spectroscopy
    progs['qspec'].display(ax=[ax[2, 2]], plot_all=False)
    ax[2, 2].set_title(f'Qubit Freq: {auto_cfg.device.qubit.f_ge[qi]:0.2f} MHz')
    ax[2, 2].axvline(x=auto_cfg.device.qubit.f_ge[qi], color='k', linestyle='--')

    fig.tight_layout()
    plt.show()
    
    # Save figure
    datestr = datetime.now().strftime("%Y%m%d_%H%M")
    fname = cfg_dict['expt_path'] + f'images\\summary\\qubit{qi}_tuneup_{datestr}.png'
    print(fname)
    fig.savefig(fname)


def chi_fig(ax, qi, progs):
    """
    Create a figure showing the dispersive shift (chi) measurement.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    qi : int
        Qubit index
    progs : dict
        Dictionary of measurement program objects
        
    Returns
    -------
    None
    """
    ax.set_title(f'Chi Measurement Q{qi}')
    ax.plot(progs['chid'][1].data['xpts'], progs['chid'][1].data['amps'], label='No Pulse')
    ax.plot(progs['chid'][0].data['xpts'], progs['chid'][0].data['amps'], label=f'e Pulse')
    
    chi_val = progs['chid'][0].data['chi_val']
    cap = f'$\chi=${chi_val:0.2f} MHz'
    ax.text(0.04, 0.35, cap, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='left', 
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Add vertical lines at key frequencies
    ax.axvline(x=progs['chid'][0].data['cval'], color='k', linestyle='--')
    ax.axvline(x=progs['chid'][0].data['rval'], color='k', linestyle='--')
    ax.legend()
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency (MHz)')


def find_spec(qi, cfg_dict, start="coarse", freq='ge', max_err=MAX_ERR, min_r2=MIN_R2):
    """
    Find qubit spectroscopy by iteratively scanning with different resolutions.
    
    This function attempts to find the qubit frequency by starting with a broad
    scan and progressively narrowing down with finer scans.
    
    Parameters
    ----------
    qi : int
        Qubit index
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    start : str, optional
        Starting scan resolution ('huge', 'coarse', 'medium', or 'fine')
    freq : str, optional
        Frequency transition to find ('ge' or 'ef')
    max_err : float, optional
        Maximum acceptable error in fits
    min_r2 : float, optional
        Minimum acceptable R² value for fits
        
    Returns
    -------
    tuple
        (success, attempts) where success is a boolean indicating whether the
        spectroscopy was found, and attempts is the number of attempts made
    """
    # Set up parameters based on transition type
    if freq == 'ef':
        f = 'f_ef'
        params = {'checkEF': True}
    else:
        f = 'f_ge'
        params = {}

    # Define scan resolutions
    style = ["huge", "coarse", "medium", "fine"]
    level = style.index(start)

    i = 0
    all_done = False
    ntries = 6
    
    # Iteratively try to find spectroscopy
    while i < ntries and not all_done:        
        print(f'Performing {style[level]} scan')
        prog = meas.QubitSpec(cfg_dict, qi=qi, min_r2=min_r2, params=params, style=style[level])
        
        if prog.status:
            # Update qubit frequency if scan successful
            config.update_qubit(cfg_dict["cfg_file"], f, prog.data["new_freq"], qi)
        
        if prog.status:
            if level == len(style) - 1:
                # If finest scan successful, we're done
                all_done = True
                print(f'Found qubit {qi}')
            else:
                # Move to finer scan
                level += 1
        else:
            # Move to coarser scan if current scan failed
            level -= 1
        
        if level < 0:
            # If coarsest scan failed, increase power and repetitions
            print('Coarsest scan failed, adding more power and reps')
            auto_cfg = config.load(cfg_dict["cfg_file"])
            config.update_readout(cfg_dict["cfg_file"], 'reps', auto_cfg.device.readout.reps[qi]*2, qi)
            config.update_qubit(cfg_dict["cfg_file"], 'spec_gain', auto_cfg.device.qubit.spec_gain[qi]*2, qi)
            level = 0
        
        i += 1

    if i == ntries:
        return False, i
    else:
        return True, i
    

def get_coherence(
    scan_name,
    qi=0,
    cfg_dict={},
    par="T1",
    params=None,
    min_r2=MIN_R2,
    max_err=MAX_ERR,
    tol=TOL,
    max_t1=MAX_T1,
):
    """
    Measure coherence time (T1, T2R, or T2E) with iterative refinement.
    
    This function repeatedly measures a coherence time until the result converges
    or the maximum number of iterations is reached.
    
    Parameters
    ----------
    scan_name : class
        Experiment class to use (T1Experiment or T2Experiment)
    qi : int, optional
        Qubit index
    cfg_dict : dict, optional
        Configuration dictionary containing experiment settings
    par : str, optional
        Parameter to measure ('T1', 'T2r', or 'T2e')
    params : dict, optional
        Additional parameters for the experiment
    min_r2 : float, optional
        Minimum acceptable R² value for fits
    max_err : float, optional
        Maximum acceptable error in fits
    tol : float, optional
        Tolerance for parameter convergence
    max_t1 : float, optional
        Maximum T1 time in microseconds
        
    Returns
    -------
    object
        Experiment program object
    """
    # Initialize parameters
    if params is None: 
        params = {}
    
    if par == 'T2e':
        params['experiment_type'] = 'echo'
        
    err = 2 * tol
    auto_cfg = config.load(cfg_dict["cfg_file"])
    old_par = auto_cfg["device"]["qubit"][par][qi]
    i = 0
    
    # Iteratively refine measurement until convergence
    while err > tol and i < 5:
        prog = scan_name(cfg_dict, qi=qi, params=params)
        
        if par == "T1":
            new_par = prog.data["new_t1_i"]
        else:
            if "best_fit" in prog.data:
                new_par = prog.data["fit_avgi"][3]
        
        if prog.status:
            # Update parameter if measurement successful
            auto_cfg = config.update_qubit(
                cfg_dict["cfg_file"], par, new_par, qi, sig=2, rng_vals=[1.5, max_t1]
            )
            err = np.abs(new_par - old_par) / old_par
        elif prog.data["fit_err"] > max_err:
            # If fit error too high, increase span
            print("Fit Error too high")
            params["span"] = 2 * old_par * 3  # Usually occurs because too little signal 
            err = 2 * tol
        else:
            # If measurement failed, increase averaging
            print("Failed")
            if 'soft_avgs' in params:
                params['soft_avgs'] = 2 * params['soft_avgs']
            else:
                params['soft_avgs'] = 2
            err = 2 * tol
            print('Increasing soft avgs due to fitting issues')

        old_par = new_par
        i += 1

    return prog


def recenter(
    qi, cfg_dict, max_err=MAX_ERR, min_r2=MIN_R2, max_t1=MAX_T1, style='coarse',
):
    """
    Recenter qubit frequency using Ramsey measurements.
    
    This function uses T2 Ramsey measurements to precisely determine and update
    the qubit frequency.
    
    Parameters
    ----------
    qi : int
        Qubit index
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    max_err : float, optional
        Maximum acceptable error in fits
    min_r2 : float, optional
        Minimum acceptable R² value for fits
    max_t1 : float, optional
        Maximum T1 time in microseconds
    style : str, optional
        Style of recentering ('coarse', 'fine', or 'giveup')
        
    Returns
    -------
    bool
        True if recentering was successful, False otherwise
    """
    # Get original frequency
    auto_cfg = config.load(cfg_dict["cfg_file"])
    start_freq = auto_cfg["device"]["qubit"]["f_ge"][qi]
    freqs = [start_freq]
    
    # Set initial Ramsey frequency based on T2r
    ramsey_freq = np.pi / 2 / auto_cfg.device.qubit.T2r[qi]
    if ramsey_freq < 0.2:
        freq = 0.2 
    else:
        freq = ramsey_freq
    
    params = {'ramsey_freq': freq}
    freq_error = []
    freq_adjust = 0.7  # Adjustment factor for frequency correction

    i = 0
    err = 1 
    tol = 0.02  # Tolerance for frequency convergence
    ntries = 3
    
    # Iteratively refine frequency measurement
    while i < ntries and err > tol:
        print(f"Try {i}")
        prog = meas.T2Experiment(cfg_dict, qi=qi, params=params)
        
        # If the scan succeeds or if the frequency error is small, update params
        if prog.status or prog.data["fit_err_par"][1] < max_err: 
            freq_error.append(prog.data["f_err"])
            err = np.abs(freq_error[-1])
            print(f"Scan successful. New frequency error is {freq_error[-1]:0.3f} MHz")
            freqs.append(prog.data["new_freq"])

            # Set new frequency and span, make sure span is not too large
            new_ramsey = err * freq_adjust 
            smart_ramsey = 1.5 / auto_cfg.device.qubit.T2r[qi]
            params['ramsey_freq'] = np.max([new_ramsey, smart_ramsey])
            span = np.pi / params['ramsey_freq']
            params['span'] = np.min([span, auto_cfg.device.qubit.T2r[qi] * 4])
             
            config.update_qubit(cfg_dict["cfg_file"], "f_ge", prog.data["new_freq"], qi)
            
            # Update T2r if scan was successful
            if prog.data["fit_err_par"][3] < max_err: 
                config.update_qubit(cfg_dict["cfg_file"], "T2r", prog.data["best_fit"][3], qi)        
        
        elif prog.data['r2'] > min_r2 and prog.data["fit_err_par"][1] > max_err:
            # Frequency error too high, increase frequency
            print('Frequency error too high, increasing frequency')
            err = 2 * tol
            params['ramsey_freq'] = 2 * params['ramsey_freq']
        
        elif prog.data['r2'] > min_r2 and prog.data['fit_err_par'][3] < 0.5: 
            # Update T2r if error is low
            config.update_qubit(cfg_dict["cfg_file"], "T2r", prog.data["best_fit"][3], qi)
        
        else: 
            # Assume that the qubit is so detuned that Rabi is not working or readout not tuned up
            print("Fit failed, trying spectroscopy.")
            status, i = find_spec(qi, cfg_dict, start='fine')
            if status:
                status = set_up_qubit(qi, cfg_dict)
                if status:
                    print('Qubit set up')
                else:
                    print('Failed to set up qubit')
                    i = ntries
            
            # Try different recentering approach if needed
            if style != 'giveup':
                recenter(qi, cfg_dict, style='giveup')
            else:
                print('Failed to recenter')

        i += 1

    print(f"Total Change in frequency: {freqs[-1] - freqs[0]:0.3f} MHz")
    print(freq_error)
    auto_cfg = config.load(cfg_dict["cfg_file"])
    end_freq = auto_cfg["device"]["qubit"]["f_ge"][qi]
    
    if err < tol:
        print(f"Qubit {qi} recentered from {start_freq} to {end_freq}")
    else:
        print(f"Qubit {qi} failed to recenter from {start_freq} to {end_freq}")

    if i == ntries:
        return False
    else:
        return True


def set_up_qubit(qi, cfg_dict):
    """
    Basic setup for a qubit, including pi pulse calibration and readout optimization.
    
    This function performs minimal calibration to get a qubit working, typically
    used as a recovery step when other calibration procedures fail.
    
    Parameters
    ----------
    qi : int
        Qubit index
    cfg_dict : dict
        Configuration dictionary containing experiment settings
        
    Returns
    -------
    bool
        True if setup was successful, False otherwise
    """
    cfg_path = cfg_dict['cfg_file']
   
    # Mark qubit as not tuned up
    config.update_qubit(cfg_path, 'tuned_up', False, qi)
    
    # Calibrate pi pulse
    amp_rabi = meas.RabiExperiment(cfg_dict, qi=qi)
    if amp_rabi.status:
        config.update_qubit(cfg_path, ('pulses', 'pi_ge', 'gain'), amp_rabi.data['pi_length'], qi)

    # Optimize readout
    shot = meas.HistogramExperiment(cfg_dict, qi=qi)            
    shot.update(cfg_path, fast=True)
    
    # Check if basic setup was successful
    if shot.data['fids'][0] > 0.1 and amp_rabi.status:
        return True
    else:
        return False


def meas_opt(cfg_dict, qubit_list, params=None, update=True, start_coarse=True, do_res=False):
    """
    Optimize readout parameters for one or more qubits.
    
    This function performs single-shot optimization to find the best readout
    parameters (gain and length) for each qubit.
    
    Parameters
    ----------
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    qubit_list : list
        List of qubit indices to optimize
    params : dict, optional
        Additional parameters for the optimization
    update : bool, optional
        Whether to update the configuration file with new parameters
    start_coarse : bool, optional
        Whether to start with coarse optimization
    do_res : bool, optional
        Whether to run resonator spectroscopy after each optimization step
        
    Returns
    -------
    None
    """
    if params is None:
        params = {}
    
    cfg_path = cfg_dict['cfg_file']
    
    for qi in qubit_list: 
        # Coarse optimization if requested
        if start_coarse:
            do_more = True
            while do_more:
                shotopt = meas.SingleShotOptExperiment(cfg_dict, qi=qi, params=params)
                do_more = shotopt.do_more
                
                if update:
                    config.update_readout(cfg_path, 'gain', shotopt.data['gain'], qi)
                    config.update_readout(cfg_path, 'readout_length', shotopt.data['length'], qi)
                
                if do_res:
                    run_res(cfg_dict, qi)
        
        # Fine optimization
        do_more = True
        while do_more:
            shotopt = meas.SingleShotOptExperiment(cfg_dict, qi=qi, params=params, style='fine')
            do_more = shotopt.do_more
            
            if update:
                config.update_readout(cfg_path, 'gain', shotopt.data['gain'], qi)   
                config.update_readout(cfg_path, 'readout_length', shotopt.data['length'], qi)
            
            if do_res and do_more:
                run_res(cfg_dict, qi)


def run_res(cfg_dict, qi):
    """
    Run resonator spectroscopy and update the resonator frequency.
    
    Parameters
    ----------
    cfg_dict : dict
        Configuration dictionary containing experiment settings
    qi : int
        Qubit index
        
    Returns
    -------
    None
    """
    rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span': 'kappa'})
    rspec.update(cfg_dict['cfg_file'], freq=True)
