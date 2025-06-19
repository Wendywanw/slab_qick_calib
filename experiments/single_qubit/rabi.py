"""
Rabi Oscillation Experiment

This module implements Rabi oscillation experiments for qubit characterization.
Rabi oscillations are observed by varying either the amplitude or length of a driving pulse
and measuring the resulting qubit state. This allows determination of the π-pulse parameters
(amplitude and duration) needed for qubit control.

The module includes:
- RabiProgram: Defines the pulse sequence for the Rabi experiment
- RabiExperiment: Main experiment class for amplitude or length Rabi oscillations
- ReadoutCheck: Class for checking readout parameters
- RabiChevronExperiment: 2D version that sweeps both frequency and amplitude/length
"""

import matplotlib.pyplot as plt
import numpy as np
from qick import *

from exp_handling.datamanagement import AttrDict
from gen.qick_experiment import QickExperiment, QickExperiment2DSimple
from gen.qick_program import QickProgram
import slab_qick_calib.fitting as fitter
from qick.asm_v2 import QickSweep1D
from scipy.optimize import curve_fit
# ====================================================== #


class RabiProgram(QickProgram):
    """
    Defines the pulse sequence for a Rabi oscillation experiment.
    
    The sequence consists of:
    1. Optional π pulse on |g>-|e> transition (if checking EF transition)
    2. Variable amplitude/length pulse on the qubit
    3. Optional second π pulse on |g>-|e> transition
    4. Optional wait time
    5. Measurement
    """
    def __init__(self, soccfg, final_delay, cfg):
        """
        Initialize the Rabi program.
        
        Args:
            soccfg: SOC configuration
            final_delay: Delay time after measurement
            cfg: Configuration dictionary
        """
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """
        Initialize the program with the necessary pulses and loops.
        
        Args:
            cfg: Configuration dictionary containing experiment parameters
        """
        cfg = AttrDict(self.cfg)
        q = cfg.expt.qubit[0]

        # Initialize with standard readout
        super()._initialize(cfg, readout="standard")
        
        # Add sweep loop for the experiment
        self.add_loop("sweep_loop", cfg.expt.expts)

        # Define the qubit pulse with parameters from config
        pulse = {
            "sigma": cfg.expt.sigma,
            "length": cfg.expt.length,
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain,
            "phase": 0,
            "type": cfg.expt.type,
        }
        super().make_pulse(pulse, "qubit_pulse")

        # If checking EF transition and using ge pulse, create a pi pulse
        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            super().make_pi_pulse(q, cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        """
        Define the main body of the experiment sequence.
        
        Args:
            cfg: Configuration dictionary containing experiment parameters
        """
        cfg = AttrDict(self.cfg)
        
        # Configure readout
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # If checking EF transition with ge pulse, apply first pi pulse
        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef")

        # Apply the main qubit pulse (variable amplitude or length)
        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        self.delay_auto(t=0.01, tag="wait")

        # If checking EF transition with ge pulse, apply second pi pulse
        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef 2")
            
        # Add optional end wait time
        if 'end_wait' in cfg.expt:
            self.delay_auto(t=cfg.expt.end_wait, tag="end_wait")
        
        # Perform measurement
        super().measure(cfg)

class RabiExperiment(QickExperiment):
    """
    Main experiment class for Rabi oscillations.
    
    This class implements Rabi oscillation experiments by sweeping either the amplitude
    or length of a driving pulse and measuring the resulting qubit state. The oscillation
    pattern allows determination of the π-pulse parameters needed for qubit control.
    
    Parameters:
    - 'expts': Number of experiments to run (default: 60)
    - 'reps': Number of repetitions for each experiment (default: self.reps)
    - 'soft_avgs': Number of soft_avgs for each experiment (default: self.soft_avgs)
    - 'gain': Max gain value for the pulse (default: gain)
    - 'sigma': Standard deviation of the Gaussian pulse (default: sigma)
    - 'checkEF': Boolean flag to check EF interaction (default: False)
    - 'pulse_ge': Boolean flag to indicate if pulse is for ground to excited state transition (default: True)
    - 'start': Starting point for the experiment (default: 0)
    - 'step': Step size for the gain (calculated as int(params['gain']/params['expts']))
    - 'qubit': List of qubits involved in the experiment (default: [qi])
    - 'pulse_type': Type of pulse used in the experiment (default: 'gauss')
    - 'num_pulses': Number of pulses used in the experiment (default: 1)
    - 'qubit_chan': Channel for the qubit readout (default: self.cfg.hw.soc.adcs.readout.ch[qi])
    - 'sweep': Type of sweep to perform ('amp' or 'length') (default: 'amp')
    - 'freq': Frequency of the qubit pulse (default: self.cfg.device.qubit.f_ge[qi])
    
    Additional keys may be added based on the specific requirements of the experiment.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        display=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        print=False
    ):
        
        if 'checkEF' in params and params['checkEF']:
            if 'pulse_ge' in params and not params['pulse_ge']:
                ef = "ef_no_ge_"
            else: ef = "ef_"
        else: ef= ""
        name = 'length' if 'sweep' in params and params['sweep'] == 'length' else 'amp'

        prefix = f"{name}_rabi_{ef}qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)
        params_def = {
            "expts": 60,
            "reps": self.reps,
            "soft_avgs": self.soft_avgs,
            "checkEF": False,
            "pulse_ge": True,
            "sweep": "amp",
            'active_reset': self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }        

        min_gain=2**-15
        # Apply style modifications
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 25    

        params = {**params_def, **params}
        
        # Pulse params
        if params['checkEF']:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]
        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]
        params = {**params_def, **params}
        
        if params["sweep"]=="amp":
            params_def['max_gain'] = params['gain'] * 5
            params_def['start']=0.003 # This is currently the minimum gain value that is linear 
            params_def["max_gain"]=np.min([params_def["max_gain"], self.cfg.device.qubit.max_gain])
            
        elif params["sweep"]=="length":
            params_def["max_length"] = 5 * params["sigma"]
            params_def["start"] = 2*cfg_dict['soc'].cycles2us(1)
        
        if style == "temp":
            params["reps"] = 40 * params["reps"]
            params["soft_avgs"] = 20 * params["soft_avgs"]
            params["pulse_ge"] = False
        
        self.cfg.expt = {**params_def, **params}
        if params['sweep']=='amp':
            gain_spc =self.cfg.expt['max_gain']/self.cfg.expt['expts']
            if gain_spc < min_gain:
                self.cfg.expt['max_gain'] = min_gain * self.cfg.expt['expts']
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()
        
        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {'plot_all': True}

        if print: 
            super().print()
            go=False
        if go:
            super().run(display=display, progress=progress, min_r2=min_r2, max_err=max_err, disp_kwargs=disp_kwargs)

    def acquire(self, progress=False, debug=False):
        """
        Acquire data for the Rabi experiment.
        
        Args:
            progress: Whether to show progress bar
            debug: Whether to print debug information
            
        Returns:
            Acquired data
        """
        self.qubit = self.cfg.expt.qubit
        
        # Configure the sweep based on whether we're sweeping amplitude or length
        # Note: 2d scans will break if you use gain/length to store the max_vals of gain/length when making qicksweep
        if self.cfg.expt.sweep == "amp":
            # Amplitude sweep configuration
            param_pulse = 'gain'
            self.cfg.expt['gain'] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt['max_gain']
            )
            if self.cfg.expt.type == 'gauss':       
                self.cfg.expt['length'] = self.cfg.expt.sigma * self.cfg.expt.sigma_inc
            else:
                self.cfg.expt['length'] = self.cfg.expt.sigma
        elif self.cfg.expt.sweep == "length":
            # Length sweep configuration
            param_pulse = 'total_length'
            if self.cfg.expt.type == 'gauss':
                # For Gaussian pulses, sweep sigma
                par = 'sigma'
                self.cfg.expt['length'] = QickSweep1D(
                    "sweep_loop", self.cfg.expt.start, 4*self.cfg.expt['max_length'])
            else:
                # For other pulse types, sweep length directly
                par = 'length'
            self.cfg.expt[par] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt['max_length']
            )
            
        # Set the parameter to sweep
        self.param = {"label": "qubit_pulse", "param": param_pulse, "param_type": "pulse"}
        
        # Acquire data using the RabiProgram
        super().acquire(RabiProgram, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data to extract Rabi oscillation parameters.
        
        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data to a sinusoidal function
            **kwargs: Additional arguments for the fit
            
        Returns:
            Analyzed data with fit parameters and π-pulse length
        """
        if data is None:
            data = self.data

        if fit:
            # Fit the data to a sinusoidal function
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            self.fitterfunc = fitter.fitsin
            self.fitfunc = fitter.sinfunc
            data = super().analyze(
                fitfunc=self.fitfunc, fitterfunc=self.fitterfunc, fit=fit, **kwargs
            )

        # Calculate π-pulse length from the fit for each data type
        ydata_lab = ["amps", "avgi", "avgq"]
        for ydata in ydata_lab:
            pi_length = fitter.fix_phase(data["fit_" + ydata])
            data["pi_length_" + ydata] = pi_length
            
        # Store the best π-pulse length
        data["pi_length"] = fitter.fix_phase(data["best_fit"])
        return data

    def display(
        self,
        data=None,
        fit=True,
        plot_all=False,
        ax=None,
        show_hist=False,
        rescale=False,
        save_fig=True,
        return_fig=False,
        **kwargs,
    ):
        """
        Display the results of the Rabi experiment.
        
        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit curve
            plot_all: Whether to plot all data types (I, Q, amplitude)
            ax: Matplotlib axis to plot on
            show_hist: Whether to show histogram
            rescale: Whether to rescale the plot
            save_fig: Whether to save the figure
            return_fig: Whether to return the figure as base64 string
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        # Set up plot title and labels based on sweep type
        q = self.cfg.expt.qubit[0]
        if self.cfg.expt.sweep == "amp":
            title = 'Amplitude'
            param = 'sigma'
            xlabel = "Gain / Max Gain"
        else:
            title = 'Length'
            param = 'gain'
            xlabel = "Pulse Length ($\mu$s)"
            
        title += f" Rabi Q{q} (Pulse {param} {self.cfg.expt[param]}"

        # Set up fit function and caption parameters
        caption_params = [{'index': "pi_length", 
                          'format': "$\pi$ length: {val:.3f}"}]
        
        # Add EF indicator to title if applicable
        if self.cfg.expt.checkEF:
            title = title + ", EF)"
        else:
            title = title + ")"

        # Display the results
        return super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=show_hist,
            fitfunc=self.fitfunc,
            caption_params=caption_params,
            rescale=rescale,
            save_fig=save_fig,
            return_fig=return_fig,
        )

class ReadoutCheck(QickExperiment):
    """
    Class for checking readout parameters.
    
    This experiment is used to characterize the readout by sweeping either
    the end wait time or the gain of a qubit pulse and measuring the response.
    It uses the same RabiProgram as the RabiExperiment but with different parameters.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        display=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
    ):
        """
        Initialize the ReadoutCheck experiment.
        
        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index
            go: Whether to run the experiment immediately
            params: Additional parameters to override defaults
            prefix: Prefix for data files
            progress: Whether to show progress bar
            display: Whether to display results
            style: Style of experiment ("fine" or "fast")
            disp_kwargs: Display keyword arguments
            min_r2: Minimum R² value for fit quality
            max_err: Maximum error for fit quality
        """
        # Set the prefix for data files
        prefix = f"'readout_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)
        
        # Default parameters
        params_def = {
            "expts": 30,
            "reps": 5*self.reps,
            "soft_avgs": self.soft_avgs,
            "checkEF": False,
            "pulse_ge": True,
            'active_reset': self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            'df': 50,
            'qubit_freq': self.cfg.device.qubit.f_ge[qi],
            'phase': 0,
            'length': 10,
            'gain': 0.5,
            'type': 'const',
            'max_wait': 10,
            'max_gain': 1,
            'end_wait': 0.2,
            'start': 0,
            'expt_type': 'end_wait'  # Can be 'end_wait' or 'gain'
        }        

        min_gain = 2**-15
        
        # Apply style modifications
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 25    

        # Merge default and user-provided parameters
        params = {**params_def, **params}
        params['sigma'] = params['length']  # Set sigma equal to length
        params_def["freq"] = params['qubit_freq'] + params['df']  # Set frequency with offset
        
        # Set experiment configuration
        self.cfg.expt = {**params_def, **params}
        
        # Check parameters and configure reset if needed
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()
        
        # Set display parameters for untuned qubits
        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {'plot_all': True}
            
        # Run the experiment if requested
        if go:
            super().run(display=display, progress=progress, min_r2=min_r2, max_err=max_err, disp_kwargs=disp_kwargs)

    def acquire(self, progress=False, debug=False, single=False):
        """
        Acquire data for the ReadoutCheck experiment.
        
        Args:
            progress: Whether to show progress bar
            debug: Whether to print debug information
            single: Whether to run a single acquisition
            
        Returns:
            Acquired data
        """
        self.qubit = self.cfg.expt.qubit

        # Configure the sweep based on experiment type
        if self.cfg.expt.expt_type == 'end_wait':
            # Sweep end wait time
            self.cfg.expt['end_wait'] = QickSweep1D("sweep_loop", self.cfg.expt.start, self.cfg.expt['max_wait'])
            self.param = {"label": "end_wait", "param": "t", "param_type": "time"}
        else:
            # Sweep gain
            self.cfg.expt['gain'] = QickSweep1D("sweep_loop", self.cfg.expt.start, self.cfg.expt['max_gain'])
            self.param = {"label": "qubit_pulse", "param": "gain", "param_type": "pulse"}
            
        # Acquire data using the RabiProgram
        super().acquire(RabiProgram, progress=progress, single=single)

        return self.data
    
    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data.
        
        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data
            **kwargs: Additional arguments for the fit
            
        Returns:
            Analyzed data
        """
        if data is None:
            data = self.data
        # No specific analysis for ReadoutCheck, just return the data
        return data
    
    def display(self, data=None, fit=False, plot_all=False, save_fig=True, return_fig=False, **kwargs):
        """
        Display the results of the ReadoutCheck experiment.
        
        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_all: Whether to plot all data types
            save_fig: Whether to save the figure
            return_fig: Whether to return the figure as base64 string
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        # Use the parent class display method
        return super().display(data=data, fit=fit, plot_all=plot_all, save_fig=save_fig, return_fig=return_fig, **kwargs)

class RabiChevronExperiment(QickExperiment2DSimple):
    """
    2D Rabi experiment that sweeps both frequency and amplitude/length.
    
    This experiment performs a 2D sweep of both qubit frequency and pulse amplitude/length
    to map out the Rabi chevron pattern. This allows visualization of how the Rabi
    oscillation frequency changes with detuning from the qubit frequency.
    
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz),
        step_f: frequency step (MHz),
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        soft_avgs: number repetitions of experiment sweep
        sigma: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        style="",
        prefix=None,
        progress=False,
    ):  
        """
        Initialize the RabiChevronExperiment.
        
        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index
            go: Whether to run the experiment immediately
            params: Additional parameters to override defaults
            style: Style of experiment
            prefix: Prefix for data files
            progress: Whether to show progress bar
        """
        # Determine prefix based on parameters
        if 'type' in params:
            pre = params['type']
        else:
            pre = "amp"
        if 'checkEF' in params and params['checkEF']: 
            ef = "ef"
        else: 
            ef = ""
        prefix = f"{pre}_rabi_chevron_{ef}_qubit{qi}"         

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        # Default parameters
        params_def = {
            'span_f': 20,
            'expts_f': 30,
            'sweep': 'amp'
        }
        params = {**params_def, **params}
        
        # Set frequency range based on whether we're checking EF transition
        if 'checkEF' in params and params["checkEF"]:
            params_def['start_f'] = self.cfg.device.qubit.f_ef[qi] - params["span_f"] / 2
        else:
            params_def['start_f'] = self.cfg.device.qubit.f_ge[qi] - params["span_f"] / 2
       
        # Create a RabiExperiment instance but don't run it yet
        self.expt = RabiExperiment(cfg_dict, qi=qi, go=False, params=params, style=style)
        params = {**params_def, **params}
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params
        
        # Run the experiment if requested
        if go:
            super().run(progress=progress)
        

    def acquire(self, progress=False, debug=False):
        """
        Acquire data for the RabiChevronExperiment.
        
        Args:
            progress: Whether to show progress bar
            debug: Whether to print debug information
            
        Returns:
            Acquired data
        """
        # Create frequency points for the sweep
        freqpts = np.linspace(
            self.cfg.expt["start_f"], 
            self.cfg.expt["start_f"] + self.cfg.expt["span_f"], 
            self.cfg.expt["expts_f"]
        )
        
        # Set up the y-sweep (frequency sweep)
        ysweep = [{"pts": freqpts, "var": "freq"}]
        
        # Acquire data
        super().acquire(ysweep, progress=progress)
        
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data.
        
        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data
            **kwargs: Additional arguments for the fit
            
        Returns:
            Analyzed data with fit parameters
        """
        if data is None:
            data = self.data

        if fit:
            # Fit the data to a sinusoidal function for each frequency
            fitterfunc = fitter.fitsin
            fitfunc = fitter.sinfunc
            data = super().analyze(
                fitfunc=fitfunc, fitterfunc=fitterfunc, fit=fit, **kwargs
            )
            
            # Extract qubit frequency and fit parameters
            qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]
            
            # Fit the chevron pattern
            try:
                p, _ = curve_fit(chevron_freq, data["ypts"]-qubit_freq, freq)
                p2, _ = curve_fit(chevron_amp, data["ypts"]-qubit_freq, amp)
                data["chevron_freq"] = p
                data["chevron_amp"] = p2
            except:
                # Silently fail if the fit doesn't converge
                pass
                
        return data

    def display(self, data=None, fit=True, plot_both=False, save_fig=True, return_fig=False, **kwargs):
        """
        Display the results of the RabiChevronExperiment.
        
        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_both: Whether to plot both amplitude and phase
            save_fig: Whether to save the figure
            return_fig: Whether to return the figure as base64 string
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data
            
        # Set up plot title and labels
        if self.cfg.expt.checkEF: 
            title = 'EF'
        else: 
            title = ''
            
        if self.cfg.expt.sweep == "amp":
            title = 'Amplitude'
            param = 'sigma'
            xlabel = "Gain / Max Gain"
        else:
            title = 'Length'
            param = 'gain'
            xlabel = "Pulse Length ($\mu$s)"
            
        title += f" Rabi Q{self.cfg.expt.qubit[0]} (Pulse {param} {self.cfg.expt[param]})"

        xlabel = xlabel
        ylabel = "Frequency (MHz)"

        # Display the 2D plot
        fig = super().display(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            data=data,
            fit=fit,
            plot_both=plot_both,
            save_fig=save_fig,
            return_fig=return_fig,
            **kwargs,
        )

        # If fit is enabled, also display the frequency and amplitude vs. detuning
        if fit: 
            fig2, ax = plt.subplots(2, 1, figsize=(6, 6))
            qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]
            
            # Plot frequency vs. detuning
            ax[0].plot(data["ypts"]-qubit_freq, freq)
            ax[0].set_ylabel('Frequency (MHz)')
            
            # Plot amplitude vs. detuning
            ax[1].plot(data["ypts"]-qubit_freq, amp)
            ax[1].set_xlabel('$\Delta$ Frequency (MHz)')
            ax[1].set_ylabel('Amplitude')
            
            # Save or return the second figure if requested
            if save_fig:
                fig2.savefig(self.fname[0:-3] + "_chevron.png")
            
            if return_fig:
                import io
                import base64
                image_buffer = io.BytesIO()
                fig2.savefig(image_buffer, format='png', bbox_inches='tight')
                image_buffer.seek(0)
                image_bytes = image_buffer.getvalue()
                image_buffer.close()
                chevron_base64 = base64.b64encode(image_bytes).decode('utf-8')
                return [fig, chevron_base64]
            
            plt.show()
            
        return fig

# Helper functions for fitting the chevron pattern

def chevron_freq(x, w0): 
    """
    Calculate the Rabi frequency as a function of detuning.
    
    The Rabi frequency is given by sqrt(w0^2 + x^2), where w0 is the
    on-resonance Rabi frequency and x is the detuning.
    
    Args:
        x: Detuning from resonance
        w0: On-resonance Rabi frequency
        
    Returns:
        Rabi frequency
    """
    return np.sqrt(w0**2 + x**2)

def chevron_amp(x, w0, a):
    """
    Calculate the Rabi oscillation amplitude as a function of detuning.
    
    The amplitude is given by a/(1 + (x/w0)^2), where a is the
    on-resonance amplitude, w0 is related to the on-resonance Rabi
    frequency, and x is the detuning.
    
    Args:
        x: Detuning from resonance
        w0: Width parameter
        a: On-resonance amplitude
        
    Returns:
        Oscillation amplitude
    """
    return a/(1 + (x/w0)**2)
