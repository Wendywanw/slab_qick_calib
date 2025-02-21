import numpy as np
from qick import *

from exp_handling.datamanagement import AttrDict
from datetime import datetime
import fitting as fitter
from gen.qick_experiment import QickExperiment, QickExperiment2D
from gen.qick_program import QickProgram
from qick.asm_v2 import QickSweep1D


class T1Program(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.add_loop("wait_loop", cfg.expt.expts)
        
        super()._initialize(cfg, readout="standard")

        super().make_pi_pulse(
            cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge"
        )
        
        if cfg.expt.acStark:
            pulse = {
                "sigma": cfg.expt.wait_time,
                "sigma_inc": 0,
                "freq": cfg.expt.stark_freq,
                "gain": cfg.expt.stark_gain,
                "phase": 0,
                "type": "const",
            }
            super().make_pulse(pulse, "stark_pulse")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.acStark:
            self.delay_auto(t=0.01, tag="wait_stark")
            self.pulse(ch=self.qubit_ch, name="stark_pulse", t=0)
            self.delay_auto(t=0.01, tag="wait")
        else:
            self.delay_auto(t=cfg.expt["wait_time"] + 0.01, tag="wait")
        
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )
        if cfg.expt.active_reset:
            self.reset(3)

    def collect_shots(self, offset=0):
        return super().collect_shots(offset=0)
    
    def reset(self, i):
        super().reset(i)

class T1Experiment(QickExperiment):
    """
    self.cfg.expt: dict
        A dictionary containing the configuration parameters for the T1 experiment. The keys and their descriptions are as follows:
        - span (float): The total span of the wait time sweep in microseconds.
        - expts (int): The number of experiments to be performed.
        - reps (int): The number of repetitions for each experiment (inner loop)
        - soft_avgs (int): The number of soft_avgs for the experiment (outer loop)
        - qubit (int): The index of the qubit being used in the experiment.
        - qubit_chan (int): The channel of the qubit being read out.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
    ):

        if prefix is None:
            prefix = f"t1_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "reps": 2 * self.reps,
            "soft_avgs": self.soft_avgs,
            "expts": 60,
            "start": 0,
            "span": 3.7 * self.cfg.device.qubit.T1[qi],
            "acStark": False,
            'active_reset': self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }

        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        #params_def.update(params)
        #self.cfg.expt = params_def
        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {'plot_all': True}
        if go:
            super().run(display=display, progress=progress, min_r2=min_r2, max_err=max_err, disp_kwargs=disp_kwargs)

    def acquire(self, progress=False, debug=False):
        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        # qi = self.cfg.expt.qubit[0]
        # self.cfg.expt.readout_length = QickSweep1D(
        #     "wait_loop", self.cfg.expt.start+self.cfg.device.readout.readout_length[qi], self.cfg.expt.start + self.cfg.expt.span+self.cfg.device.readout.readout_length[qi]
        # )
        super().acquire(T1Program, progress=progress)

        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        # fitparams=[y-offset, amp, x-offset, decay rate]
        fitfunc = fitter.expfunc
        fitterfunc = fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data, **kwargs)
        data["new_t1"] = data["best_fit"][2]
        data["new_t1_i"] = data["fit_avgi"][2]
        return data

    def display(
        self, data=None, fit=True, plot_all=False, ax=None, show_hist=False, rescale=False,**kwargs
    ):
        qubit = self.cfg.expt.qubit[0]
        title = f"$T_1$ Q{qubit}"
        xlabel = "Wait Time ($\mu$s)"


        caption_params = [
            {"index": 2, "format": "$T_1$ fit: {val:.3} $\pm$ {err:.2} $\mu$s"},           
        ]
        fitfunc = fitter.expfunc

        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=show_hist,
            fitfunc=fitfunc,
            caption_params=caption_params,
            rescale=rescale,
        )

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class T1Continuous(QickExperiment):
    """
    T1 Continuous
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        soft_avgs: number soft_avgs to repeat experiment sweep
    )
    """

    def __init__(
        self,
        soccfg=None,
        path="",
        prefix="T1Continuous",
        config_file=None,
        progress=None,
    ):
        super().__init__(
            soccfg=soccfg,
            path=path,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False, debug=False):

        self.update_config(q_ind=self.cfg.expt.qubit)
        t1 = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = t1.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
            debug=debug,
        )

        shots_i, shots_q = t1.collect_shots()

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
        phases = np.angle(avgi + 1j * avgq)  # Calculating the phase

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        data = {
            "xpts": x_pts,
            "avgi": avgi,
            "avgq": avgq,
            "amps": amps,
            "phases": phases,
            "time": current_time,
            "raw_i": shots_i,
            "raw_q": shots_q,
            "raw_amps": np.abs(shots_i + 1j * shots_q),
        }

        self.data = data
        return data

    def analyze(self, data=None, **kwargs):
        pass

    def display(self, data=None, fit=True, show=False, **kwargs):
        pass

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


class T1_2D(QickExperiment2D):
    """
    sweep_pts = number of points in the 2D sweep
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=None,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix is None:
            prefix = f"t1_2d_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "expts": 60,
            "span": 3.7 * self.cfg.device.qubit.T1[qi],
            "reps": 2 * self.reps,
            "soft_avgs": self.soft_avgs,
            "start": 0,
            "sweep_pts": 200,
            "qubit": qi,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 30
        params = {**params_def, **params}

        params["step"] = params["span"] / params["expts"]
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):

        sweep_pts = np.arange(self.cfg.expt["sweep_pts"])
        y_sweep = [{"pts": sweep_pts, "var": "count"}]
        super().acquire(T1Program, y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        fitfunc = fitter.expfunc
        fitterfunc = fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data)

    def display(self, data=None, fit=True, ax=None, **kwargs):
        if data is None:
            data = self.data

        title = f"$T_1$ 2D Q{self.cfg.expt.qubit}"
        xlabel = f"Wait Time ($\mu$s)"
        ylabel = "Time (s)"

        super().display(
            data=data,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            fit=fit,
            **kwargs,
        )

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname
