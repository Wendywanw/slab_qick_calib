import numpy as np
import matplotlib.pyplot as plt

from qick import *
from exp_handling.datamanagement import AttrDict
from gen.qick_experiment import QickExperiment
from gen.qick_program import QickProgram

"""
Run this calibration when the wiring of the setup is changed.

This calibration measures the time of flight of measurement pulse so we only start capturing data from this point in time onwards. Time of flight (tof) is stored in parameter cfg.device.readout.trig_offset.
"""


class LoopbackProgram(QickProgram):
    
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay, cfg)
    
    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        self.readout_length = cfg.expt.readout_length
        self.phase = 0
        super()._initialize(cfg, readout="custom")

    def _body(self, cfg):
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=0,
            ddr4=True,
        )

# ====================================================== #

class ToFCalibrationExperiment(QickExperiment):
    """
    Time of flight experiment
    Experimental Config
    expt_cfg = dict(
        readout_length [us]
        gain [DAC units]
        frequency [MHz]
        adc_trig_offset [Clock ticks]
    }
    """

    def __init__(
        self,
        cfg_dict={},
        progress=None,
        prefix=None,
        qi=0,
        params={},
        go=True,
    ):
        if prefix is None:
            prefix = f"adc_trig_offset_calibration_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)
        params_def = {
            "soft_avgs": 1000,
            "readout_length": 3,  # [us]
            "trig_offset": self.cfg.device.readout.trig_offset[qi],  # [us]
            "gain": self.cfg.device.readout.max_gain,
            "frequency": self.cfg.device.readout.frequency[qi],  # [MHz]
            "reps": 1,  # Number of averages per point
            "qubit": [qi],
            "final_delay": 0.1,
        }
        self.cfg.expt = {**params_def, **params}

        if go:
            self.go(analyze=False, display=False, progress=True, save=True)
            self.display(adc_trig_offset=self.cfg.expt.trig_offset)

    def acquire(self, progress=False):
        
        final_delay = 10

        prog = LoopbackProgram(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )
        iq_list = prog.acquire_decimated(self.im[self.cfg.aliases.soc],
            soft_avgs=self.cfg.expt.soft_avgs,)
        t = prog.get_time_axis(ro_index=0)
        i  = iq_list[0][:,0]
        q  = iq_list[0][:,1]
        plt.show()
        amp = np.abs(i + 1j * q)  # Calculating the magnitude
        phase = np.angle(i + 1j * q)  # Calculating the phase

        data = {'xpts':t, 'i':i, 'q':q, 'amps':amp, 'phases':phase}

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=False, findpeaks=False, **kwargs):
        if data is None:
            data = self.data
        return data

    def display(self, data=None, adc_trig_offset=0, save_fig=True, **kwargs):
        if data is None:
            data = self.data

        q_ind = self.cfg.expt.qubit[0]
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]
        dac_ch = self.cfg.hw.soc.dacs.readout.ch[q_ind]
        fig, ax = plt.subplots(1,1, figsize=(8,3))
        ax.set_title(f"Time of Flight: DAC Ch. {dac_ch} to ADC Ch. {adc_ch}, f: {self.cfg.expt.frequency} MHz")
        ax.set_xlabel("Time ($\mu$s)")
        ax.set_ylabel("Transmission (ADC units)")

        plt.plot(data["xpts"], data["i"], label="I")
        plt.plot(data["xpts"], data["q"], label="Q")
        plt.axvline(adc_trig_offset, c="k", ls="--")
        plt.legend()
        plt.show()

        if save_fig:  # Save figure if save_fig is True
            imname = self.fname.split("\\")[-1]
            fig.tight_layout()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
