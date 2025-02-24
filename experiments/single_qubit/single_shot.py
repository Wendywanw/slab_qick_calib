import matplotlib.pyplot as plt
import numpy as np
from qick import *
import copy
import seaborn as sns
from exp_handling.datamanagement import AttrDict
from tqdm import tqdm_notebook as tqdm
from gen.qick_experiment import QickExperiment
from gen.qick_program import QickProgram
from scipy.optimize import curve_fit
from scipy.special import erf
from copy import deepcopy

blue = "#4053d3"
red = "#b51d14"
int_rgain = True

# Make it possible to turn fitting off 
def hist(data, plot=True, span=None, ax=None, verbose=False, qubit=0):
    """
    span: histogram limit is the mean +/- span
    """
    # FIXME: f state analysis is broken
    Ig = data["Ig"]
    Qg = data["Qg"]
    Ie = data["Ie"]
    Qe = data["Qe"]
    if "If" in data.keys():
        plot_f = True
        If = data["If"]
        Qf = data["Qf"]
    else:
        plot_f = False

    numbins = 100

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f:
        xf, yf = np.median(If), np.median(Qf)

    if verbose:
        print("Unrotated:")
        print(
            f"Ig {xg:0.3f} +/- {np.std(Ig):0.3f} \t Qg {yg:0.3f} +/- {np.std(Qg):0.3f} \t Amp g {np.abs(xg+1j*yg):0.3f}"
        )
        print(
            f"Ie {xe:0.3f} +/- {np.std(Ie):0.3f} \t Qe {ye:0.3f} +/- {np.std(Qe):0.3f} \t Amp e {np.abs(xe+1j*ye):0.3f}"
        )
        if plot_f:
            print(
                f"If {xf:0.3f} +/- {np.std(If)} \t Qf {yf:0.3f} +/- {np.std(Qf):0.3f} \t Amp f {np.abs(xf+1j*yf):0.3f}"
            )

    """Compute the rotation angle"""
    theta = -np.arctan2((ye - yg), (xe - xg))
    if plot_f:
        theta = -np.arctan2((yf - yg), (xf - xg))

    """Rotate the IQ data"""
    Ig_new = Ig * np.cos(theta) - Qg * np.sin(theta)
    Qg_new = Ig * np.sin(theta) + Qg * np.cos(theta)

    Ie_new = Ie * np.cos(theta) - Qe * np.sin(theta)
    Qe_new = Ie * np.sin(theta) + Qe * np.cos(theta)

    if plot_f:
        If_new = If * np.cos(theta) - Qf * np.sin(theta)
        Qf_new = If * np.sin(theta) + Qf * np.cos(theta)

    """Post-rotation means of each blob"""
    xg_new, yg_new = np.median(Ig_new), np.median(Qg_new)
    xe_new, ye_new = np.median(Ie_new), np.median(Qe_new)
    if plot_f:
        xf, yf = np.median(If_new), np.median(Qf_new)
    if verbose:
        print("Rotated:")
        print(
            f"Ig {xg_new:.3f} +/- {np.std(Ig):.3f} \t Qg {yg_new:.3f} +/- {np.std(Qg):.3f} \t Amp g {np.abs(xg_new+1j*yg_new):.3f}"
        )
        print(
            f"Ie {xe_new:.3f} +/- {np.std(Ie):.3f} \t Qe {ye_new:.3f} +/- {np.std(Qe):.3f} \t Amp e {np.abs(xe_new+1j*ye_new):.3f}"
        )
        if plot_f:
            print(
                f"If {xf:.3f} +/- {np.std(If)} \t Qf {yf:.3f} +/- {np.std(Qf):.3f} \t Amp f {np.abs(xf+1j*yf):.3f}"
            )

    if span is None:
        span = (
            np.max(np.concatenate((Ie_new, Ig_new)))
            - np.min(np.concatenate((Ie_new, Ig_new)))
        ) / 2
    xlims = [(xg_new + xe_new) / 2 - span, (xg_new + xe_new) / 2 + span]
    ylims = [yg_new - span, yg_new + span]

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []

    """X and Y ranges for histogram"""

    ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
    ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
    if plot_f:
        nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    contrast = np.abs(
        ((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum()))
    )
    tind = contrast.argmax()
    thresholds.append(binsg[tind])
    fids.append(contrast[tind])
    err_e = np.cumsum(ne)[tind]/ne.sum()
    err_g = 1-np.cumsum(ng)[tind]/ng.sum()

    if plot_f:
        contrast = np.abs(
            ((np.cumsum(ng) - np.cumsum(nf)) / (0.5 * ng.sum() + 0.5 * nf.sum()))
        )
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        contrast = np.abs(
            ((np.cumsum(ne) - np.cumsum(nf)) / (0.5 * ne.sum() + 0.5 * nf.sum()))
        )
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])
    m = 0.7
    a = 0.25
    if plot:
        if ax is None:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
            ax = [axs[0,1],axs[1,0]]
            
            # Plot unrotated data 
            axs[0, 0].plot(Ig, Qg, ".", label="g", color=blue, alpha=a, markersize=m)
            axs[0, 0].plot(Ie, Qe, ".", label="e", color=red, alpha=a, markersize=m)
            if plot_f:
                axs[0, 0].plot(If, Qf, ".", label="f", color="g", alpha=a, markersize=m)
            axs[0, 0].plot(xg, yg, color="k", marker="o")
            axs[0, 0].plot(xe, ye, color="k", marker="o")
            if plot_f:
                axs[0, 0].plot(xf, yf, color="k", marker="o")

            axs[0,0].set_xlabel('I (ADC levels)')
            axs[0, 0].set_ylabel("Q (ADC levels)")
            axs[0, 0].legend(loc="upper right")
            axs[0, 0].set_title("Unrotated")
            axs[0, 0].axis("equal")
            set_fig=True

            # Plot log histogram         
            bin_cent = (binsg[1:] + binsg[:-1]) / 2
            axs[1,1].semilogy(bin_cent, ng, color=blue)
            bin_cent = (binse[1:] + binse[:-1]) / 2
            axs[1,1].semilogy(bin_cent, ne, color=red)
            
            axs[1, 1].set_xlabel("I (ADC levels)")
            axs[0, 0].set_xlabel("I (ADC levels)")

            plt.subplots_adjust(hspace=0.25, wspace=0.15)
            if qubit is not None: 
                fig.suptitle(f"Single Shot Histogram Analysis Q{qubit}")
        else:
            set_fig=False
        
        # Plot rotated data
        ax[0].plot(Ig_new, Qg_new, ".", label="g", color=blue, alpha=a, markersize=m)
        ax[0].plot(Ie_new, Qe_new, ".", label="e", color=red, alpha=a, markersize=m)
        if plot_f:
            ax[0].plot(If_new, Qf_new, ".", label="f", color="g", alpha=a, markersize=m)
        #ax[0].plot(xg_new, yg_new, color="k", marker="o")
        #ax[0].plot(xe_new, ye_new, color="k", marker="o")
        ax[0].text(0.95, 0.95, f'g: {xg_new:.2f}\ne: {xe_new:.2f}', 
                   transform=ax[0].transAxes, fontsize=10, 
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(facecolor='white', alpha=0.5))

        ax[0].set_xlabel('I (ADC levels)')
        lgnd=ax[0].legend(loc='lower right')
        # lgnd.legendHandles[0].set_markersize(6)
        # lgnd.legendHandles[1].set_markersize(6)
        ax[0].set_title("Angle: {:.2f}$^\circ$".format(theta * 180 / np.pi))
        ax[0].axis("equal")        

        # Plot histogram 
        ax[1].set_ylabel("Probability")
        ax[1].set_xlabel("I (ADC levels)")
        ax[1].legend(loc="upper right")
        ax[1].set_title(f"Histogram (Fidelity g-e: {100*fids[0]:.3}%)")
        ax[1].axvline(thresholds[0], color="0.2", linestyle="--")
        ax[1].plot(data["vhg"], data["histg"], '.-',color=blue, markersize=0.5, linewidth=0.3)
        ax[1].fill_between(data["vhg"], data["histg"], color=blue, alpha=0.3)
        ax[1].fill_between(data["vhe"], data["histe"], color=red, alpha=0.3)
        ax[1].plot(data["vhe"], data["histe"], '.-',color=red, markersize=0.5, linewidth=0.3)
        ax[1].plot(data["vhg"], gaussian(data["vhg"], 1, *data['paramsg']), 'k', linewidth=1)
        ax[1].plot(data["vhe"], excited_func(data["vhe"], data['vg'], data['ve'], data['sigma'], data['tm']), 'k', linewidth=1)
        if plot_f:
            nf, binsf, pf = ax[1].hist(
                If_new, bins=numbins, range=xlims, color="g", label="f", alpha=0.5
            )
            ax[1].axvline(thresholds[1], color="0.2", linestyle="--")
            ax[1].axvline(thresholds[2], color="0.2", linestyle="--")

                
        sigma = data['sigma']
        tm = data['tm']
        txt = f"Threshold: {thresholds[0]:.2f}"
        txt += f" \n Width: {sigma:.2f}"
        txt += f" \n $T_m/T_1$: {tm:.2f}"
        ax[1].text(0.025, 0.965, txt, 
               transform=ax[1].transAxes, fontsize=10, 
               verticalalignment='top', horizontalalignment='left', 
               bbox=dict(facecolor='none', edgecolor='black', alpha=0.5))

        if set_fig:
            fig.tight_layout()
        else:
            fig=None
        plt.show()
    else:
        fig = None
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    params = {'fids': fids, 'thresholds': thresholds, 'angle': theta * 180 / np.pi, 'ig':xg_new, 'ie':xe_new, 'err_e':err_e, 'err_g':err_g}
    return params, fig

def gaussian(x, mag, cen, wid):
    return mag / np.sqrt(2 * np.pi) / wid * np.exp(-((x - cen) ** 2) / 2 / wid**2)

def two_gaussians(x, mag1, cen1, wid, mag2, cen2):
    return 1 / np.sqrt(2 * np.pi) / wid * (mag1 *np.exp(-((x - cen1) ** 2) / 2 / wid**2) + mag2 * np.exp(-((x - cen2) ** 2) / 2 / wid**2))

def make_hist(d, nbins=200):
    hist, bin_edges = np.histogram(d, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

# Histogram and fit ground state data
def fit_gaussian(d, nbins=200, p0=None, plot=True):
    v, hist = make_hist(d, nbins)
    if p0 is None:
        p0 = [np.mean(v * hist) / np.mean(hist), np.std(d)]
    params, params_err = curve_fit(lambda x, a, b: gaussian(x, 1, a, b), v, hist, p0=p0)
    if plot:
        plt.plot(v, hist, "k.")
        plt.plot(v, gaussian(v, 1, *params), label="g")
    return params, v, hist

# Tail from T1 decay
# vg = ground state voltage, ve = excited state voltage, tm = measurement time/T1 time, sigma = SD of measurement noise
def distfn(v, vg, ve, sigma, tm):
    dv = ve - vg
    return np.abs(
        tm
        / 2
        / dv
        * np.exp(tm * (tm * sigma**2 / 2 / dv**2 - (v - vg) / dv))
        * (
            erf((tm * sigma**2 / dv + ve - v) / np.sqrt(2) / sigma)
            + erf((-tm * sigma**2 / dv + v - vg) / np.sqrt(2) / sigma)
        )
    )

# Fit function for excited state
def excited_func(x, vg, ve, sigma, tm):
    y = gaussian(x, 1, ve, sigma) * np.exp(-tm) + distfn(x, vg, ve, sigma, tm)
    return y

# Fit for sum of excited and ground states (adds in fitting of relative magnitudes)
def fit_all(x, mag_g, vg, ve, sigma, tm):
    yg = gaussian(x, mag_g, vg, sigma)
    ye = gaussian(x, 1 - mag_g, ve, sigma) * np.exp(-tm) + (1 - mag_g) * distfn(
        x, vg, ve, sigma, tm
    )
    return ye + yg

def rotate(x, y, theta):
    return x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)

def full_rotate(d, theta):
    
    d["Ig"], d["Qg"] = rotate(d["Ig"], d["Qg"], theta)
    d["Ie"], d["Qe"] = rotate(d["Ie"], d["Qe"], theta)
    return d

# Fit single shot data
def fit_single_shot(d, plot=True, rot=True):
    # rot: Rotate the data so that signal in I
    data_rot = {}
    d = deepcopy(d)
    if rot:
        params, _ = hist(d, plot=False, verbose=False)
        theta = np.pi * params['angle'] / 180
        data = full_rotate(d, theta)
    else:
        data = d
        theta = 0

    # Fit the 3 gaussian data sets
    paramsg, vhg, histg = fit_gaussian(data["Ig"], nbins=100, p0=None, plot=False)
    paramsqg, vhqg, histqg = fit_gaussian(data["Qg"], nbins=100, p0=None, plot=False)
    paramsqe, vhqe, histqe = fit_gaussian(data["Qe"], nbins=100, p0=None, plot=False)

    vqg = paramsqg[0]
    vqe = paramsqe[0]
    vg = paramsg[0]
    sigma = paramsg[1]

    # Fit the I excited state, including T1 decay.
    # Use previously fit value for vg and sigma
    vhe, histe = make_hist(data["Ie"], nbins=100)
    p0 = [np.mean(vhe * histe) / np.mean(histe), 0.2]
    paramse, params_err = curve_fit(
        lambda x, ve, tm: excited_func(x, vg, ve, sigma, tm), vhe, histe, p0=p0
    )
    ve = paramse[0]
    tm = paramse[1]
    paramse2 = [vg, ve, sigma, tm]
    # Theta from gaussian fit
    theta_corr = -np.arctan2((vqe - vqg), (ve - vg))

    #g_rot = rotate(data["Ig"], data["Qg"], theta_corr)
    #e_rot = rotate(data["Ie"], data["Qe"], theta_corr)

    g_rot = rotate(data['Ig'], data['Qg'], 0)
    e_rot = rotate(data['Ie'], data['Qe'], 0)

    #pg = ['mag', 'cen', 'wid', 'mag', 'cen']
    pg = [0.99, vg, sigma, 0.01, ve]
    #print(pg)
    try: 
        paramsg2, params_err = curve_fit(two_gaussians, vhg, histg, p0=pg)
    except:
        paramsg2 = np.nan
    #print(paramsg2)
    #print(paramsg2-pg)


    data["Ie_rot"] = e_rot[0]
    data["Qe_rot"] = e_rot[1]
    data["Ig_rot"] = g_rot[0]
    data["Qg_rot"] = g_rot[1]

    data["vhg"] = vhg
    data["histg"] = histg
    data["vhe"] = vhe
    data["histe"] = histe

    data["vqg"] = vhqg
    data["histqg"] = histqg
    data["vqe"] = vhqe
    data["histqe"] = histqe

    if plot:
        plt.plot(vhg, histg, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhe, histe, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhg, gaussian(vhg, 1, *paramsg), label="g", linewidth=1)
        plt.plot(vhe, excited_func(vhe, vg, ve, sigma, tm), label="e", linewidth=1)
        plt.ylabel("Probability")
        plt.title("Single Shot Histograms")
        plt.xlabel("Voltage")
    p = {'theta': theta, 'vg': vg, 've': ve, 'sigma': sigma, 'tm': tm, 'vqg': vqg, 'vqe': vqe, 'theta_corr': theta_corr}
    return data, p, paramsg, paramse2

# ====================================================== #

class HistogramProgram(QickProgram):

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.add_loop("shotloop", cfg.expt.shots)  # number of total shots

        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        if cfg.expt.active_reset:
            self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]]
        else:
            self.phase = 0
        self.readout_length = cfg.expt.readout_length
        super()._initialize(cfg, readout="")

        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
        self.delay_auto(t=0.01, tag="wait")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], pins=[0],t=self.trig_offset)

        if cfg.expt.active_reset:
            self.reset(7)


    def reset(self, i):
        super().reset(i)
        
    
    def collect_shots(self, offset=0):

        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            #nsamp = rocfg["length"]
            iq_raw = self.get_raw()
            i_shots = iq_raw[i][:, :, 0, 0]# / nsamp - offset
            i_shots = i_shots.flatten()
            q_shots = iq_raw[i][:, :, 0, 1] #/ nsamp - offset
            q_shots = q_shots.flatten()
        return i_shots, q_shots


class HistogramExperiment(QickExperiment):
    """
    Histogram Experiment
    expt = dict(
        shots: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(
        self,
        cfg_dict,
        prefix=None,
        progress=True,
        qi=0,
        go=True,
        check_f=False,
        params={},
        style="",
        display=True,
    ):

        if prefix is None:
            prefix = f"single_shot_qubit_{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = dict(
            shots=10000,
            reps=1,
            soft_avgs=1,
            readout_length=self.cfg.device.readout.readout_length[qi],
            frequency=self.cfg.device.readout.frequency[qi],
            gain=self.cfg.device.readout.gain[qi],
            active_reset = False,
            check_e=True,
            check_f=check_f,
            qubit=[qi],
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],
        )
        
        self.cfg.expt = {**params_def, **params}
        if self.cfg.expt.active_reset:
            super().configure_reset()
        
        if go:
            self.go(analyze=True, display=display, progress=progress, save=True)

    def acquire(self, progress=False, debug=False):

        data = dict()
        if 'setup_reset' in self.cfg.expt and self.cfg.expt.setup_reset:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        elif self.cfg.expt.active_reset:
            final_delay = self.cfg.expt.readout_length
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Ground state shots
        cfg2 = copy.deepcopy(dict(self.cfg))
        cfg = AttrDict(cfg2)
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False

        histpro = HistogramProgram(soccfg=self.soccfg, final_delay=final_delay, cfg=cfg)
        iq_list = histpro.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
        )
        data["Ig"] = iq_list[0][0][:, 0]
        data["Qg"] = iq_list[0][0][:, 1]
        if self.cfg.expt.active_reset:
            data["Igr"]=iq_list[0][1:,:, 0]

        irawg, qrawg = histpro.collect_shots()
        
        rawd = [irawg[-1], qrawg[-1]]
        #print("buffered readout:", rawd)

        # Excited state shots
        if self.cfg.expt.check_e:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = False
            histpro = HistogramProgram(
                soccfg=self.soccfg, final_delay=final_delay, cfg=cfg
            )
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
            )

            data["Ie"] = iq_list[0][0][:, 0]
            data["Qe"] = iq_list[0][0][:, 1]
            irawe, qraw = histpro.collect_shots()
            #rawd = [iraw[-1], qraw[-1]]
            #print("buffered readout:", rawd)
            #print("feedback readout:", self.soc.read_mem(2,'dmem'))
            if self.cfg.expt.active_reset:
                data["Ier"]=iq_list[0][1:,:, 0]
            #print(f"{np.mean(irawg)} mean raw g, {np.mean(irawe)} mean raw e")

        # Excited state shots
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = True
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
            )
            data["If"], data["Qf"] = histpro.collect_shots()

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=False, **kwargs):
        if data is None:
            data = self.data

        params, _ = hist(
            data=data, plot=False, span=span, verbose=verbose
        )
        data.update(params)
        try:
            data2, p, paramsg, paramse2 = fit_single_shot(data, plot=False)
            data.update(p)
            data["vhg"]=data2["vhg"]
            data["histg"]=data2["histg"]
            data["vhe"]=data2["vhe"]
            data["histe"]=data2["histe"]
            data["paramsg"] = paramsg
            data["shots"] = self.cfg.expt.shots
        except:
            print('Fits failed')
             
        return data

    def display(
        self,
        data=None,
        span=None,
        verbose=False,
        plot_e=True,
        plot_f=False,
        ax=None,
        plot=True,
        **kwargs,
    ):
        if data is None:
            data = self.data

        if ax is not None:
            savefig = False
        else:
            savefig = True

        params, fig = hist(
            data=data, plot=plot, verbose=verbose, span=span, ax=ax, qubit=self.cfg.expt.qubit[0]
        )
        fids = params["fids"]
        thresholds = params["thresholds"]
        angle = params["angle"]
        print(f"ge Fidelity (%): {100*fids[0]:.3f}")
        if "expt" not in self.cfg:
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f
        if self.cfg.expt.check_f:
            print(f"gf Fidelity (%): {100*fids[1]:.3f}")
            print(f"ef Fidelity (%): {100*fids[2]:.3f}")
        print(f"Rotation angle (deg): {angle:.3f}")
        print(f"Threshold ge: {thresholds[0]:.3f}")
        if self.cfg.expt.check_f:
            print(f"Threshold gf: {thresholds[1]:.3f}")
            print(f"Threshold ef: {thresholds[2]:.3f}")
        imname = self.fname.split("\\")[-1]

        if savefig:
            plt.show()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )

    def save_data(self, data=None):
        super().save_data(data=data)

    def check_reset(self): 
        nbins=75
        fig, ax = plt.subplots(2,1, figsize=(6,7))
        fig.suptitle(f"Q{self.cfg.expt.qubit[0]}")
        vg, histg = make_hist(self.data['Ig'], nbins=nbins)
        ax[0].semilogy(vg, histg, color=blue, linewidth=2)
        ax[1].semilogy(vg, histg, color=blue, linewidth=2)
        b  = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(self.data['Igr']))
        ve, histe = make_hist(self.data['Ie'], nbins=nbins)
        ax[1].semilogy(ve, histe, color=red, linewidth=2)
        for i in range(len(self.data['Igr'])):
            v, hist = make_hist(self.data['Igr'][i], nbins=nbins)
            ax[0].semilogy(v, hist, color=b[i], linewidth=1, label=f'{i+1}')
            v, hist = make_hist(self.data['Ier'][i], nbins=nbins)
            ax[1].semilogy(v, hist, color=b[i], linewidth=1, label=f'{i+1}')

        def find_bin_closest_to_value(bins, value):
            return np.argmin(np.abs(bins - value))

        ind= find_bin_closest_to_value(v, self.data['ie'])
        ind_e= find_bin_closest_to_value(ve, self.data['ie'])
        ind_g= find_bin_closest_to_value(vg, self.data['ie'])

        reset_level = hist[ind]
        e_level = histe[ind_e]
        g_level = histg[ind_g]

        print(f"Reset is {reset_level/e_level:3g} of e and {reset_level/g_level:3g} of g")

        self.data['reset_e'] = reset_level/e_level
        self.data['reset_g'] = reset_level/g_level



        

        
        ax[0].legend()

        ax[0].set_title('Ground state')
        ax[1].set_title('Excited state')
        plt.show()


# ====================================================== #
class SingleShotOptExperiment(QickExperiment):
    """
    start_f (float): Starting frequency for the experiment.
    span_f (float): Frequency span for the experiment.
    expts_f (int): Number of frequency experiments.

    start_gain (float): Starting gain for the experiment.
    span_gain (float): Gain span for the experiment.
    expts_gain (int): Number of gain experiments.

    start_len (float): Starting readout length for the experiment.
    span_len (float): Readout length span for the experiment.
    expts_len (int): Number of readout length experiments.
    reps (int): Number of repetitions for each experiment.
    check_f (bool): Flag to check frequency.

    step_f (float): Frequency step size.
    step_gain (float): Gain step size.
    step_len (float): Readout length step size.

    qubit (list): List of qubits to be used in the experiment.
    save_data (bool): Flag to save data.
    qubit_chan (int): Qubit channel for readout.
    """

    def __init__(
        self, cfg_dict, prefix=None, progress=None, qi=0, go=True, params={}, style=""
    ):

        if prefix is None:
            prefix = f"single_shot_opt_qubit_{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)
        self.im = cfg_dict["im"]
        self.soccfg = cfg_dict["soc"]
        self.config_file = cfg_dict["cfg_file"]
        self.cfg_dict = cfg_dict

        params_def = {
            "span_f": 0.3,
            "expts_f": 5,
            "expts_gain": 5,
            "expts_len": 5,
            "shots": 10000,
            "check_f": False,
            "qubit": [qi],
            "save_data": True,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params = {**params_def, **params}

        # Start vals
        if params["expts_f"] == 1:
            params_def["start_f"] = self.cfg.device.readout.frequency[qi]
        else:
            params_def["start_f"] = (
                self.cfg.device.readout.frequency[qi] - 0.5 * params["span_f"]
            )

        if params["expts_gain"] == 1:
            params_def["start_gain"] = self.cfg.device.readout.gain[qi]
        else:
            if style == "fine":
                params_def["start_gain"] = self.cfg.device.readout.gain[qi] * 0.8
                params_def["span_gain"] = 0.4 * self.cfg.device.readout.gain[qi]
            else:
                params_def["start_gain"] = self.cfg.device.readout.gain[qi] * 0.3
                params_def["span_gain"] = 1.8 * self.cfg.device.readout.gain[qi]

        if params["expts_len"] == 1:
            params_def["start_len"] = self.cfg.device.readout.readout_length[qi]
        else:
            if style == "fine":
                params_def["start_len"] = (
                    self.cfg.device.readout.readout_length[qi] * 0.8
                )
                params_def["span_len"] = (
                    0.4 * self.cfg.device.readout.readout_length[qi]
                )
            else:
                params_def["start_len"] = (
                    self.cfg.device.readout.readout_length[qi] * 0.3
                )
                params_def["span_len"] = (
                    1.8 * self.cfg.device.readout.readout_length[qi]
                )

        params = {**params_def, **params}
        if params["expts_f"] == 1:
            params_def["step_f"] = 0
        else:
            params_def["step_f"] = params_def["span_f"] / (params["expts_f"] - 1)

        if params["expts_gain"] == 1:
            params_def["step_gain"] = 0
            params_def["span_gain"] = 0
        else:
            params_def["step_gain"] = params_def["span_gain"] / (params["expts_gain"] - 1)

        if params["expts_len"] == 1:
            params_def["step_len"] = 0
        else:
            params_def["step_len"] = params_def["span_len"] / (params["expts_len"] - 1)

        if params_def["span_gain"] + params_def["start_gain"] > self.cfg.device.qubit.max_gain:
            params_def["span_gain"] = self.cfg.device.qubit.max_gain - params_def["start_gain"]
        self.cfg.expt = {**params_def, **params}

        # Check for unexpected parameters
        super().check_params(params)

        if go:
            self.go(analyze=False, display=False, progress=False, save=True)
            self.analyze()
            self.display()

    def acquire(self, progress=True):
        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"] * np.arange(
            self.cfg.expt["expts_f"]
        )

        max_gain = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"] * (
            self.cfg.expt["expts_gain"] - 1
        )
        if max_gain > self.cfg.device.qubit.max_gain:
            self.cfg.expt["step_gain"] = (self.cfg.device.qubit.max_gain - self.cfg.expt["start_gain"])/(self.cfg.expt["expts_gain"]-1)
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"] * np.arange(
            self.cfg.expt["expts_gain"]
        )
        
        lenpts = self.cfg.expt["start_len"] + self.cfg.expt["step_len"] * np.arange(
            self.cfg.expt["expts_len"]
        )

        if "save_data" not in self.cfg.expt:
            self.cfg.expt.save_data = False

        fid = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        threshold = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        angle = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        tm = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        sigma = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        if "check_f" not in self.cfg.expt:
            check_f = False
        else:
            check_f = self.cfg.expt.check_f
        Ig, Ie, Qg, Qe = [], [], [], []
        if check_f:
            If, Qf = [], []
        gprog = False
        fprog = False
        lprog = False
        if len(fpts) > 1:
            fprog = True
        else:
            fprog = False
            if len(gainpts) > 1:
                gprog = True
            else:
                gprog = False
                if len(lenpts) > 1:
                    lprog = True
                else:
                    lprog = False

        for f_ind, f in enumerate(tqdm(fpts, disable=not fprog)):
            Ig.append([])
            Ie.append([])
            Qg.append([])
            Qe.append([])
            if check_f:
                If.append([])
                Qf.append([])
            for g_ind, gain in enumerate(tqdm(gainpts, disable=not gprog)):
                Ig[-1].append([])
                Ie[-1].append([])
                Qg[-1].append([])
                Qe[-1].append([])
                if check_f:
                    If[-1].append([])
                    Qf[-1].append([])
                for l_ind, l in enumerate(tqdm(lenpts, disable=not lprog)):
                    shot = HistogramExperiment(
                        self.cfg_dict,
                        go=False,
                        progress=False,
                        qi=self.cfg.expt.qubit[0],
                        params=dict(
                            frequency=f,
                            gain=gain,
                            readout_length=l,
                            reps=1,
                            check_e=True,
                            check_f=check_f,
                            shots=self.cfg.expt.shots,
                            save_data=self.cfg.expt.save_data,
                            qubit_chan=self.cfg.expt.qubit_chan,
                        ),
                    )
                    #shot.cfg = self.cfg
                    
                    shot.go(analyze=False, display=False, progress=progress, save=False)
                    Ig[-1][-1].append(shot.data["Ig"])
                    Ie[-1][-1].append(shot.data["Ie"])
                    Qg[-1][-1].append(shot.data["Qg"])
                    Qe[-1][-1].append(shot.data["Qe"])
                    if check_f:
                        If[-1][-1].append(shot.data["If"])
                        Qf[-1][-1].append(shot.data["Qf"])
                    results = shot.analyze(verbose=False)
                    fid[f_ind, g_ind, l_ind] = (
                        results["fids"][0] if not check_f else results["fids"][1]
                    )
                    threshold[f_ind, g_ind, l_ind] = (
                        results["thresholds"][0]
                        if not check_f
                        else results["thresholds"][1]
                    )
                    try:
                        tm[f_ind, g_ind, l_ind] = results["tm"]
                        sigma[f_ind, g_ind, l_ind] = results["sigma"]
                    except: 
                        pass
                    angle[f_ind, g_ind, l_ind] = results["angle"]
                    # print(f'freq: {f}, gain: {gain}, len: {l}')
                    # print(f'\tfid ge [%]: {100*results["fids"][0]}')
                    if check_f:
                        print(f'\tfid gf [%]: {100*results["fids"][1]:.3f}')

        if check_f:
            self.data["If"] = np.array(If)
            self.data["Qf"] = np.array(Qf)
        if self.cfg.expt.save_data:
            self.data = dict(
                fpts=fpts,
                gainpts=gainpts,
                lenpts=lenpts,
                fid=fid,
                threshold=threshold,
                angle=angle,
                Ig=Ig,
                Ie=Ie,
                Qg=Qg,
                Qe=Qe,tm=tm, sigma=sigma,
            )
            if check_f:
                self.data["If"] = If
                self.data["Qf"] = Qf
        else:
            self.data = dict(
                fpts=fpts,
                gainpts=gainpts,
                lenpts=lenpts,
                fid=fid,
                threshold=threshold,
                angle=angle,tm=tm, sigma=sigma,
            )

        for key in self.data.keys():
            self.data[key] = np.array(self.data[key])
        return self.data

    def analyze(self, data=None, **kwargs):
        if data == None:
            data = self.data
        fid = data["fid"]
        threshold = data["threshold"]
        angle = data["angle"]
        fpts = data["fpts"]
        gainpts = data["gainpts"]
        lenpts = data["lenpts"]

        imax = np.unravel_index(np.argmax(fid), shape=fid.shape)

        print(f"Max fidelity {100*fid[imax]:.3f} %")
        print(
            f"Set params: \n angle (deg) {-angle[imax]:.3f} \n threshold {threshold[imax]:.3f} \n freq [MHz] {fpts[imax[0]]:.3f} \n Gain [DAC units] {gainpts[imax[1]]:.3f} \n readout length [us] {lenpts[imax[2]]:.3f}"
        )
        self.data["freq"] = fpts[imax[0]]
        self.data["gain"] = gainpts[imax[1]]
        self.data["length"] = lenpts[imax[2]]

        return imax

    def display(self, data=None,plot_pars=False, **kwargs):
        if data is None:
            data = self.data

        fid = data["fid"]

        fpts = data["fpts"]  # outer sweep, index 0
        gainpts = data["gainpts"]  # middle sweep, index 1
        lenpts = data["lenpts"]  # inner sweep, index 2
        ndims = 0
        npts = []
        inds = []
        sweep_var = []
        labs = ["Freq. (MHz)", "Gain", "Readout Length ($\mu$s)"]
        if len(fpts) > 1:
            ndims += 1
            sweep_var.append("fpts")
            npts.append(len(fpts))
            inds.append(0)
        if len(gainpts) > 1:
            ndims += 1
            sweep_var.append("gainpts")
            npts.append(len(gainpts))
            inds.append(1)
        if len(lenpts) > 1:
            ndims += 1
            sweep_var.append("lenpts")
            npts.append(len(lenpts))
            inds.append(2)

        def smart_ax(n):
            row = int(np.ceil(n / 5))
            if n < 5:
                col = n
            else:
                col = 5
            return row, col

        title = f"Single Shot Optimization Q{self.cfg.expt.qubit[0]}"

        def return_dim(data, dim, i):
            if len(dim) == 1:
                if dim[0] == 0:
                    return data[i, :, :].reshape(-1)
                elif dim[0] == 1:
                    return data[:, i, :].reshape(-1)
                elif dim[0] == 2:
                    return data[:, :, i].reshape(-1)
            elif len(dim) == 2:
                if dim == [0, 1]:
                    return data[i[0], i[1], :].reshape(-1)
                if dim == [0, 2]:
                    return data[i[0], :, i[1]].reshape(-1)
                if dim == [1, 2]:
                    return data[:, i[0], i[1]].reshape(-1)

        m = 0.5
        if ndims == 1:
            row, col = smart_ax(npts[0])
            fig, ax = plt.subplots(row, col, figsize=(col * 3, row * 3))
            ax = ax.flatten()
            for i in range(npts[0]):

                ax[i].plot(
                    return_dim(self.data["Ig"], inds, i),
                    return_dim(self.data["Qg"], inds, i),
                    ".",
                    color=blue,
                    alpha=0.2,
                    markersize=m,
                )
                ax[i].plot(
                    return_dim(self.data["Ie"], inds, i),
                    return_dim(self.data["Qe"], inds, i),
                    ".",
                    color=red,
                    alpha=0.2,
                    markersize=m,
                )

                ax[i].set_title(f"{labs[inds[0]]} {data[sweep_var[0]][i]:.2f}")

        elif ndims == 2:
            fig, ax = plt.subplots(npts[0], npts[1], figsize=(npts[1] * 3, npts[0] * 3))

            for i in range(npts[0]):
                for j in range(npts[1]):
                    ax[i, j].plot(
                        return_dim(self.data["Ig"], inds, [i, j]),
                        return_dim(self.data["Qg"], inds, [i, j]),
                        ".",
                        color=blue,
                        alpha=0.2,
                        markersize=m,
                    )
                    ax[i, j].plot(
                        return_dim(self.data["Ie"], inds, [i, j]),
                        return_dim(self.data["Qe"], inds, [i, j]),
                        ".",
                        color=red,
                        alpha=0.2,
                        markersize=m,
                    )

                    if i == npts[0] - 1:
                        ax[i, j].set_xlabel(np.round(self.data[sweep_var[1]][j], 2))
                    if j == 0:
                        ax[i, j].set_ylabel(np.round(self.data[sweep_var[0]][i], 2))
            plt.figtext(0.5, 0.0, labs[inds[1]], horizontalalignment="center")
            plt.figtext(
                0.0, 0.5, labs[inds[0]], verticalalignment="center", rotation="vertical"
            )
        else:
            for k in range(npts[2]):
                fig, ax = plt.subplots(
                    npts[0], npts[1], figsize=(npts[1] * 3, npts[0] * 3)
                )
                for i in range(npts[0]):
                    for j in range(npts[1]):
                        ax[i, j].plot(
                            self.data["Ig"][i, j, k, :],
                            self.data["Qg"][i, j, k],
                            ".",
                            color=blue,
                            alpha=0.2,
                            markersize=m,
                        )
                        ax[i, j].plot(
                            self.data["Ie"][i, j, k, :],
                            self.data["Qe"][i, j, k],
                            ".",
                            color=red,
                            alpha=0.2,
                            markersize=m,
                        )
            fig.suptitle(title)
            fig.tight_layout()
            imname = self.fname.split("\\")[-1]
            fig.savefig(
                self.fname[0 : -len(imname)]
                + "images\\"
                + imname[0:-3]
                + "_raw_{k}.png"
            )

        fig.suptitle(title)
        fig.tight_layout()
        imname = self.fname.split("\\")[-1]
        fig.savefig(
            self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "_raw_{k}.png"
        )

        title = f"Single Shot Optimization Q{self.cfg.expt.qubit[0]}"
        fig = plt.figure(figsize=(9, 5.5))
        plt.title(title)
        if len(fpts) > 1:
            xval = fpts
            xlabel = "Frequency (MHz)"
            var1 = gainpts
            var2 = lenpts
            npts = len(var1) * len(var2)
            bb = sns.color_palette("coolwarm", npts)
            leg_title = "Gain, Len ($\mu$s)"
            for v1_ind, v1 in enumerate(var1):
                for v2_ind, v2 in enumerate(var2):
                    plt.plot(
                        xval,
                        100 * fid[:, v1_ind, v2_ind],
                        "o-",
                        label=f"{v1:.2f}, {v2:.2f}",
                        color=bb[v1_ind * len(var2) + v2_ind],
                    )
        elif len(gainpts) > 1:
            xval = gainpts
            xlabel = "Gain/Max Gain"
            var1 = fpts
            var2 = lenpts
            npts = len(var1) * len(var2)
            bb = sns.color_palette("coolwarm", npts)
            leg_title = "Freq (MHz), Len ($\mu$s)"
            for v1_ind, v1 in enumerate(var1):
                for v2_ind, v2 in enumerate(var2):
                    plt.plot(
                        xval,
                        100 * fid[v1_ind, :, v2_ind],
                        "o-",
                        label=f"{v1:.2f}, {v2:.2f}",
                        color=bb[v1_ind * len(var2) + v2_ind],
                    )
        else:
            xval = lenpts
            xlabel = "Readout length ($\mu$s)"
            var1 = fpts
            var2 = gainpts
            npts = len(var1) * len(var2)
            bb = sns.color_palette("coolwarm", npts)
            leg_title = "Freq (MHz), Gain"
            for v1_ind, v1 in enumerate(var1):
                for v2_ind, v2 in enumerate(var2):
                    plt.plot(
                        xval,
                        100 * fid[v1_ind, v2_ind, :],
                        "o-",
                        label=f"{v2:1.0f},  {v1:.2f}",
                        color=bb[v1_ind * len(var2) + v2_ind],
                    )

        plt.xlabel(xlabel)
        plt.ylabel(f"Fidelity [%]")
        plt.legend(title=leg_title)
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png")
        plt.show()

        if plot_pars:

            tmv = self.data['tm'][0]
            tmv[tmv<0.001]=np.nan
            sns.set_palette('coolwarm', len(tmv))
            fig, ax = plt.subplots(2,1, figsize=(8,6))
            #tm = np.transpose(tmv)
            for i, tm_arr in enumerate(tmv):
                gain=self.data['gainpts'][i]
                ax[0].plot(self.data['lenpts'],self.data['lenpts']/tm_arr,'o-', label=f"{gain:.2f}")
            ax[0].set_xlabel('Readout Length')
            ax[0].axhline(y=self.cfg.device.qubit.T1[self.cfg.expt.qubit[0]], color='k', linestyle='--', label='T1')
            ax[0].set_ylabel('$T_m/(T_m/T_1)$')
            ax[0].legend()
            sigma = self.data['sigma'][0]

            for i, s in enumerate(sigma):
                gain=self.data['gainpts'][i]
                ax[1].loglog(self.data['lenpts'],s,'o-', label=f"{gain:.2f}")
            ax[1].legend()
            ax[1].set_xlabel('Readout Length')
            ax[1].set_ylabel('$\sigma$')
            fig.tight_layout()

        self.do_more= self.check_edges()
        if self.data['gain']==1: # change to max_gain 
            self.do_more=False
        
    
    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname
    
    def check_edges(self):
        do_more=False
        fid = self.data["fid"]
        fid_expts = fid.shape
        if all(dim % 2 != 0 for dim in fid_expts):
            old_fid = fid[(fid_expts[0] // 2), (fid_expts[1] // 2), (fid_expts[2] // 2)]
            max_fid = np.max(fid)
            if (max_fid - old_fid)/old_fid > 0.1:
                print("Fidelity is not maximized at the center of the sweep.")
                max_indices = np.unravel_index(np.argmax(fid), fid.shape)
                print(f"Max fidelity found at indices: {max_indices}")
                if max_indices[1]==0 or max_indices[1]==fid_expts[1]-1 or max_indices[2]==0 or max_indices[2]==fid_expts[2]-1:
                    do_more=True
        else:
            print("Not all elements in fid_expts are odd.")
        return do_more

        
