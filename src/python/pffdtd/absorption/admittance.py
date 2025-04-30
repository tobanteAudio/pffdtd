# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

"""Miscellaneous functions for dealing with wall admittances (materials)
DEF is normalised RLC triplet (i.e., for specific admittance)
"""
from pathlib import Path

import click
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.optimize as scpo

from pffdtd.filters.octave import center_frequencies


def convert_nabs_to_R(nabs):
    """normal-incidence absorption to reflection coefficient
    """
    nabs = np.float64(nabs)
    assert np.all(nabs >= 0)
    assert np.all(nabs <= 1)
    return np.sqrt(1.0-nabs)


def convert_Yn_to_R(Yn):
    """specific admittance to reflection coefficient
    """
    assert np.all(Yn > 0.0)
    R = (1.0-Yn)/(1.0+Yn)
    return R


def convert_R_to_Yn(R):
    """reflection coefficient to specific admittance
    """
    assert np.all(R < 1.0)
    Yn = (1.0-R)/(1.0+R)
    return Yn


def convert_R_to_Zn(R):
    """reflection coefficient to specific impedance
    """
    assert np.all(R < 1.0)
    Zn = 1.0/convert_R_to_Yn(R)
    return Zn


def convert_Sabs_to_Yn(Sabs, max_iter=100):
    """sabine absorption to specific admittance, Paris formula inversion with Newton method
    """
    def _print(fstring):
        print(f'--MATERIALS: {fstring}')

    def fg(g):
        return 8.0*g*(1+g/(1+g)-2*g*np.log((g+1)/g))

    def fgd(g):
        return -8.0*(-4*g**2 - 6*g + 4 * (1+g)**2*g*np.log((g+1)/g)-1)/(1+g)**2

    if Sabs > 0.9512:
        _print('warning, Sabs>0.9512 -- not possible for locally-reactive model')
        Sabs = 0.9512

    if Sabs == 0:
        g = 0
    else:  # newton solve
        x_old = Sabs/8.0  # starting guess
        x_new = 0
        err = np.inf
        niter = 0
        while (niter < max_iter) and (err > 1e-6):
            x_new = x_old - (fg(x_old)-Sabs)/fgd(x_old)
            niter += 1
            err = np.abs(1-x_new/x_old)
            # _print(f'Iteration {niter}: {x_new=}, {err=}')
            x_old = x_new
        g = x_new
    return g


def write_freq_ind_mat_from_Zn(Zn, filename):
    """write HDF5 mat file from a specific impedance (frequency-independent)
    """
    def _print(fstring):
        print(f'--MATERIALS: {fstring}')

    assert ~np.isnan(Zn)
    assert ~np.isinf(Zn)  # rigid should be specified in scene (no material)
    assert Zn >= 0
    filename = Path(filename)
    h5f = h5py.File(filename, 'w')
    DEF = np.array([0, Zn, 0])
    assert np.all(np.sum(DEF > 0, axis=-1))  # at least one non-zero
    _print(f'{DEF=}')
    h5f.create_dataset('DEF', data=np.atleast_2d(DEF))
    h5f.close()


def write_freq_ind_mat_from_Yn(Yn, filename):
    """write HDF5 mat file from a specific impedance (frequency-independent)
    """
    def _print(fstring):
        print(f'--MATERIALS: {fstring}')

    assert ~np.isnan(Yn)
    assert ~np.isinf(Yn)
    assert Yn > 0  # rigid should be specified in scene (no material)
    write_freq_ind_mat_from_Zn(1/Yn, filename)


def write_freq_dep_mat(DEF, filename):
    """write HDF5 mat file from frequency-independent triplet (D=F=0)
    """
    def _print(fstring):
        print(f'--MATERIALS: {fstring}')

    _print(f'{DEF=}')
    DEF = np.atleast_2d(DEF)
    # rigid should be specified in scene (no material)
    assert ~np.any(np.isinf(DEF))
    assert ~np.any(np.isnan(DEF))
    assert np.all(DEF >= 0)  # all non-neg
    assert np.all(np.sum(DEF > 0, axis=-1))  # at least one non-zero
    assert DEF.shape[1] == 3
    filename = Path(filename)
    h5f = h5py.File(filename, 'w')
    _print(f'{DEF=}')
    h5f.create_dataset('DEF', data=DEF)
    h5f.close()


def read_mat_DEF(filename) -> np.ndarray:
    """write HDF5 mat file from frequency-independent triplet (D=F=0)
    """
    h5f = h5py.File(Path(filename), 'r')
    DEF = h5f['DEF'][()]
    h5f.close()
    return DEF


def plot_DEF_admittance(fv: np.ndarray, DEF: np.ndarray, model_Rf=None):
    """plot some admittance based on DEF triplets (specific admittance)
    """
    DEF = np.atleast_2d(DEF)
    assert DEF.shape[1] == 3
    D, E, F = DEF.T
    jw = 1j*fv*2*np.pi

    Rf, Yn, Zn_br, Rf_br = compute_Rf_from_DEF(jw, D, E, F)

    if model_Rf is not None:
        model_Yn = convert_R_to_Yn(model_Rf)

    fig = plt.figure()
    # reflection coefficient
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(fv, np.abs(Rf), linestyle='-')
    ax.plot(fv, np.abs(Rf_br), linestyle=':')
    if model_Rf is not None:
        ax.plot(fv, np.abs(model_Rf), linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$|R|$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(fv, np.angle(Rf), linestyle='-')
    ax.plot(fv, np.angle(Rf_br), linestyle=':')
    if model_Rf is not None:
        ax.plot(fv, np.angle(model_Rf), linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$\angle R$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    fig = plt.figure()
    # specific (normalised) admittance (abs/ang)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(fv, np.abs(Yn), linestyle='-')
    ax.plot(fv, np.abs(1.0/Zn_br), linestyle=':')
    if model_Rf is not None:
        ax.plot(fv, np.abs(model_Yn), linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$|Y|$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(fv, np.angle(Yn), linestyle='-')
    ax.plot(fv, np.angle(1.0/Zn_br), linestyle=':')
    if model_Rf is not None:
        ax.plot(fv, np.angle(model_Yn), linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$\angle Y$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    fig = plt.figure()
    # specific (normalised) admittance (real/imag)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(fv, np.real(Yn), linestyle='-')
    ax.plot(fv, np.real(1.0/Zn_br), linestyle=':')
    if model_Rf is not None:
        ax.plot(fv, np.real(model_Yn), linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$\Re(Y)$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(fv, np.imag(Yn), linestyle='-')
    ax.plot(fv, np.imag(1.0/Zn_br), linestyle=':')
    if model_Rf is not None:
        ax.plot(fv, np.imag(model_Yn), linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$\Im(Y)$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    plt.show()


def compute_Rf_from_DEF(jw, D, E, F):
    """plot compute reflection coefficient from DEF triplets (specific admittance)
    """
    Zn_br = jw[:, None]*D[None, :] + E + F[None, :]/jw[:, None]
    Yn = np.sum(1.0/Zn_br, axis=-1)
    Rf = (1.0-Yn)/(1.0+Yn)
    Rf_br = (Zn_br-1.0)/(Zn_br+1.0)
    return Rf, Yn, Zn_br, Rf_br


def _to_DEF(Ynm, dw, w0):
    # Ynm is max abs normalised admittance
    # dw is half-power bandwidth in radians/s
    # w0 is resonant frequency in radians/s
    D = 1.0/Ynm/dw
    E = 1.0/Ynm
    F = w0**2/Ynm/dw
    return D, E, F


def _from_DEF(D, E, F):
    Ynm = 1.0/E
    dw = E/D
    w0 = np.sqrt(F/D)
    return Ynm, dw, w0


def fit_to_Sabs_oct_11(Sabs, filename, plot=False, verbose=False):
    # fit Yn from 11 Sabine octave-band coefficients (16Hz to 16KHz)
    # this is a simple fitting routine that is a starting point for octave-band input data
    assert Sabs.size == 11
    Noct = Sabs.size
    # frequency vector to fit over
    fv = np.logspace(np.log10(10), np.log10(20e3), 1000)
    jw = 1j*fv*2*np.pi
    fcv = center_frequencies(1, 1000, 6, 5)
    ymv = np.zeros(Noct)
    dwv = np.zeros(Noct)
    w0v = np.zeros(Noct)
    Y_target = np.zeros(fv.shape)
    for j in range(Noct):
        fc = fcv[j]
        sa = Sabs[j]
        Ynm = convert_Sabs_to_Yn(sa)
        if j == 0:
            i1 = 0
        else:
            i1 = np.flatnonzero(fv >= fc/np.sqrt(2))[0]
        if j == Noct-1:
            i2 = fv.size
        else:
            i2 = np.flatnonzero(fv >= fc*np.sqrt(2))[0]

        Y_target[i1:i2] = Ynm
        w0 = 2*np.pi*fc
        dw = w0/np.sqrt(2)  # using resonance with half-octave bandwidth
        ymv[j] = Ynm
        dwv[j] = dw
        w0v[j] = w0

    R_target = (1.0-Y_target)/(1.0+Y_target)

    def cost_function_3(x0):
        if np.any(x0 < 0):
            # something big to disallow neg values
            return np.finfo(np.float64).max

        x0 = x0.reshape((-1, 3))
        ym = x0[:, 0]
        dw = x0[:, 1]
        w0 = x0[:, 2]

        D, E, F = _to_DEF(ym, dw, w0)

        Rf_opt, Yn_opt, _, _ = compute_Rf_from_DEF(jw, D, E, F)
        assert Yn_opt.ndim == 1
        assert Rf_opt.ndim == 1
        abs_opt = 1-np.abs(Rf_opt)**2
        abs_target = 1-np.abs(R_target)**2
        # simply fit to absorption coefficients
        cost = np.sum(np.abs(abs_opt-abs_target))
        # other possible cost functions
        # cost = np.sum(np.abs(np.abs(Y_target) - np.abs(Yn_opt)))
        # cost = np.sum(np.abs(1-np.abs(Y_target)/np.abs(Yn_opt)))
        # cost = np.sum(np.abs(Y_target - Yn_opt))
        # cost = np.sum(np.abs(R_target - Rf_opt))
        return cost

    # limit optimisation to just Yabs (keep bandwidths fix and w0 fixed)
    def cost_function(x0):
        return cost_function_3(np.c_[x0, dwv, w0v].flat[:])

    # run optimisation
    x0 = ymv
    initial_cost = cost_function(x0)
    # 'Powell' works well too
    res = scpo.minimize(cost_function, x0, method='Nelder-Mead')
    final_cost = cost_function(res.x)
    assert final_cost <= initial_cost
    ymv_opt = res.x
    dwv_opt = dwv
    w0v_opt = w0v

    D, E, F = _to_DEF(ymv_opt, dwv_opt, w0v_opt)
    DEF = np.c_[D, E, F]

    # now save
    h5f = h5py.File(filename, 'w')
    assert np.all(np.sum(DEF > 0, axis=-1))  # at least one non-zero
    h5f.create_dataset('DEF', data=np.atleast_2d(DEF))
    h5f.close()

    print(f'--MATERIALS: Fit {Path(filename).stem}')
    if verbose:
        print(f'{DEF=}')
    if plot:
        plot_DEF_admittance(fv, np.c_[D, E, F], model_Rf=R_target)


@click.command(name='admittance', help='Plot admittance from material file.')
@click.argument('material_file', nargs=1, type=click.Path(exists=True))
def main(material_file):
    frequencies = np.logspace(np.log10(10), np.log10(20e3), 4000)
    material = read_mat_DEF(Path(material_file))
    plot_DEF_admittance(frequencies, material)

# freq-independent impedance from reflection coefficients
# write_freq_ind_mat_from_Yn(convert_R_to_Yn(0.90),filename=Path(write_folder / 'R90_mat.h5'))
# write_freq_ind_mat_from_Yn(convert_R_to_Yn(0.5),filename=Path(write_folder / 'R50.h5'))

# #input DEF values directly
# write_freq_dep_mat(npa([[0,1.0,0],[2,3,4]]),filename=Path(write_folder / 'ex_mat.h5'))
