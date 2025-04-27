# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
from dataclasses import dataclass

import click
import numba as nb
import numpy as np
from numpy import exp, pi, cos, sqrt, log
from numpy.typing import ArrayLike
from scipy.fft import dct, idct  # default type2
from scipy.fft import rfft, irfft
from tqdm import tqdm

from pffdtd.geometry.math import iceil, iround

GAS_CONSTANT = 287.05  # Gas constant (J/Kg.K)
GAMMA = 1.402  # Specific heat ratio
AIR_DENSITY_0C = 1.293  # Air density at 0C (Kg.m^-3)
ONE_ATM = 101325.0  # One atmosphere (Pa)
KELVIN_OFFSET = 273.15  # Zero celsius in degrees Kelvin
AIR_VISCOSITY = 0.0000185  # Kinemetric viscosity of air (m^2/s)


@dataclass
class Air:
    temperature: np.float64
    pressure: np.float64
    density: np.float64
    velocity: np.float64
    impedance: np.float64
    tau_over_c: np.float64


def air_density(pressure, temp):
    return (pressure * ONE_ATM) / (GAS_CONSTANT * (temp + KELVIN_OFFSET))


def sound_velocity(temp):
    base = np.sqrt((GAMMA * ONE_ATM) / AIR_DENSITY_0C)
    return base * np.sqrt(1.0 + (temp / KELVIN_OFFSET))


def wave_number_in_air(air: Air, frequency):
    return air.tau_over_c * frequency


@dataclass
class AirAbsorption:
    gamma_p: np.float64
    gamma: np.float64
    etaO: np.float64
    eta: np.float64
    almN: np.float64
    almO: np.float64
    c: np.float64
    frO: np.float64
    frN: np.float64

    # frequency-dependent coefficeints in Np/m or dB/m
    absVibN_dB: np.ndarray
    absVibO_dB: np.ndarray
    absClRo_dB: np.ndarray
    absfull_dB: np.ndarray

    absVibN_Np: np.ndarray
    absVibO_Np: np.ndarray
    absClRo_Np: np.ndarray
    absfull_Np: np.ndarray


def air_absorption(
        frequencies: ArrayLike,
        temperature_celsius: float,
        rel_humidity_pnct: float,
        pressure_atmospheric_kPa: float = 101.325
) -> AirAbsorption:
    """This is an implementation of formulae in the ISO9613-1 standard for air absorption
    """
    assert pressure_atmospheric_kPa <= 200
    assert temperature_celsius >= -20
    assert temperature_celsius <= 50
    assert rel_humidity_pnct <= 100
    assert rel_humidity_pnct >= 10

    f = frequencies
    T = temperature_celsius
    rh = rel_humidity_pnct

    f2 = f*f
    pi2 = np.pi*np.pi

    # convert temperature to kelvin
    Tk = T+273.15

    # triple point isothermal temperature (Section B)
    T01 = 273.16  # kelvin

    # standard temperature
    T0 = 293.15

    # ambient pressure
    pa = 101.325  # kPa
    # reference pressure (standard)
    pr = 101.325  # kPa

    # characteristic vibrational temperature (A.8)
    thO = 2239.1
    thN = 3352.0
    # fractional molar concentrations (A.8)
    XO = 0.209
    XN = 0.781

    # constant (A.9)
    const = 2*np.pi/35*(10*np.log10(np.exp(2)))  # 1.559

    # (A.6)
    almO = const*XO*(thO/Tk)**2*np.exp(-thO/Tk)
    # (A.7)
    almN = const*XN*(thN/Tk)**2*np.exp(-thN/Tk)

    # pressure ratio
    p = pa/pr
    # temperature ratio
    Tr = Tk/T0

    # speed of sound
    c = 343.2*np.sqrt(Tr)
    c2 = c*c

    # (B.3)
    C = -6.8346*(T01/Tk)**1.261 + 4.6151
    # (B.1) and (B.2)
    h = rh*(10**C)*p

    # relaxation frequencies
    # (3)
    frO = p * (24 + 4.04e4 * h * (0.02 + h)/(0.391 + h))

    # (4)
    frN = p * Tr**(-0.5) * (9 + 280 * h * np.exp(-4.17 * (Tr**(-1/3) - 1)))

    # (5)
    absfull1 = 8.686*f2*(1.84e-11 * np.sqrt(Tr)/p + Tr**-2.5 * (0.01275*(
        np.exp(-2239.1/Tk)/(frO + f2/frO)) + 0.1068*(np.exp(-3352.0/Tk)/(frN + f2/frN))))

    # (A.2)
    absClRo = 1.6e-10*np.sqrt(Tr)*f2/p

    # derived
    eta = np.log(10)*1.6e-11/(4*pi2)*(c2)*np.sqrt(Tr)/p

    # (A.3)
    absVibO = almO*(f/c)*(2*(f/frO)/(1+(f/frO)**2))
    # (A.4)
    absVibN = almN*(f/c)*(2*(f/frN)/(1+(f/frN)**2))
    # (A.1)
    absfull2 = absClRo + absVibO + absVibN

    assert np.allclose(absfull1, absfull2, rtol=1e-2)

    # modified viscothermal coefficient (see FA2014 or DAFx2021 papers)
    etaO = almO*(c/pi2/frO)*np.log(10)/20

    # return a dictionary of different constants
    return AirAbsorption(
        gamma_p=etaO/c,
        gamma=eta/c,
        etaO=etaO,
        eta=eta,
        almN=almN,
        almO=almO,
        c=c,
        frO=frO,
        frN=frN,

        absVibN_dB=absVibN,
        absVibO_dB=absVibO,
        absClRo_dB=absClRo,
        absfull_dB=absfull2,
        absVibN_Np=absVibN*np.log(10)/20,
        absVibO_Np=absVibO*np.log(10)/20,
        absClRo_Np=absClRo*np.log(10)/20,
        absfull_Np=absfull2*np.log(10)/20,
    )


def apply_modal_filter(x, Fs, Tc, rh, pad_t=0.0):
    """
    This is an implementation of an air absorption filter based on a modal approach.
    It solves a system of 1-d dissipative wave equations tune to air attenuation
    curves using a soft-source boundary condition.

    See paper for details:
    Hamilton, B. "Adding air attenuation to simulated room impulse responses: A
    modal approach", to be presented at I3DA 2021 conference in Bologna Italy.
    """

    # apply filter, x is (Nchannel,Nsamples) array
    # see I3DA paper for details
    # Tc is temperature in deg Celsius
    # rh is relative humidity

    Ts = 1/Fs

    x = np.atleast_2d(x)
    Nt0 = x.shape[-1]
    Nt = iceil(pad_t/Ts)+Nt0
    xp = np.zeros((x.shape[0], Nt))
    xp[:, :Nt0] = x
    del x

    y = np.zeros(xp.shape)

    Nx = Nt
    wqTs = pi*(np.arange(Nx)/Nx)
    wq = wqTs/Ts

    rd = air_absorption(wq/2/pi, Tc, rh)
    alphaq = rd.absfull_Np
    c = rd.c

    P0 = np.zeros(xp.shape)
    P1 = np.zeros(xp.shape)

    fx = np.zeros(xp.shape)
    fx[:, 0] = 1
    Fm = dct(fx, type=2, norm='ortho', axis=-1)

    sigqTs = c*alphaq*Ts
    a1 = 2*exp(-sigqTs)*cos(wqTs)
    a2 = -exp(-2*sigqTs)
    Fmsig1 = Fm*(1+sigqTs/2)/(1+sigqTs)
    Fmsig2 = Fm*(1-sigqTs/2)/(1+sigqTs)

    u = np.zeros((xp.shape[0], Nt+1))
    u[:, 1:] = xp[:, ::-1]  # flip

    pbar = tqdm(total=Nt, desc='modal filter', ascii=True)

    @nb.jit(nopython=True, parallel=True)
    def _run_step(P0, P1, a1, a2, Fmsig1, Fmsig2, un1, un0):
        P0[:] = a1*P1 + a2*P0 + Fmsig1*un1 - Fmsig2*un0

    for n in range(Nt):
        # P0 = a1*P1 + a2*P0 + Fmsig1*u[:,n+1] - Fmsig2*u[:,n]
        for i in range(P0.shape[0]):
            _run_step(P0[i], P1[i], a1, a2, Fmsig1[i],
                      Fmsig2[i], u[i, n+1], u[i, n])
        if n < Nt-1:  # dont swap on last sample
            P1, P0 = P0, P1
        pbar.update(1)
    pbar.close()

    y = idct(P0, type=2, norm='ortho', axis=-1)
    return np.squeeze(y)  # squeeze to 1d in case


def apply_ola_filter(x, Fs, Tc, rh, Nw=1024):
    """
    This is an implementation of overlap-add (STFT/iSTFT) air absorption filtering.
    Tuned for 75% overlap and 1024-sample Hann window at 48kHz.

    - x is (Nchannels,Nsamples) array
    - Tc is temperature degrees Celsius
    - rh is relative humidity

    Used in paper:
    Hamilton, B. "Adding air attenuation to simulated room impulse responses: A
    modal approach", to be presented at I3DA 2021 conference in Bologna Italy.
    """
    Ts = 1/Fs

    x = np.atleast_2d(x)
    Nt0 = x.shape[-1]

    OLF = 0.75
    Ha = iround(Nw*(1-OLF))
    Nfft = np.int_(2**np.ceil(np.log2(Nw)))
    NF = iceil((Nt0+Nw)/Ha)
    Np = (NF-1)*Ha-Nt0
    assert Np >= Nw-Ha
    assert Np < Nw
    Nfft_h = np.int_(Nfft//2+1)

    xp = np.zeros((x.shape[0], Nw+Nt0+Np))
    xp[:, Nw:Nw+Nt0] = x
    y = np.zeros((x.shape[0], Nt0+Np))
    del x

    wa = 0.5*(1-np.cos(2*np.pi*np.arange(Nw)/Nw))  # hann window
    ws = wa/(3/8*Nw/Ha)  # scaled for COLA

    fv = np.arange(Nfft_h)/Nfft*Fs
    rd = air_absorption(fv, Tc, rh)
    c = rd.c
    absNp = rd.absfull_Np

    for i in range(xp.shape[0]):
        pbar = tqdm(total=NF, desc=f'OLA filter channel {i}', ascii=True)
        yp = np.zeros((xp.shape[-1],))
        for m in range(NF):
            na0 = m*Ha
            assert na0+Nw <= Nw+Nt0+Np
            dist = c*Ts*(na0-Nw/2)
            xf = xp[i, na0:na0+Nw]
            if dist < 0:  # dont apply gain (negative times - pre-padding)
                yp[na0:na0+Nw] += ws*xf
            else:
                Yf = rfft(wa*xf, Nfft)*np.exp(-absNp*dist)
                yf = irfft(Yf, Nfft)[:Nw]
                yp[na0:na0+Nw] += ws*yf

            pbar.update(1)
        y[i] = yp[Nw:]
        pbar.close()
    return np.squeeze(y)  # squeeze to 1d in case


def apply_visco_filter(x, Fs, Tc, rh, NdB=120, t_start=None):
    """
    This is an implementation of an air absorption filter based on approximate
    Green's function Stoke's equation (viscothermal wave equation)

    Main input being x, np.ndarray (Nchannels,Nsamples)
    enter temperature (Tc) and relative humidity (rh)
    NdB should be above 60dB, that is for truncation of Gaussian kernel

    See paper for details:
    Hamilton, B. "Air absorption filtering method based on approximate Green's
    function for Stokes' equation", to be presented at the DAFx2021 e-conference.
    """
    rd = air_absorption(1, Tc, rh)
    g = rd.gamma_p

    Ts = 1/Fs
    if t_start is None:
        t_start = Ts**2/(2*pi*g)
        print(f'{t_start=}')

    x = np.atleast_2d(x)
    Nt0 = x.shape[-1]

    n_last = Nt0-1
    dt_end = Fs*sqrt(0.1*log(10)*NdB*n_last*Ts*g)
    Nt = Nt0+iceil(dt_end)

    y = np.zeros((x.shape[0], Nt))
    n_start = iceil(t_start*Fs)
    assert n_start > 0

    y[:, :n_start] = x[:, :n_start]
    Tsg2 = 2*Ts*g
    Tsg2pi = 2*Ts*g*pi
    gTs = g*Ts
    dt_fac = 0.1*log(10)*NdB*gTs
    pbar = tqdm(total=Nt, desc='visco filter', ascii=True)
    for n in range(n_start, Nt0):
        dt = sqrt(dt_fac*n)/Ts
        dt_int = iceil(dt)
        nv = np.arange(n-dt_int, n+dt_int+1)
        assert n >= dt_int
        y[:, nv] += (Ts/sqrt(n*Tsg2pi))*x[:, n][:, None] * \
            exp(-((n-nv)*Ts)**2/(n*Tsg2))[None, :]
        pbar.update(1)
    pbar.close()

    return np.squeeze(y)  # squeeze to 1d in case


@click.command(name='air', help='Air absorption.')
def main():
    f = np.logspace(np.log10(1), np.log10(80e3))
    rh = 15
    Tc = 10
    print(f'{Tc=} {rh=}%')

    rd = air_absorption(f, Tc, rh)
    print(f"{rd.almO=}")
    print(f"{rd.almN=}")
    print(f"{rd.c=}")
    print(f"{rd.frO=}")
    print(f"{rd.frN=}")
    print(f"{rd.eta=}")

    rh = (100 - 10)*np.random.random_sample()+10
    Tc = (50 - -20)*np.random.random_sample()-20
    print(f'{Tc=} {rh=}%')
    rd = air_absorption(f, Tc, rh)
    print(f"{rd.almO=}")
    print(f"{rd.almN=}")
    print(f"{rd.c=}")
    print(f"{rd.frO=}")
    print(f"{rd.frN=}")
    print(f"{rd.eta=}")
