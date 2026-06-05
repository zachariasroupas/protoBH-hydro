#!/usr/bin/env python3
"""
================================================================================
3D BH Accretion Simulation Analysis — Simple Harmonic Orbit
Part of the project: protoBH — MSCA grant agreement No. 101149270 funded by the E.U.
Author: Zacharias Roupas
Please cite:
  Roupas, Z., to appear (2026)
  Code DOI: https://doi.org/10.5281/zenodo.XXXXXXX
Built on Athena++ v21.0 (Stone et al. 2020, ApJS 249, 4)
================================================================================
Disclaimer: This software is provided "as is", without warranty of any kind.
The author accepts no liability for any errors or consequences of its use.
Users are responsible for verifying results against the cited publication.
================================================================================
Unit system (set in C++ problem generator):
  length_unit  = R_Bondi_max = 2 G m_BH / (c_s^2 * (1 + Mach_min^2))
  velocity_unit = c_s
  time_unit    = R_Bondi_max / c_s
  density_unit = rho_gas = (1 - epsilon) * rho_cluster
================================================================================
Array layout in Athena++ 3D spherical HDF5 (after np.squeeze):
  shape  = (n_phi, n_theta, n_r)   axes (k, j, i)
  x1v    = r       (n_r,)
  x2v    = theta   (n_theta,)   0 = north pole, pi/2 = equator, pi = south pole
  x3v    = phi     (n_phi,)
  vel1   = v_r
  vel2   = v_theta
  vel3   = v_phi

Disk detection definitions
--------------------------
  R_outer  : outermost radius where beta > BETA_THRESHOLD AND vr/vff > VR_THRESHOLD
             at any snapshot up to k_final.  Always computed.  Used as the H(r)
             loop limit and outer xlim for radial profile plots.
  tau_outer: first time at which both beta AND vr thresholds are simultaneously
             satisfied at any radius.  Always computed.
  tau_bump : time of the first beta bump at the inner boundary that exceeds
             BETA_THRESHOLD.  Always attempted; None if no qualifying bump is found.
  tau_Hmax : earliest time at which both beta > BETA_THRESHOLD AND
             vr/vff > VR_THRESHOLD are simultaneously met at ir_disk
             (the grid point bracketing R_disk).
             Only computed when R_outer is found and a disk is detected.
  R_disk   : spline argmax of H_avg(r) subject to beta and vr thresholds at
             the final snapshot.  Only computed when R_outer is found.
"""

import gc
import json
import sys
import numpy as np
from pathlib import Path
import athena_read as ar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

# ============================================================
# TEE — mirror stdout to a log file
# ============================================================

class _Tee:
    """Write to both the real stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._log    = open(log_path, 'w', buffering=1)
        sys.stdout   = self
    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)
    def flush(self):
        self._stdout.flush()
        self._log.flush()
    def close(self):
        sys.stdout = self._stdout
        self._log.close()

# ============================================================
# CONFIGURATION
# ============================================================
DIR_NAME     = "mBH50.0_B0.010_ecc0.00_psi00.00_nR128_nT105_nP240_rho1.0e+07/nu0e+00_NonI1_Rinn5.0e-03_Rout2.00_tend12.00"
SIM_DATA_DIR = Path(f"./simulation_3D/{DIR_NAME}")
OUTPUT_DIR   = Path(f"./images_3D/{DIR_NAME}")

BETA_THRESHOLD = 0.1
# v_r/v_ff threshold: equipartition condition at beta = BETA_THRESHOLD
VR_THRESHOLD   = - np.sqrt(BETA_THRESHOLD / 2.0)

FINAL_SNAP_IDX = None   # None -> last available snapshot

# fit of H(r)
SPLINE_SMOOTH_FACTOR = 2.0   # increase to smooth more; decrease to follow data closer

matplotlib.rcParams.update({
    'text.usetex':       True,
    'font.family':       'serif',
    'axes.labelsize':    24,
    'xtick.labelsize':   16,
    'ytick.labelsize':   16,
    'legend.fontsize':   12,
    'xtick.major.size':  6,
    'ytick.major.size':  6,
    'xtick.minor.size':  4,
    'ytick.minor.size':  4,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.top':         True,
    'ytick.right':       True,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'lines.linewidth':   2.0,
    'axes.grid':         False,
    'figure.figsize':    (8, 5),
    'savefig.dpi':       300,
})

# ============================================================
# I/O
# ============================================================

def read_parameters(sim_dir):
    with open(sim_dir / "parameters.json") as f:
        return json.load(f)

def read_snapshot_paths(sim_dir):
    prim_files = sorted(sim_dir.glob("*.prim.*.athdf"))
    if not prim_files:
        raise FileNotFoundError(f"No prim files in {sim_dir}")
    return prim_files

# ============================================================
# GRID HELPERS
# ============================================================

def orbital_period(params):
    return 2.0 * np.pi / params['code_units']['Omega_code']

def find_equatorial_index(x2v):
    return int(np.argmin(np.abs(x2v - 0.5 * np.pi)))

def gen_radial_ind(x1v):
    n_r = len(x1v)
    return 0, (n_r - 1) // 2, n_r - 1

# ============================================================
# PHI-AVERAGING HELPERS
# ============================================================

def phi_avg_equatorial(snap, field, j_eq):
    data = np.squeeze(snap[field])
    return np.mean(data[:, j_eq, :], axis=0)

def phi_avg_theta_profile(snap, field, i_r):
    data = np.squeeze(snap[field])
    return np.mean(data[:, :, i_r], axis=0)

def phi_avg_all(snap, field):
    data = np.squeeze(snap[field])
    return np.mean(data, axis=0)            # (n_theta, n_r)

# ============================================================
# DATA EXTRACTION
# ============================================================

def extract_timeseries(prim_files, x1v, x2v, params):
    j_eq  = find_equatorial_index(x2v)
    gm    = params['code_units']['gm_code']
    T_orb = orbital_period(params)
    v_kep = np.sqrt(gm / x1v)
    n_snap, n_r = len(prim_files), len(x1v)
    t_per     = np.zeros(n_snap)
    vphi_avg  = np.zeros((n_snap, n_r))
    vr_avg    = np.zeros((n_snap, n_r))
    beta_prof = np.zeros((n_snap, n_r))
    print(f"  Reading {n_snap} snapshots (vel1 + vel3 only) ...")
    for k, pf in enumerate(prim_files):
        snap         = ar.athdf(str(pf), quantities=['vel1', 'vel3'])
        t_per[k]     = snap['Time'] / T_orb
        vphi_avg[k]  = phi_avg_equatorial(snap, 'vel3', j_eq)
        vr_avg[k]    = phi_avg_equatorial(snap, 'vel1', j_eq)
        beta_prof[k] = (vphi_avg[k] / v_kep) ** 2
        del snap; gc.collect()
        if (k + 1) % 10 == 0 or (k + 1) == n_snap:
            print(f"    {k + 1}/{n_snap} done", flush=True)
    beta_prof = np.where(np.isfinite(beta_prof) & (beta_prof > 0.0), beta_prof, np.nan)
    return t_per, vphi_avg, vr_avg, beta_prof, v_kep, j_eq

def extract_theta_profiles(prim_files, x2v, i_r, k_indices, fields):
    n_theta = len(x2v)
    result  = {f: np.zeros((len(k_indices), n_theta)) for f in fields}
    for ii, k in enumerate(k_indices):
        snap = ar.athdf(str(prim_files[k]), quantities=fields)
        for f in fields:
            result[f][ii] = phi_avg_theta_profile(snap, f, i_r)
        del snap; gc.collect()
    return result

# ============================================================
# DISK DETECTION  -- R_outer and tau_outer
# ============================================================

def scan_threshold_crossings(vphi_avg_kep, vr_avg_ff, x1v, t_per,
                              idx_inner, idx_outer, threshold_beta,
                              threshold_vr, it_last):
    """
    Scan all (time, radius) pairs up to it_last for simultaneous threshold
    crossings in the disk diagnostics:
        beta  = (<v_phi> / v_Kep)^2  >  threshold_beta
        vr/vff                        >  threshold_vr   (negative, inflow)

    Returns
    -------
    tau_outer : float or None
        Time (in P_orb) of the FIRST snapshot at which any radius satisfies
        both thresholds simultaneously.  None if never satisfied.
    R_outer   : float or None
        OUTERMOST radius (in R_B) satisfying both thresholds at any snapshot
        up to it_last.  Used as the H(r) loop limit and outer xlim.
        None if never satisfied.
    ir_out    : int or None
        Radial index corresponding to R_outer.
    k_out     : int or None
        Snapshot index at which R_outer was first reached.
    """
    t_out = x1v_out = ir_out = k_out = None

    # If the outer boundary cell satisfies beta at t=0, walk inward to find
    # where it drops below threshold, to avoid flagging the boundary ghost.
    beta_bc  = vphi_avg_kep[0, idx_outer] ** 2
    ir_bc    = idx_outer
    bc_found = False
    if beta_bc > threshold_beta:
        ir = idx_outer
        while ir > idx_inner and not bc_found:
            if vphi_avg_kep[it_last, ir] ** 2 < threshold_beta:
                ir_bc    = ir
                bc_found = True
            ir -= 1

    # Scan all snapshots and radii within [inner, ir_bc]
    disk_found = False
    for k in range(0, it_last + 1):
        for ir in range(idx_inner, ir_bc + 1):
            vr   = vr_avg_ff[k, ir]
            beta = vphi_avg_kep[k, ir] ** 2
            if beta > threshold_beta and vr > threshold_vr:
                if not disk_found:
                    x1v_out    = x1v[ir];  ir_out = ir
                    t_out      = t_per[k]; k_out  = k
                    disk_found = True
                elif x1v[ir] > x1v_out:
                    x1v_out = x1v[ir];  ir_out = ir
                    t_out   = t_per[k]; k_out  = k

    return t_out, x1v_out, ir_out, k_out

# ============================================================
# DISK DETECTION  -- tau_bump
# ============================================================

def detect_bump_initiation(vphi_avg_kep, t_per, idx_inner, threshold_beta):
    """
    Locate tau_bump: the first snapshot at the inner boundary where
    a bump in v_phi/v_Kep is detected with beta > threshold_beta.

    NaN entries are treated as zero for peak detection.

    Returns (tau_bump, k_bump) or (None, None).
    """
    series       = - vphi_avg_kep[:, idx_inner] 
    series_clean = np.where(np.isfinite(series), series, 0.0)
    beta_clean   = series_clean ** 2
    peaks, _     = find_peaks(series_clean)

    out_t = None
    out_k = None

    valid_peaks = peaks[beta_clean[peaks] > threshold_beta]

    if len(valid_peaks) > 0:
        k_0 = valid_peaks[0]
        out_t  = t_per[k_0]
        out_k  = int(k_0)

    return out_t, out_k

# ============================================================
# DISK DETECTION  -- R_disk and tau_Hmax
# ============================================================

def detect_disk_radius_Hmax(H_N_r, H_S_r, x1v, vphi_avg_kep, vr_avg_ff,
                             t_per, k_final, threshold_beta, threshold_vr):
    """
    Fit a smoothing spline to H_avg(r) in log(r) space, anchored to zero
    two cells beyond the outermost valid radius, and scan the time series
    for the disk formation time at the radius of maximum H.

    Returns
    -------
    R_disk     : spline argmax of H_avg(r); characteristic disk radius
    ir_disk    : bracketing grid index with the larger H_avg
    H_max      : spline maximum value of H_avg
    theta_half : opening half-angle arcsin(H_max / R_disk), in degrees
    tau_Hmax   : earliest time both thresholds are simultaneously met at ir_disk
    k_Hmax     : snapshot index corresponding to tau_Hmax
    spl        : fitted spline object (x-axis: log r)
    r_valid    : radii of valid data points used for spline fitting
    H_valid    : H_avg values at valid radii

    Returns (R_disk, ir_disk, H_max, theta_half, tau_Hmax, k_Hmax, spl, r_valid, H_valid)
    or      (None,   None,    0.0,   None,       None,     None,   None, None,    None)
    on failure.
    """
    _FAIL = (None, None, 0.0, None, None, None, None, None, None)

    H_avg = 0.5 * (H_N_r + H_S_r)
    valid = ((vphi_avg_kep[k_final] ** 2 > threshold_beta) &
             (vr_avg_ff[k_final]         > threshold_vr)   &
             (H_avg                      > 0.0))
    idx   = np.where(valid)[0]
    if len(idx) < 6:
        return _FAIL

    r_v, H_v = x1v[idx], H_avg[idx]
    R_outer  = float(r_v[-1])

    # Spline in log(r); zero anchor two cells beyond valid range
    r_anc = x1v[min(idx[-1] + 2, len(x1v) - 1)]
    log_r = np.log(np.append(r_v, r_anc))
    H_fit = np.append(H_v, 0.0)
    noise = float(np.median(np.abs(np.diff(H_v)))) or float(np.std(H_v)) * 0.1
    try:
        spl = UnivariateSpline(log_r, H_fit, k=3,
                               s=SPLINE_SMOOTH_FACTOR * len(H_fit) * noise**2,
                               ext=1)
    except Exception:
        return _FAIL

    # Fine grid restricted to valid data range (anchor excluded by construction)
    r_f  = np.exp(np.linspace(log_r[0], np.log(R_outer), 2000))
    H_f  = spl(np.log(r_f))

    ip    = int(np.argmax(H_f))
    H_max = float(H_f[ip])
    if H_max <= 0.0:
        return _FAIL
    R_disk     = float(r_f[ip])
    theta_half = float(np.degrees(np.arcsin(np.clip(H_max / R_disk, 0.0, 1.0))))

    # ir_disk: whichever bracketing grid point has the larger H_avg
    i_right = int(np.searchsorted(x1v, R_disk))
    i_left  = max(i_right - 1, 0)
    i_right = min(i_right, len(x1v) - 1)
    ir_disk = i_left if H_avg[i_left] >= H_avg[i_right] else i_right

    # tau_Hmax: earliest time both thresholds are simultaneously met at ir_disk
    tau_Hmax = None
    k_Hmax   = None
    found    = False
    k        = 0
    while k < len(t_per) and not found:
        if (vphi_avg_kep[k, ir_disk] ** 2 > threshold_beta and
                vr_avg_ff[k, ir_disk]     > threshold_vr):
            tau_Hmax = t_per[k]
            k_Hmax   = int(k)
            found    = True
        k += 1

    return R_disk, ir_disk, H_max, theta_half, tau_Hmax, k_Hmax, spl, r_v, H_v

# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_summary(params, prim_files, t_per, j_eq, x2v,
                  tau_bump, tau_Hmax, R_disk, R_outer, tau_outer,
                  T_orb_Myr, T_orb_yr,
                  R_circ_code, R_circ_pc,
                  tau_ff_Porb, tau_ff_yr,
                  tau_prop_Porb, tau_prop_yr):
    pi  = params['physical_input']
    dq  = params['derived_quantities_cgs']
    us  = params['unit_system_cgs']
    cu  = params['code_units']
    cst = params['physical_constants_cgs']
    print("\n" + "="*60)
    print("SIMULATION  --  ORBITING BH  3D Spherical  (Elliptical Orbit)")
    print("="*60)
    print(f"  M_BH           = {pi['M_BH_Msun']:.1f} M_sun")
    print(f"  B (minor)      = {pi['B_pc']:.4e} pc"
          f"  |  A (major) = {dq['A_pc']:.4e} pc")
    print(f"  eccentricity   = {pi['ecc']:.4e}")
    print(f"  rho_gas        = {pi['rho_gas_cgs']:.3e} g/cm^3")
    print(f"  T_gas          = {pi['T_gas_K']:.1f} K")
    print(f"  c_s            = {dq['c_s_cgs']*1e-5:.4e} km/s")
    print(f"  Orbital period = {T_orb_Myr:.4e} Myr  ({T_orb_yr:.4e} yr)")
    print(f"  length_unit    = {us['length_unit_cgs']/cst['au_cgs']:.4e} AU"
          f"  = {us['length_unit_cgs']/cst['pc_cgs']:.4e} pc")
    print(f"  time_unit      = {us['time_unit_yr']:.4e} yr")
    print(f"  gm_code        = {cu['gm_code']:.4e}")
    print(f"  Omega_code     = {cu['Omega_code']:.4e}")
    print(f"  Equatorial j   = {j_eq}  (theta = {x2v[j_eq]*180/np.pi:.2f} deg)")
    print(f"  Snapshots      = {len(prim_files)}"
          f"  ({prim_files[0].name} ... {prim_files[-1].name})")
    print(f"  t_end          = {t_per[-1]:.4e} P_orb")
    print(f"\n  beta threshold = {BETA_THRESHOLD}")
    print(f"  vr   threshold = {VR_THRESHOLD}")
    print(f"\n  R_circ         = {R_circ_code:.4e} R_B     =  {R_circ_pc:.4e} pc")
    print(f"  tau_ff (R_B)   = {tau_ff_Porb:.4e} P_orb  =  {tau_ff_yr:.4e} yr")
    print(f"  tau_prop(R_c)  = {tau_prop_Porb:.4e} P_orb  =  {tau_prop_yr:.4e} yr")
    if R_outer is not None:
        print(f"\n  R_outer    (outermost threshold radius)   = {R_outer:.4e} R_B")
        print(f"  tau_outer  (first threshold crossing)     = {tau_outer:.4e} P_orb")
    else:
        print(f"\n  R_outer    : no radius satisfying both thresholds detected")
    if tau_bump is not None:
        print(f"  tau_bump   (first beta bump, inner bdy)   = {tau_bump:.4e} P_orb")
    else:
        print(f"  tau_bump   : no qualifying bump at inner boundary")
    if tau_Hmax is not None:
        print(f"  tau_Hmax   (first crossing at R_disk)     = {tau_Hmax:.4e} P_orb")
    else:
        print(f"  tau_Hmax   : not computed (no H maximum detected)")
    if R_disk is not None:
        print(f"  R_disk     (radius of H maximum)          = {R_disk:.4e} R_B")
    elif R_outer is None:
        print(f"  R_disk     : not computed (no R_outer found)")
    else:
        print(f"  R_disk     : not found (no valid H maximum under thresholds)")
    print("="*60 + "\n")

# ============================================================
# HELPERS
# ============================================================

def save_fig(fig, name):
    fig.savefig(OUTPUT_DIR / name, bbox_inches='tight')
    plt.close(fig)

def style_ax(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(which='both', top=True, right=True, direction='in')

def safe_log_ylim(data, margin=2.0, floor=1.0e-6):
    finite = data[np.isfinite(data) & (data > 0.0)]
    if finite.size == 0:
        return None
    return max(floor, finite.min() / margin), finite.max() * margin

def theta_xy_deg(x2v):
    return (x2v - 0.5 * np.pi) * 180.0 / np.pi

# ============================================================
# DISK OPENING  (3D-only)
# ============================================================

def _find_theta_edges(vphi_theta, x2v, j_eq, r, params, threshold_beta):
    """
    Find outermost theta indices where beta > threshold_beta in each hemisphere.
    Returns (j_north, j_south); either may be None if no crossing is found.
    """
    gm         = params['code_units']['gm_code']
    v_kep      = np.sqrt(gm / r)
    beta_theta = (vphi_theta / v_kep) ** 2
    n_theta    = len(x2v)

    j_north = None
    for j in range(0, j_eq + 1):
        if beta_theta[j] > threshold_beta:
            j_north = j; break

    j_south = None
    for j in range(n_theta - 1, j_eq - 1, -1):
        if beta_theta[j] > threshold_beta:
            j_south = j; break

    return j_north, j_south

def _disk_height(vphi_theta, x2v, j_eq, r, params, threshold_beta):
    gm      = params['code_units']['gm_code']
    v_kep   = np.sqrt(gm / r)
    beta_eq = (vphi_theta[j_eq] / v_kep) ** 2

    if not np.isfinite(beta_eq) or beta_eq <= threshold_beta:
        return 0.0, 0.0

    j_north, j_south = _find_theta_edges(
        vphi_theta, x2v, j_eq, r, params, threshold_beta)

    if j_north is None: j_north = j_eq
    if j_south is None: j_south = j_eq

    return r * np.cos(x2v[j_north]), r * np.abs(np.cos(x2v[j_south]))

def compute_H_profile(vphi_all, x1v, x2v, j_eq, ir_outer, params, threshold_beta):
    H_N = np.zeros(len(x1v))
    H_S = np.zeros(len(x1v))
    for ir in range(ir_outer + 1):
        H_N[ir], H_S[ir] = _disk_height(
            vphi_all[:, ir], x2v, j_eq, x1v[ir], params, threshold_beta)
    return H_N, H_S

# ============================================================
# EQUATORIAL PLOT FUNCTIONS
# ============================================================

def plot_vphi_over_vkep(t_per, vphi_avg, x1v, idx_inner, params, ir_disk, disk_k):
    gm    = params['code_units']['gm_code']
    r     = x1v[idx_inner]
    v_kep = np.sqrt(gm / r)
    fig, ax = plt.subplots()
    ax.plot(t_per, vphi_avg[:, idx_inner] / v_kep, 'b-', lw=1.8)
    ax.axhline(0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if disk_k is not None and ir_disk is not None:
        ax.axvline(t_per[disk_k], color='k', ls='--', lw=0.8, alpha=0.5)
        vphi_steady = 1.0 if vphi_avg[disk_k, ir_disk] > 0 else -1.0
        ax.axhline(vphi_steady, color='k', ls='--', lw=0.8, alpha=0.5)
    style_ax(ax, r'$t \;/\; P_{\rm orb}$',
             r'$\langle v_\phi \rangle_\phi \;/\; v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig, f'vphi_over_vkep_inner_r{r:.4f}.png')

def plot_vr(t_per, vr_avg, x1v, idx_inner, tau_bump):
    r = x1v[idx_inner]
    fig, ax = plt.subplots()
    ax.plot(t_per, vr_avg[:, idx_inner], 'g-', lw=1.8)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    if tau_bump is not None:
        ax.axvline(tau_bump, color='k', ls='--', lw=0.8, alpha=0.5)
    style_ax(ax, r'$t \;/\; P_{\rm orb}$', r'$\langle v_r \rangle_\phi \;/\; c_s$')
    plt.tight_layout()
    save_fig(fig, f'vr_inner_r{r:.4f}.png')

def plot_beta_timeseries(t_per, beta_prof, x1v, disk_t_per,
                         idx_inner, idx_median, idx_outer):
    configs = [(idx_inner,'inner','b'),(idx_median,'median','r'),(idx_outer,'outer','darkorange')]
    for idx, label, color in configs:
        r = x1v[idx]; series = beta_prof[:, idx]
        fig, ax = plt.subplots()
        ax.plot(t_per, series, color=color, lw=1.8)
        ax.axhline(1.0,           color='k', ls='--', lw=0.8, alpha=0.5)
        ax.axhline(BETA_THRESHOLD, color='r', ls='--', lw=0.8, alpha=0.5)
        ax.set_yscale('log')
        lim = safe_log_ylim(series)
        if disk_t_per is not None:
            ax.axvline(disk_t_per, color='k', ls='--', lw=0.8, alpha=0.5)
        if lim is not None:
            ax.set_ylim(*lim)
        else:
            ax.set_yscale('linear')
            print(f"  WARNING: beta_{label} has no positive values -- using linear scale")
        style_ax(ax, r'$t \;/\; P_{\rm orb}$',
                 r'$\beta = \langle v_\phi \rangle_\phi^2 \;/\; v_{\rm Kep}^2$')
        plt.tight_layout()
        save_fig(fig, f'beta_r{r:.4f}_{label}.png')

def plot_beta_profile(t_per, beta_prof, x1v, k, label, r_disk):
    profile = beta_prof[k, :]
    lim = safe_log_ylim(profile)
    if lim is None:
        print(f"  WARNING: skipping beta_profile '{label}' -- no positive beta at k={k}")
        return
    fig, ax = plt.subplots()
    ax.plot(x1v, profile, 'b-', lw=1.8)
    ax.axhline(1.0,           color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(BETA_THRESHOLD, color='r', ls='--', lw=0.8, alpha=0.5)
    if r_disk is not None:
        ax.axvline(r_disk, color='b', ls=':', lw=1.0)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(*lim)
    style_ax(ax, r'$r/R_{\rm B}$',
             r'$\beta = \langle v_\phi \rangle_\phi^2 \;/\; v_{\rm Kep}^2$')
    plt.tight_layout()
    save_fig(fig, f'beta_profile_t{t_per[k]:.4f}_{label.replace(" ","_")}.png')

def plot_profile_disk(t_per, vphi_avg_kep, vr_avg, x1v,
                      disk_t_per, r_disk, ir_disk, disk_k):
    vphi_t = vphi_avg_kep[:, ir_disk];  vphi_r = vphi_avg_kep[disk_k, :]
    vr_t   = vr_avg[:, ir_disk];        vr_r   = vr_avg[disk_k, :]
    vphi_steady = 1.0 if vphi_t[disk_k] > 0 else -1.0
    fig1, ax1 = plt.subplots()
    ax1.plot(x1v, vphi_r, 'b-', lw=1.8); ax1.set_xscale('log')
    ax1.axhline(0.0, color='b', ls=':', lw=1.0)
    ax1.axhline(vphi_steady, color='b', ls=':', lw=1.0)
    ax1.axvline(r_disk,      color='b', ls=':', lw=1.0)
    style_ax(ax1, r'$r/R_{\rm B}$',
             r'$\langle v_\phi(t_{\rm d},r) \rangle_\phi \;/\; v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig1, f'disk_detection_vp_r_t{disk_t_per:.4f}.png')
    fig2, ax2 = plt.subplots()
    ax2.plot(t_per, vphi_t, 'b-', lw=1.8)
    ax2.axhline(0.0, color='b', ls=':', lw=1.0)
    ax2.axhline(vphi_steady, color='b', ls=':', lw=1.0)
    ax2.axvline(disk_t_per,  color='b', ls=':', lw=1.0)
    style_ax(ax2, r'$t \;/\; P_{\rm orb}$',
             r'$\langle v_\phi(t,r_{\rm d}) \rangle_\phi \;/\; v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig2, f'disk_detection_vp_t_r{r_disk:.4f}.png')
    fig3, ax3 = plt.subplots()
    ax3.plot(x1v, vr_r, 'b-', lw=1.8); ax3.set_xscale('log')
    ax3.axhline(0.0, color='b', ls=':', lw=1.0)
    ax3.axvline(r_disk, color='b', ls=':', lw=1.0)
    style_ax(ax3, r'$r/R_{\rm B}$', r'$\langle v_r(t_{\rm d},r) \rangle_\phi \;/\; c_s$')
    plt.tight_layout()
    save_fig(fig3, f'disk_detection_vr_r_t{disk_t_per:.4f}.png')
    fig4, ax4 = plt.subplots()
    ax4.plot(t_per, vr_t, 'b-', lw=1.8)
    ax4.axhline(0.0, color='b', ls=':', lw=1.0)
    ax4.axvline(disk_t_per, color='b', ls=':', lw=1.0)
    style_ax(ax4, r'$t \;/\; P_{\rm orb}$',
             r'$\langle v_r(t,r_{\rm d}) \rangle_\phi \;/\; c_s$')
    plt.tight_layout()
    save_fig(fig4, f'disk_detection_vr_t_r{r_disk:.4f}.png')

def plot_beta_overlay(t_per, beta_prof, x1v, idx_inner, ir_disk, r_disk, disk_t_per):
    r_inner = x1v[idx_inner]; r_disk = x1v[ir_disk]
    idx_1 = np.argmin(np.abs(x1v - 0.5*(r_inner+r_disk)))
    idx_2 = np.argmin(np.abs(x1v - 5.0*r_disk))
    idx_3 = np.argmin(np.abs(x1v - 10.0*r_disk))
    r_1, r_2, r_3 = x1v[idx_1], x1v[idx_2], x1v[idx_3]
    fig, ax = plt.subplots()
    ax.plot(t_per, beta_prof[:, idx_inner], color='m',          ls='--', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_inner)
    ax.plot(t_per, beta_prof[:, idx_1],     color='r',          ls='-.', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_1)
    ax.plot(t_per, beta_prof[:, ir_disk],   color='b',          ls='-',
            label=r'$r = %.4f \; R_{\rm B}$' % r_disk)
    ax.plot(t_per, beta_prof[:, idx_2],     color='darkorange', ls=':', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_2)
    ax.plot(t_per, beta_prof[:, idx_3],     color='k',          ls=':', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_3)
    ax.axhline(1.0,           color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(BETA_THRESHOLD, color='r', ls='--', lw=0.8, alpha=0.5)
    ax.axvline(disk_t_per,    color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_yscale('log')
    lim = safe_log_ylim(beta_prof[:, [idx_inner, idx_1, ir_disk, idx_2, idx_3]])
    if lim is not None: ax.set_ylim(*lim)
    else: ax.set_yscale('linear')
    ax.legend(frameon=False)
    style_ax(ax, r'$t \;/\; P_{\rm orb}$',
             r'$\beta = \langle v_\phi \rangle_\phi^2 \;/\; v_{\rm Kep}^2$')
    plt.tight_layout()
    save_fig(fig, 'beta_vs_t_overlay_r.png')

def plot_vphi_kep_overlay(t_per, vphi_avg_kep, x1v, idx_inner, ir_disk, r_disk, disk_t_per):
    r_inner = x1v[idx_inner]; r_disk = x1v[ir_disk]
    idx_1 = np.argmin(np.abs(x1v - 0.5*(r_inner+r_disk)))
    idx_2 = np.argmin(np.abs(x1v - 5.0*r_disk))
    idx_3 = np.argmin(np.abs(x1v - 10.0*r_disk))
    r_1, r_2, r_3 = x1v[idx_1], x1v[idx_2], x1v[idx_3]
    fig, ax = plt.subplots()
    ax.plot(t_per, vphi_avg_kep[:, idx_inner], color='m',          ls='--', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_inner)
    ax.plot(t_per, vphi_avg_kep[:, idx_1],     color='r',          ls='-.', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_1)
    ax.plot(t_per, vphi_avg_kep[:, ir_disk],   color='b',          ls='-',
            label=r'$r = %.4f \; R_{\rm B}$' % r_disk)
    ax.plot(t_per, vphi_avg_kep[:, idx_2],     color='darkorange', ls=':', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_2)
    ax.plot(t_per, vphi_avg_kep[:, idx_3],     color='k',          ls=':', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_3)
    vmax = np.nanmax(vphi_avg_kep[:, [idx_inner, idx_1, ir_disk, idx_2, idx_3]])
    ax.axhline( 0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if vmax > 1.0: ax.axhline(1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.axvline(disk_t_per, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.legend(frameon=False)
    style_ax(ax, r'$t \;/\; P_{\rm orb}$',
             r'$\langle v_\phi \rangle_\phi \;/\; v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig, 'vphi_vs_t_multiradius.png')

def plot_vr_ff_overlay(t_per, vr_avg_ff, x1v, idx_inner, ir_disk, disk_t_per):
    r_inner = x1v[idx_inner]; r_disk = x1v[ir_disk]
    idx_1 = np.argmin(np.abs(x1v - 0.5*(r_inner+r_disk)))
    idx_2 = np.argmin(np.abs(x1v - 5.0*r_disk))
    idx_3 = np.argmin(np.abs(x1v - 10.0*r_disk))
    r_1, r_2, r_3 = x1v[idx_1], x1v[idx_2], x1v[idx_3]
    fig, ax = plt.subplots()
    ax.plot(t_per, vr_avg_ff[:, idx_inner], color='m',          ls='--', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_inner)
    ax.plot(t_per, vr_avg_ff[:, idx_1],     color='r',          ls='-.', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_1)
    ax.plot(t_per, vr_avg_ff[:, ir_disk],   color='b',          ls='-',
            label=r'$r = %.4f \; R_{\rm B}$' % r_disk)
    ax.plot(t_per, vr_avg_ff[:, idx_2],     color='darkorange', ls=':', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_2)
    ax.plot(t_per, vr_avg_ff[:, idx_3],     color='k',          ls=':', lw=1.0,
            label=r'$r = %.4f \; R_{\rm B}$' % r_3)
    vmax = np.nanmax(vr_avg_ff[:, [idx_inner, idx_1, ir_disk, idx_2, idx_3]])
    ax.axhline( 0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if vmax > 1.0: ax.axhline(1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.axvline(disk_t_per, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.legend(frameon=False)
    style_ax(ax, r'$t \;/\; P_{\rm orb}$',
             r'$\langle v_r \rangle_\phi \;/\; v_{\rm ff}(r)$')
    plt.tight_layout()
    save_fig(fig, 'vr_vs_t_multiradius.png')

def plot_vphi_kep_multitime(t_per, vphi_avg_kep, x1v, disk_k, r_disk, k_final):
    k_half  = np.argmin(np.abs(t_per - t_per[disk_k] / 2.0))
    indices = [0, k_half, disk_k, k_final]
    configs = [
        (0,       'k',  '--', r'$t = 0$'),
        (k_half,  'g',  ':',  r'$t = %.3f \; P_{\rm orb}$' % t_per[k_half]),
        (disk_k,  'b',  '-',  r'$t = %.3f \; P_{\rm orb}$' % t_per[disk_k]),
        (k_final, 'r',  '-.', r'$t = %.3f \; P_{\rm orb}$' % t_per[k_final]),
    ]
    vmax = np.nanmax(vphi_avg_kep[np.ix_(indices, range(len(x1v)))])
    vmin = np.nanmin(vphi_avg_kep[np.ix_(indices, range(len(x1v)))])
    fig, ax = plt.subplots()
    for k, color, ls, lbl in configs:
        ax.plot(x1v, vphi_avg_kep[k, :], color=color, ls=ls, label=lbl)
    ax.axhline(0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if vmax > 1.0: ax.axhline( 1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    if vmin < -1.0: ax.axhline(-1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    if r_disk is not None: ax.axvline(r_disk, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xscale('log'); ax.legend(frameon=False)
    style_ax(ax, r'$r \;/\; R_{\rm B}$',
             r'$\langle v_\phi \rangle_\phi \;/\; v_{\rm Kep}$')
    plt.tight_layout(); save_fig(fig, 'vphi_vs_r_multitime.png')

def plot_vr_ff_multitime(t_per, vr_avg_ff, x1v, disk_k, r_disk, k_final):
    k_half  = np.argmin(np.abs(t_per - t_per[disk_k] / 2.0))
    indices = [k_half, disk_k, k_final]
    configs = [
        (k_half,  'g', ':',  r'$t = %.3f \; P_{\rm orb}$' % t_per[k_half]),
        (disk_k,  'b', '-',  r'$t = %.3f \; P_{\rm orb}$' % t_per[disk_k]),
        (k_final, 'r', '-.', r'$t = %.3f \; P_{\rm orb}$' % t_per[k_final]),
    ]
    vmax = np.nanmax(vr_avg_ff[np.ix_(indices, range(len(x1v)))])
    vmin = np.nanmin(vr_avg_ff[np.ix_(indices, range(len(x1v)))])
    fig, ax = plt.subplots()
    for k, color, ls, lbl in configs:
        ax.plot(x1v, vr_avg_ff[k, :], color=color, ls=ls, label=lbl)
    ax.axhline(0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if vmin < -1.0: ax.axhline(-1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    if vmax > 1.0:  ax.axhline( 1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    if r_disk is not None: ax.axvline(r_disk, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xscale('log'); ax.legend(frameon=False)
    style_ax(ax, r'$r \;/\; R_{\rm B}$',
             r'$\langle v_r \rangle_\phi \;/\; v_{\rm ff}(r)$')
    plt.tight_layout(); save_fig(fig, 'vr_vs_r_multitime.png')

def plot_vphi_kep_profile(t_per, vphi_avg_kep, x1v, k, label, r_disk=None):
    profile = vphi_avg_kep[k, :]
    fig, ax = plt.subplots()
    ax.plot(x1v, profile, 'b-', lw=1.8)
    ax.axhline( 0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axhline( 1.0, color='k', ls=':',  lw=0.8, alpha=0.5)
    ax.axhline(-1.0, color='k', ls=':',  lw=0.8, alpha=0.5)
    if r_disk is not None:
        ax.axvline(r_disk, color='b', ls=':', lw=1.0, label=r'$r_{\rm disk}$')
        ax.legend()
    ax.set_xscale('log')
    style_ax(ax, r'$r/R_{\rm B}$',
             r'$\langle v_\phi \rangle_\phi \;/\; v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig, f'vphi_kep_profile_{label.replace(" ","_")}.png')

def plot_vr_freefall_profile(t_per, vr_avg, x1v, params, k, label, r_disk=None):
    gm   = params['code_units']['gm_code']
    v_ff = np.sqrt(2.0 * gm / x1v)
    ratio = vr_avg[k, :] / v_ff
    fig, ax = plt.subplots()
    ax.plot(x1v, ratio, 'g-', lw=1.8)
    ax.axhline( 0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(-1.0, color='k', ls=':',  lw=0.8, alpha=0.5, label=r'$v_r = -v_{\rm ff}$')
    if r_disk is not None:
        ax.axvline(r_disk, color='g', ls=':', lw=1.0, label=r'$r_{\rm disk}$')
    ax.legend(fontsize=10); ax.set_xscale('log')
    style_ax(ax, r'$r/R_{\rm B}$', r'$\langle v_r \rangle_\phi \;/\; v_{\rm ff}(r)$')
    plt.tight_layout()
    save_fig(fig, f'vr_freefall_profile_{label.replace(" ","_")}.png')

def plot_vphi_radial_envelope(t_per, vphi_avg, x1v, params):
    gm    = params['code_units']['gm_code']
    v_kep = np.sqrt(gm / x1v)
    ratio = vphi_avg / v_kep[np.newaxis, :]
    var_amplitude = np.nanmax(ratio, axis=0) - np.nanmin(ratio, axis=0)
    fig, ax = plt.subplots()
    ax.plot(x1v, var_amplitude, 'b-', lw=1.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    style_ax(ax, r'$r/R_{\rm B}$', r'$\Delta(\langle v_\phi\rangle / v_{\rm Kep})$')
    ax.set_title('Temporal variation amplitude vs radius', fontsize=12)
    plt.tight_layout(); save_fig(fig, 'vphi_variation_amplitude.png')

# ============================================================
# THETA-PROFILE PLOT FUNCTIONS  (3D-only)
# ============================================================

def plot_rho_theta_profile(rho_theta, x2v, label, i_r, x1v):
    txy = theta_xy_deg(x2v)
    fig, ax = plt.subplots()
    ax.plot(txy, rho_theta, 'b-', lw=1.8)
    style_ax(ax, r'$\theta_{\rm xy}$ [deg]', r'$\langle \rho \rangle_\phi \;/\; \rho_\infty$')
    ax.set_title(r'$r = %.4f \; R_{\rm B}$  --  %s' % (x1v[i_r], label), fontsize=11)
    plt.tight_layout()
    save_fig(fig, f'rho_theta_r{x1v[i_r]:.4f}_{label.replace(" ","_")}.png')

def plot_vphi_kep_theta_profile(vphi_theta, x2v, label, i_r, x1v, params,
                                j_north=None, j_south=None):
    gm    = params['code_units']['gm_code']
    v_kep = np.sqrt(gm / x1v[i_r])
    ratio = vphi_theta / v_kep
    txy   = theta_xy_deg(x2v)
    fig, ax = plt.subplots()
    ax.plot(txy, ratio, 'b-', lw=1.8)
    ax.axvline( 0.0, color='k', ls=':',  lw=0.8, alpha=0.5)
    if j_north is not None:
        ax.axvline(txy[j_north], color='r', ls='--', lw=1.0, alpha=0.8,
                   label=r'disk edge')
    if j_south is not None:
        ax.axvline(txy[j_south], color='r', ls='--', lw=1.0, alpha=0.8)
    if j_north is not None or j_south is not None:
        ax.legend(frameon=False, fontsize=10)
    style_ax(ax, r'$\theta_{\rm xy}$ [deg]',
             r'$\langle v_\phi \rangle_\phi \;/\; v_{\rm Kep}$')
    ax.set_title(r'$r = %.4f \; R_{\rm B}$  --  %s' % (x1v[i_r], label), fontsize=11)
    plt.tight_layout()
    save_fig(fig, f'vphi_kep_theta_r{x1v[i_r]:.4f}_{label.replace(" ","_")}.png')

def plot_beta_theta_profile(vphi_theta, x2v, label, i_r, x1v, params,
                            j_north=None, j_south=None):
    gm         = params['code_units']['gm_code']
    v_kep      = np.sqrt(gm / x1v[i_r])
    beta_theta = (vphi_theta / v_kep) ** 2
    beta_theta = np.where(np.isfinite(beta_theta) & (beta_theta > 0.0), beta_theta, np.nan)
    txy = theta_xy_deg(x2v)
    lim = safe_log_ylim(beta_theta)
    fig, ax = plt.subplots()
    ax.plot(txy, beta_theta, 'b-', lw=1.8)
    ax.axvline(0.0,           color='k', ls=':',  lw=0.8, alpha=0.5)
    if j_north is not None:
        ax.axvline(txy[j_north], color='r', ls='--', lw=1.0, alpha=0.8, label=r'disk edge')
    if j_south is not None:
        ax.axvline(txy[j_south], color='r', ls='--', lw=1.0, alpha=0.8)
    if j_north is not None or j_south is not None:
        ax.legend(frameon=False, fontsize=10)
    if lim is not None:
        ax.set_yscale('log'); ax.set_ylim(*lim)
    style_ax(ax, r'$\theta_{\rm xy}$ [deg]',
             r'$\beta = \langle v_\phi \rangle_\phi^2 \;/\; v_{\rm Kep}^2$')
    plt.tight_layout()
    save_fig(fig, f'beta_theta_r{x1v[i_r]:.4f}_{label.replace(" ","_")}.png')

def plot_rho_theta_multitime(rho_profiles, x2v, t_per, k_indices, i_r, x1v):
    txy = theta_xy_deg(x2v)
    colors  = ['k', 'g', 'b', 'r']
    lstyles = ['--', ':', '-', '-.']
    labels  = [r'$t = 0$',
               r'$t = %.3f \; P_{\rm orb}$' % t_per[k_indices[1]],
               r'$t = %.3f \; P_{\rm orb}$' % t_per[k_indices[2]],
               r'$t = %.3f \; P_{\rm orb}$' % t_per[k_indices[3]]]
    fig, ax = plt.subplots()
    for ii in range(len(k_indices)):
        ax.plot(txy, rho_profiles[ii], color=colors[ii], ls=lstyles[ii], label=labels[ii])
    ax.axvline(0.0, color='k', ls=':', lw=0.8, alpha=0.3)
    ax.legend(frameon=False)
    style_ax(ax, r'$\theta_{\rm xy}$ [deg]', r'$\langle \rho \rangle_\phi \;/\; \rho_\infty$')
    plt.tight_layout()
    save_fig(fig, f'rho_vs_theta_multitime_r{x1v[i_r]:.4f}.png')

def plot_vphi_kep_theta_multitime(vphi_profiles, x2v, t_per, k_indices,
                                   i_r, x1v, params, j_disk_n=None, j_disk_s=None):
    gm    = params['code_units']['gm_code']
    v_kep = np.sqrt(gm / x1v[i_r])
    txy   = theta_xy_deg(x2v)
    colors  = ['m', 'g', 'r', 'k']
    lstyles = [':', '--', '-.', '-']
    labels  = [r'$t = 0$',
               r'$t = %.3f \; P_{\rm orb}$' % t_per[k_indices[1]],
               r'$t = %.3f \; P_{\rm orb}$' % t_per[k_indices[2]],
               r'$t = %.3f \; P_{\rm orb}$' % t_per[k_indices[3]]]
    ratios = vphi_profiles / v_kep
    fig, ax = plt.subplots()
    for ii in range(len(k_indices)):
        ax.plot(txy, ratios[ii], color=colors[ii], ls=lstyles[ii], label=labels[ii])
    ax.axvline(0.0, color='k', ls=':', lw=0.8, alpha=0.3)
    vphi_thr = - np.sqrt(BETA_THRESHOLD)
    ax.axhline(vphi_thr, color='r', ls='--', lw=0.8, alpha=0.5)
    if j_disk_n is not None:
        ax.axvline(txy[j_disk_n], color=colors[3], ls='--', lw=1.0, alpha=0.8)
    if j_disk_s is not None:
        ax.axvline(txy[j_disk_s], color=colors[3], ls='--', lw=1.0, alpha=0.8)
    ax.legend(frameon=False)
    style_ax(ax, r'$\theta_{\rm xy}$ [deg]',
             r'$\langle v_\phi \rangle_\phi \;/\; v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig, f'vphi_vs_theta_multitime_r{x1v[i_r]:.4f}.png')

# ============================================================
# HEIGHT RADIAL PROFILE PLOTS  (3D-only)
# ============================================================

def _xlim_extended(x1v, ir_outer, n_extra=5):
    return x1v[0], x1v[min(ir_outer + n_extra, len(x1v) - 1)]

def plot_H_vs_r(H_N, H_S, x1v, ir_outer, R_disk,
                spl=None, r_pts=None, H_pts=None):
    """
    Raw data as empty circles, smoothing spline as solid curve,
    R_disk as a single dotted vertical line.  No legend.
    """
    fig, ax = plt.subplots()
    if spl is not None and r_pts is not None:
        ax.plot(r_pts, H_pts, 'o', color='steelblue',
                ms=4.0, mfc='none', mew=0.8, lw=0, zorder=3)
        log_r_fine = np.linspace(np.log(r_pts[0]), np.log(r_pts[-1]), 800)
        ax.plot(np.exp(log_r_fine), spl(log_r_fine),
                '-', color='steelblue', lw=1.8, zorder=2)
    else:
        ax.plot(x1v, 0.5 * (H_N + H_S), 'b-', lw=1.8)
    if R_disk is not None:
        ax.axvline(R_disk, color='k', ls=':', lw=1.0)
    ax.set_xscale('log')
    ax.set_xlim(*_xlim_extended(x1v, ir_outer))
    style_ax(ax, r'$r / R_{\rm B}$', r'$H / R_{\rm B}$')
    plt.tight_layout()
    save_fig(fig, 'H_vs_r.png')

def plot_H_over_r_vs_r(H_N, H_S, x1v, ir_outer, R_disk):
    """ir_outer sets xlim (R_outer + 5 cells); R_disk dotted line."""
    H_over_r_N = H_N / x1v
    fig, ax = plt.subplots()
    ax.plot(x1v, H_over_r_N, 'b-', lw=1.8)
    if R_disk is not None:
        ax.axvline(R_disk, color='k', ls=':', lw=1.0)
    ax.set_xscale('log'); ax.set_xlim(*_xlim_extended(x1v, ir_outer))
    style_ax(ax, r'$r / R_{\rm B}$', r'$H / r$')
    plt.tight_layout(); save_fig(fig, 'H_over_r_vs_r.png')

# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log = _Tee(OUTPUT_DIR / "run_log.txt")

    params     = read_parameters(SIM_DATA_DIR)
    prim_files = read_snapshot_paths(SIM_DATA_DIR)
    k_final    = FINAL_SNAP_IDX if FINAL_SNAP_IDX is not None else len(prim_files) - 1
    prim_files = prim_files[:k_final + 1]

    snap0 = ar.athdf(str(prim_files[0]), quantities=['vel1'])
    x1v   = snap0['x1v'];  x2v = snap0['x2v']
    del snap0; gc.collect()

    j_eq                             = find_equatorial_index(x2v)
    idx_inner, idx_median, idx_outer = gen_radial_ind(x1v)
    gm                               = params['code_units']['gm_code']

    # equatorial timeseries
    t_per, vphi_avg, vr_avg, beta_prof, v_kep, j_eq = \
        extract_timeseries(prim_files, x1v, x2v, params)
    vphi_avg_kep = vphi_avg / v_kep
    v_ff         = np.sqrt(2.0 * gm / x1v)
    vr_avg_ff    = vr_avg / v_ff[np.newaxis, :]

    # -- R_outer and tau_outer: always computed --------------------------------
    tau_outer, R_outer, ir_outer, _ = scan_threshold_crossings(
        vphi_avg_kep, vr_avg_ff, x1v, t_per,
        idx_inner, idx_outer, BETA_THRESHOLD, VR_THRESHOLD, k_final)

    # -- tau_bump: always attempted -------------------------------------------
    tau_bump, k_bump = detect_bump_initiation(vphi_avg_kep, t_per, idx_inner, BETA_THRESHOLD)

    # -- H(r) profile, R_disk, tau_Hmax: computed whenever R_outer is found ---
    R_disk    = None;  ir_disk   = None
    tau_Hmax  = None;  k_Hmax    = None
    H_max_val = 0.0;   disk_formed = False;  theta_half = 0.0
    H_spl     = None;  H_r_pts   = None;  H_H_pts = None
    H_N_r     = None;  H_S_r     = None
    rho_all_final = vphi_all_final = None

    if ir_outer is not None:
        # load final snapshot: phi-averaged rho and vel3 over all theta
        snap_f         = ar.athdf(str(prim_files[k_final]), quantities=['rho', 'vel3'])
        rho_all_final  = phi_avg_all(snap_f, 'rho')    # (n_theta, n_r)
        vphi_all_final = phi_avg_all(snap_f, 'vel3')   # (n_theta, n_r)
        del snap_f; gc.collect()

        # scale height profile H_N(r), H_S(r) up to R_outer
        H_N_r, H_S_r = compute_H_profile(
            vphi_all_final, x1v, x2v, j_eq, ir_outer, params, BETA_THRESHOLD)

        # spline H maximum -> R_disk, ir_disk, tau_Hmax, k_Hmax
        R_disk, ir_disk, H_max_val, theta_half, tau_Hmax, k_Hmax, \
            H_spl, H_r_pts, H_H_pts = detect_disk_radius_Hmax(
                H_N_r, H_S_r, x1v, vphi_avg_kep, vr_avg_ff,
                t_per, k_final, BETA_THRESHOLD, VR_THRESHOLD)
        disk_formed = H_max_val > 0.0

    # timescales
    T_orb         = orbital_period(params)
    T_orb_yr      = T_orb * params['unit_system_cgs']['time_unit_yr']
    T_orb_Myr     = T_orb_yr / 1.0e6
    Omega_code    = params['code_units']['Omega_code']
    gm_code       = params['code_units']['gm_code']
    R_circ_code   = Omega_code**2 / gm_code
    # tau_ff: free-fall time at R_B = sqrt(R_B^3 / G*m_BH), R_B=1 in code units
    tau_ff_code   = np.sqrt(1.0 / gm_code)
    tau_ff_yr     = tau_ff_code * params['unit_system_cgs']['time_unit_yr']
    tau_ff_Porb   = tau_ff_code / T_orb
    # tau_prop: free-fall time at R_circ = sqrt(R_circ^3 / G*m_BH)
    tau_prop_code = np.sqrt(R_circ_code**3 / gm_code)
    tau_prop_yr   = tau_prop_code * params['unit_system_cgs']['time_unit_yr']
    tau_prop_Porb = tau_prop_code / T_orb
    length_unit_pc = (params['unit_system_cgs']['length_unit_cgs']
                      / params['physical_constants_cgs']['pc_cgs'])
    R_circ_pc     = R_circ_code * length_unit_pc

    print(f"  Final-time profiles use snapshot k={k_final}"
          f"  (t = {t_per[k_final]:.4e} P_orb)")

    print_summary(params, prim_files, t_per, j_eq, x2v,
                  tau_bump, tau_Hmax, R_disk, R_outer, tau_outer,
                  T_orb_Myr, T_orb_yr,
                  R_circ_code, R_circ_pc,
                  tau_ff_Porb, tau_ff_yr,
                  tau_prop_Porb, tau_prop_yr)

    # beta diagnostics
    if tau_bump is not None:
        beta_at_bump = float(np.nan_to_num(beta_prof[k_bump, idx_inner]))
        print(f"  beta at inner boundary at tau_bump (k={k_bump}) = {beta_at_bump:.4f}")
    if tau_Hmax is not None:
        beta_at_disk = float(vphi_avg_kep[k_Hmax, ir_disk] ** 2)
        print(f"  beta at R_disk at tau_Hmax (k={k_Hmax})          = {beta_at_disk:.4f}")
    if R_disk is not None:
        beta_Rdisk_final = float(vphi_avg_kep[k_final, ir_disk] ** 2)
        print(f"  beta at R_disk at final time (ir={ir_disk})      = {beta_Rdisk_final:.4f}")
    print()

    # ================================================================
    # EQUATORIAL PLOTS
    # ================================================================
    plot_vphi_over_vkep(t_per, vphi_avg, x1v, idx_inner, params, ir_disk, k_bump)
    plot_vr(t_per, vr_avg, x1v, idx_inner, tau_bump)
    plot_beta_timeseries(t_per, beta_prof, x1v, tau_bump,
                         idx_inner, idx_median, idx_outer)
    plot_beta_profile(t_per, beta_prof, x1v, k_final, 'final time', R_disk)
    plot_vphi_kep_profile(t_per, vphi_avg_kep, x1v, k_final, 'final time', R_disk)
    plot_vr_freefall_profile(t_per, vr_avg, x1v, params, k_final, 'final time', R_disk)
    plot_vphi_radial_envelope(t_per, vphi_avg, x1v, params)

    if k_bump is not None and ir_disk is not None:
        plot_profile_disk(t_per, vphi_avg_kep, vr_avg, x1v,
                          tau_bump, R_disk, ir_disk, k_bump)
        plot_beta_overlay(t_per, beta_prof, x1v, idx_inner,
                          ir_disk, R_disk, tau_bump)
        plot_vphi_kep_overlay(t_per, vphi_avg_kep, x1v, idx_inner,
                               ir_disk, R_disk, tau_bump)
        plot_vr_ff_overlay(t_per, vr_avg_ff, x1v, idx_inner,
                           ir_disk, tau_bump)
        plot_vphi_kep_multitime(t_per, vphi_avg_kep, x1v, k_bump, R_disk, k_final)
        plot_vr_ff_multitime(t_per, vr_avg_ff, x1v, k_bump, R_disk, k_final)

    # ================================================================
    # H AND THETA-PROFILE PLOTS  (3D-only)
    # executed when R_outer is found and the final snapshot has been loaded
    # ================================================================
    if vphi_all_final is not None:

        # inner boundary -- final time
        rho_inner_final  = rho_all_final[:, idx_inner]
        vphi_inner_final = vphi_all_final[:, idx_inner]
        plot_rho_theta_profile(rho_inner_final, x2v, 'final time', idx_inner, x1v)
        plot_vphi_kep_theta_profile(vphi_inner_final, x2v, 'final time', idx_inner, x1v, params)
        plot_beta_theta_profile(vphi_inner_final, x2v, 'final time', idx_inner, x1v, params)

        if disk_formed:
            # theta profiles at R_disk -- final time
            rho_disk_final  = rho_all_final[:, ir_disk]
            vphi_disk_final = vphi_all_final[:, ir_disk]

            # disk geometry at the scale height maximum
            print(f"\n  Disk geometry (spline-derived):")
            print(f"    R_disk     = {R_disk:.4e} R_B  (ir_disk = {ir_disk})")
            print(f"    H_max      = {H_max_val:.4e} R_B")
            print(f"    theta_half = {theta_half:.2f} deg"
                  f"   H_max/R_disk = {H_max_val/R_disk:.4f}")

            # H-profile maxima from data
            ir_max_HN = int(np.argmax(H_N_r))
            ir_max_HS = int(np.argmax(H_S_r))
            print(f"\n  Height maxima over r in [r_inner, R_outer]:")
            print(f"    North : H_max = {H_N_r[ir_max_HN]:.4e} R_B"
                  f"   at r = {x1v[ir_max_HN]:.4e} R_B"
                  f"   H/r = {H_N_r[ir_max_HN]/x1v[ir_max_HN]:.4f}")
            print(f"    South : H_max = {H_S_r[ir_max_HS]:.4e} R_B"
                  f"   at r = {x1v[ir_max_HS]:.4e} R_B"
                  f"   H/r = {H_S_r[ir_max_HS]/x1v[ir_max_HS]:.4f}")

            plot_H_vs_r(H_N_r, H_S_r, x1v, ir_outer, R_disk, H_spl, H_r_pts, H_H_pts)
            plot_H_over_r_vs_r(H_N_r, H_S_r, x1v, ir_outer, R_disk)

            plot_rho_theta_profile(rho_disk_final, x2v, 'final time at disk r', ir_disk, x1v)
            plot_vphi_kep_theta_profile(vphi_disk_final, x2v, 'final time at disk r',
                                        ir_disk, x1v, params)
            plot_beta_theta_profile(vphi_disk_final, x2v, 'final time at disk r',
                                    ir_disk, x1v, params)

        if k_bump is not None and ir_disk is not None:
            # theta profiles at R_disk -- tau_bump
            snap_disk      = ar.athdf(str(prim_files[k_bump]), quantities=['rho', 'vel3'])
            rho_disk_form  = phi_avg_theta_profile(snap_disk, 'rho',  ir_disk)
            vphi_disk_form = phi_avg_theta_profile(snap_disk, 'vel3', ir_disk)
            del snap_disk; gc.collect()

            plot_rho_theta_profile(rho_disk_form, x2v, 'disk formation', ir_disk, x1v)
            plot_vphi_kep_theta_profile(vphi_disk_form, x2v, 'disk formation',
                                        ir_disk, x1v, params)
            plot_beta_theta_profile(vphi_disk_form, x2v, 'disk formation',
                                    ir_disk, x1v, params)

            # theta profiles at R_disk at four representative times
            k_half    = np.argmin(np.abs(t_per - t_per[k_bump] / 2.0))
            k_indices = [0, k_half, k_bump, k_final]
            th_prof   = extract_theta_profiles(prim_files, x2v, ir_disk,
                                               k_indices, ['rho', 'vel3'])

            # -- theta disk edge at final -----------------------------------
            txy      = theta_xy_deg(x2v)
            j_n, j_s = _find_theta_edges(
                th_prof['vel3'][3], x2v, j_eq,
                x1v[ir_disk], params, BETA_THRESHOLD)
            hr_N = np.tan(np.pi/2 - x2v[j_n])   if j_n is not None else None
            hr_S = np.tan(x2v[j_s] - np.pi/2)   if j_s is not None else None
            print(f"\n  Disk opening at r = {x1v[ir_disk]:.4e} R_B"
                  f"  (final time  t = {t_per[k_indices[3]]:.4f} P_orb):")
            if hr_N is not None:
                print(f"    North : theta_edge = {txy[j_n]:.2f} deg"
                      f"   H/R = {hr_N:.4f}")
            else:
                print(f"    North : no threshold crossing found")
            if hr_S is not None:
                print(f"    South : theta_edge = {txy[j_s]:.2f} deg"
                      f"   H/R = {hr_S:.4f}")
            else:
                print(f"    South : no threshold crossing found")
            if hr_N is not None and hr_S is not None:
                print(f"    Mean  : H/R = {0.5*(hr_N + hr_S):.4f}")

            plot_rho_theta_multitime(th_prof['rho'],  x2v, t_per, k_indices, ir_disk, x1v)
            plot_vphi_kep_theta_multitime(th_prof['vel3'], x2v, t_per,
                                          k_indices, ir_disk, x1v, params, j_n, j_s)

    print(f"Plots written to: {OUTPUT_DIR}\n")
    _log.close()


if __name__ == "__main__":
    main()
