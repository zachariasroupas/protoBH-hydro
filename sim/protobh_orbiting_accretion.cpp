//========================================================================================
// Accretion-disk formation around orbiting stellar holes inside gaseous star clusters
// Part of the project: "protoBH" — MSCA grant agreement No. 101149270 funded by the E.U.
// Author: Zacharias Roupas
// Please cite: 
// Roupas, Z., to appear (2026)
// Code DOI:    https://doi.org/10.5281/zenodo.20557844
//
// Built on Athena++ v21.0 (Stone et al. 2020, ApJS 249, 4)
//========================================================================================
//! \file protobh_orbiting_accretion.cpp
//! \brief Orbiting BH accretion - simple harmonic elliptical orbit.
//!        Uniform ambient gas density. 
//!
//! Orbit (isotropic harmonic oscillator in uniform-density star cluster core):
//!   Omega  = sqrt(4 pi G rho_cluster / 3)          
//!   A      = B / sqrt(1 - ecc^2)               [semi-major axis]
//!   X(t)   = A cos(Omega t + xi0)
//!   Y(t)   = B sin(Omega t + xi0)
//!   ell_BH      = Omega * A * B                [conserved BH angular momentum]
//!   xi0 = atan2(A sin(psi0), B cos(psi0))      [atan2(Y_0, X_0)]
//!
//! Gas velocity in BH frame:
//!   vr     = - (V_BH_R  * cos(phi) + V_BH_psi * sin(phi)) * sin(theta);
//!   vtheta = - (V_BH_R  * cos(phi) + V_BH_psi * sin(phi)) * cos(theta);
//!   vphi   = - omega_BH * r * sin(theta) + V_BH_R  * sin(phi) - V_BH_psi * cos(phi);
//!
//! Source accelerations in BH frame:
//!   a_r     = - G * m_BH/r^2 - Omega^2 * r + omega_BH^2 * r * sin(theta)^2 + 2.0 * omega_BH * v_phi * sin(theta);
//!   a_theta = omega_BH^2 * r * sin(theta) * cos(theta) + 2 * omega_BH * v_phi * cos(theta);
//!   a_phi   = - 2.0 * omega_BH * (v_r * sin(theta) + v_theta * cos(theta)) - dotomega * r * sin(theta);
//========================================================================================

// Disclaimer: This software is provided "as is", without warranty of any kind.
// The author accepts no liability for any errors or consequences of its use.
// Users are responsible for verifying results against the cited publication.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"

//========================================================================================
// Physical constants (CGS)
//========================================================================================
static const Real G_cgs     = 6.67430e-8;
static const Real M_sun_cgs = 1.98847e33;
static const Real k_B_cgs   = 1.38065e-16;
static const Real m_p_cgs   = 1.6726e-24;
static const Real pc_cgs    = 3.08567758e18;
static const Real yr_cgs    = 3.15576e7;
static const Real au_cgs    = 1.495978707e13;

//========================================================================================
// Global parameters (code units, set once in InitUserMeshData)
//========================================================================================
static Real gm_code;          // G * M_BH  [c_s^2 * R_Bondi_max]
static Real gamma_gas;        // adiabatic index
static Real B_code;           // semi-minor axis       [R_Bondi_max]
static Real A_code;           // semi-major axis       [R_Bondi_max]
static Real Omega_code;       // orbital frequency     [c_s / R_Bondi_max]
static Real ell_BH_code;      // BH angular momentum ell_BH = Omega*A*B  [R_Bondi_max * c_s]
static Real xi0;              // initial phase [rad], derived from psi0
static bool NonInert;         // true for allowing non-inertial acceleration
static Real nu_iso_code;      // kinematic shear viscosity [code units]

//========================================================================================
// Code units
//   Length : R_Bondi_max = 2 G m_BH / (c_s^2 * (1 + Mach_min^2))
//             where Mach_min = (ell / A) / c_s is the BH speed at apoastron
//   Velocity: c_s = sqrt(gamma k_B T_gas / (mu m_p))
//   Time    : R_Bondi_max / c_s
//   Density : rho_gas = (1 - epsilon) * rho_cluster
//   Pressure: rho_gas * c_s^2
//
//   Derived code-unit values:
//   gm_code     = G m_BH / (c_s^2 R_Bondi_max) = 0.5 * (1 + Mach_min^2)
//   Omega_code  = Omega * R_Bondi_max / c_s
//   ell_BH_code = Omega_code * A_code * B_code
//   R_circ_code = Omega_code^2 / gm_code  [= Omega^2 R_B^4 / (G m_BH) in phys. units]
//
//   Equation of state: P = (gamma - 1) * e,  ideal gas with gamma = 5/3
//   Initial conditions: rho = 1, P = 1/gamma  (uniform; c_s^2 = 1 by construction)
//========================================================================================

//========================================================================================
// Helper functions declarations
//========================================================================================
struct BHOrbitState {
    Real R;         // distance from cluster center   [R_Bondi_max]
    Real V_R;       // radial velocity (inertial)     [c_s]
    Real V_psi;     // tangential velocity (inertial) [c_s]
    Real omega;     // frame angular velocity         [c_s / R_Bondi_max]
    Real omegadot;  // d(omega)/dt                    [c_s^2 / R_Bondi_max^2]
};

BHOrbitState ComputeBHOrbitState(Real t);

void ComputeGasVelocity(Real r, Real theta, Real phi, const BHOrbitState &orb, Real &vr, Real &vtheta, Real &vphi);

Real ComputeXi0(Real psi0);

std::string GetSimulationDirectory(Real M_BH_Msun, Real B_pc, Real ecc_val, Real psi0_rad, int nx1, int nx2, int nx3, 
                                   Real rho_cluster, Real nu_iso_code, Real R_inner_code, Real R_outer_code, Real t_end,
                                   const std::string &base_path = "./data/3D");

std::string SetupOutputDirectory(ParameterInput *pin,
                                Real M_BH_Msun, Real B_pc, Real ecc_val, Real psi0_rad, int nx1, int nx2, int nx3,
                                Real rho_cluster, Real nu_iso_code, Real R_inner_code, Real R_outer_code, Real t_end,
                                const std::string &out_dir_override = "");

void WriteSimulationParameters(const std::string& dir, Real M_BH_Msun, Real B_pc, Real ecc_val, Real psi0_rad, Real R_inner_code, 
                                Real R_outer_code, Real R_circ_code, Real Dt_inflow_yr, int nx1, int nx2, int nx3,
                               Real rho_cluster_cgs, Real rho_gas_cgs, Real epsilon, Real T_gas_K, Real mu, 
                               Real c_s_cgs, Real Omega_cgs, Real time_unit_cgs, Real length_unit_cgs, Real gm_code_val, 
                               Real nu_iso_code, Real xi0_val, Real t_end);

//========================================================================================
// User function declarations
//========================================================================================
void OuterX1_User(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void SourceFunction(MeshBlock *pmb, const Real time, const Real dt,
                    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                    AthenaArray<Real> &cons_scalar);


//========================================================================================
// Mesh::InitUserMeshData
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {

    // ---------- Read inputs ----------
    const Real M_BH_Msun   = pin->GetReal("problem", "M_BH");
    const Real B_pc        = pin->GetReal("problem", "B");
    const Real ecc_in      = pin->GetReal("problem", "ecc");
    const Real psi0_rad    = pin->GetReal("problem", "psi_0");
    const Real rho_cluster = pin->GetReal("problem", "rho_cluster");
    const Real epsilon     = pin->GetReal("problem", "epsilon");
    const Real T_gas_K     = pin->GetReal("problem", "T_gas");
    const int nx1          = pin->GetInteger("mesh", "nx1");
    const int nx2          = pin->GetInteger("mesh", "nx2");
    const int nx3          = pin->GetInteger("mesh", "nx3");
    NonInert               = pin->GetBoolean("problem",   "NonInert");
    nu_iso_code            = pin->GetOrAddReal("problem", "nu_iso", 0.0);

    const Real t_end       = pin->GetReal("time",    "tlim");

    const Real mu          = pin->GetReal("hydro",   "mu");
    gamma_gas              = pin->GetReal("hydro",   "gamma");

    const std::string out_dir_override = pin->GetOrAddString("problem", "output_dir", "");

    const Real R_inner_code = mesh_size.x1min;
    const Real R_outer_code = mesh_size.x1max;
    
    // ---------- Derived quantities ----------
    const Real rho_cluster_cgs = rho_cluster * M_sun_cgs / (pc_cgs * pc_cgs * pc_cgs);
    const Real c_s_cgs   = std::sqrt(gamma_gas * k_B_cgs * T_gas_K / (mu * m_p_cgs));
    const Real rho_gas_cgs = (1.0 - epsilon) * rho_cluster_cgs; 
    const Real Omega_cgs   = std::sqrt(4.0 * M_PI * G_cgs * rho_cluster_cgs / 3.0); 
    const Real A_pc      = B_pc / std::sqrt(1.0 - ecc_in * ecc_in);
    const Real v_BH_min_kms = Omega_cgs * B_pc * pc_cgs * 1e-5; // BH velocity at apoastron [km/s]
    const Real v_BH_max_kms = Omega_cgs * A_pc * pc_cgs * 1e-5; // BH velocity at periastron [km/s]
    const Real Mach_BH_min = v_BH_min_kms / (c_s_cgs * 1e-5);
    const Real Mach_BH_max = v_BH_max_kms / (c_s_cgs * 1e-5);
    const Real R_Bondi_min_cgs = 2.0 * (G_cgs * M_BH_Msun * M_sun_cgs / (c_s_cgs * c_s_cgs)) / (1 + Mach_BH_max * Mach_BH_max);
    const Real R_Bondi_max_cgs = 2.0 * (G_cgs * M_BH_Msun * M_sun_cgs / (c_s_cgs * c_s_cgs)) / (1 + Mach_BH_min * Mach_BH_min);
        
    const Real Dt_inflow_yr = std::sqrt( R_Bondi_max_cgs * R_Bondi_max_cgs * R_Bondi_max_cgs 
                                    / (2.0 * G_cgs * M_BH_Msun * M_sun_cgs)) / yr_cgs;

    // ---------- Scales ----------
    const Real length_unit_cgs = R_Bondi_max_cgs; // length unit [cm]
    const Real time_unit_cgs   = length_unit_cgs / c_s_cgs;   // time unit [s]


    // ---------- Code-unit orbital parameters ----------
    gm_code    = 0.5 * (1 + Mach_BH_min * Mach_BH_min);    // G*m_BH / (L * V^2)

    A_code      = A_pc * pc_cgs / length_unit_cgs;
    B_code      = B_pc * pc_cgs / length_unit_cgs;
    Omega_code  = Omega_cgs * time_unit_cgs;
    ell_BH_code = Omega_code * A_code * B_code;

    const Real R_circ_code = Omega_code * Omega_code / gm_code;

    // Initial phase from inertial angle psi0 
    xi0 = ComputeXi0(psi0_rad);

    // ---------- Setup output ----------
    std::string out_dir = SetupOutputDirectory(pin, M_BH_Msun, B_pc, ecc_in, psi0_rad, nx1, nx2, nx3, rho_cluster,
                                                nu_iso_code, R_inner_code, R_outer_code, t_end, out_dir_override);
    WriteSimulationParameters(out_dir, M_BH_Msun, B_pc, ecc_in, psi0_rad, R_inner_code, R_outer_code, 
                              R_circ_code, Dt_inflow_yr, nx1, nx2, nx3, rho_cluster_cgs, rho_gas_cgs, epsilon, T_gas_K, mu, 
                              c_s_cgs, Omega_cgs, time_unit_cgs, length_unit_cgs, gm_code, nu_iso_code, xi0, t_end);

    // ---------- Console summary ----------
    if (Globals::my_rank == 0) {
        std::cout << "========================================\n";
        std::cout << "  Orbiting BH  --  Simple Harmonic Orbit\n";
        std::cout << "========================================\n";
        std::cout << "Physical inputs:\n";
        std::cout << "  M_BH        = " << M_BH_Msun              << " M_sun\n";
        std::cout << "  B           = " << B_pc                   << " pc\n";
        std::cout << "  ecc         = " << ecc_in                 << "\n";
        std::cout << "  psi_0       = " << psi0_rad * 180.0 / M_PI << " deg\n";
        std::cout << "  rho_cluster = " << rho_cluster             << " M_sol/pc^3\n";
        std::cout << "  rho_gas     = " << rho_gas_cgs             << " g/cm^3\n";        
        std::cout << "  T_gas       = " << T_gas_K                 << " K\n";
        std::cout << "  epsilon     = " << epsilon                 << "\n";
        std::cout << "Derived quantities:\n";
        std::cout << "  A           = " << A_pc                   << " pc\n";
        std::cout << "  c_s         = " << c_s_cgs * 1.0e-5       << " km/s\n";
        std::cout << "  v_BH_min    = " << v_BH_min_kms           << " km/s\n";
        std::cout << "  v_BH_max    = " << v_BH_max_kms           << " km/s\n";
        std::cout << "  R_Bondi_min     = " << R_Bondi_min_cgs / au_cgs   << " AU"
                  << "  = "            << R_Bondi_min_cgs / pc_cgs   << " pc\n";
        std::cout << "  R_Bondi_max     = " << R_Bondi_max_cgs / au_cgs   << " AU"
                  << "  = "            << R_Bondi_max_cgs / pc_cgs   << " pc\n";
        std::cout << "  Omega        = " << Omega_cgs*1.0e6*yr_cgs << " rad/Myr\n";
        std::cout << "  R_circ_code  = " << R_circ_code          << " [R_B]\n";
        std::cout << "  Dt_inflow    = " << Dt_inflow_yr         << " yr\n";
        std::cout << "Unit system:\n";
        std::cout << "  length_unit = " << length_unit_cgs / au_cgs   << " AU"
                  << "  = "            << length_unit_cgs / pc_cgs    << " pc\n";
        std::cout << "  time_unit      = " << time_unit_cgs / yr_cgs      << " yr\n";
        std::cout << "Code units:\n";
        std::cout << "  nu_iso      = " << nu_iso_code            << "\n";
        std::cout << "  gm_code     = " << gm_code                << "\n";
        std::cout << "  B_code      = " << B_code                 << "\n";
        std::cout << "  A_code      = " << A_code                 << "\n";
        std::cout << "  Omega_code  = " << Omega_code             << "\n";
        std::cout << "  ell_code    = " << ell_BH_code            << "\n";
        std::cout << "  xi0         = " << xi0                    << " rad\n";
        std::cout << "========================================\n";
    }

    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1_User);
    EnrollUserExplicitSourceFunction(SourceFunction);
}

//========================================================================================
// MeshBlock::ProblemGenerator
//   Uniform density, velocity = far-field BH-frame value at t=0
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    const BHOrbitState orb = ComputeBHOrbitState(0.0);

    for (int k = ks; k <= ke; ++k) {
        const Real phi = pcoord->x3v(k);
        for (int j = js; j <= je; ++j) {
            const Real theta = pcoord->x2v(j);
            for (int i = is; i <= ie; ++i) {
                const Real r = pcoord->x1v(i);

                Real vr, vtheta, vphi;
                ComputeGasVelocity(r, theta, phi, orb, vr, vtheta, vphi);

                const Real rho = 1.0;
                const Real P   = rho / gamma_gas;

                phydro->u(IDN, k, j, i) = rho;
                phydro->u(IM1, k, j, i) = rho * vr;
                phydro->u(IM2, k, j, i) = rho * vtheta;
                phydro->u(IM3, k, j, i) = rho * vphi;
                const Real KE = 0.5 * rho * (vr*vr + vtheta*vtheta + vphi*vphi);
                phydro->u(IEN, k, j, i) = P / (gamma_gas - 1.0) + KE;
            }
        }
    }
}

//========================================================================================
// OuterX1_User: time-dependent far-field inflow  
//========================================================================================
void OuterX1_User(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
    const BHOrbitState orb = ComputeBHOrbitState(time);

    for (int k = kl; k <= ku; ++k) {
        const Real phi = pco->x3v(k);
        for (int j = jl; j <= ju; ++j) {
            const Real theta = pco->x2v(j);
            for (int i = 1; i <= ngh; ++i) {
                const int  ig  = iu + i;
                const Real r_g = pco->x1v(ig);

                Real vr, vtheta, vphi;
                ComputeGasVelocity(r_g, theta, phi, orb, vr, vtheta, vphi);

                prim(IDN, k, j, ig) = 1.0;
                prim(IPR, k, j, ig) = 1.0 / gamma_gas;
                prim(IVX, k, j, ig) = vr;
                prim(IVY, k, j, ig) = vtheta;
                prim(IVZ, k, j, ig) = vphi;
            }
        }
    }
}

//========================================================================================
// SourceFunction: tidal + centrifugal + Coriolis + BH gravity 
//========================================================================================
void SourceFunction(MeshBlock *pmb, const Real time, const Real dt,
                    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                    AthenaArray<Real> &cons_scalar) {
    const BHOrbitState orb = ComputeBHOrbitState(time);

    const Real Omega2 = Omega_code * Omega_code;
    const Real omega2 = orb.omega  * orb.omega;

    for (int k = pmb->ks; k <= pmb->ke; ++k) {
        for (int j = pmb->js; j <= pmb->je; ++j) {
            const Real theta = pmb->pcoord->x2v(j);
            const Real st = std::sin(theta);
            const Real ct = std::cos(theta);
            for (int i = pmb->is; i <= pmb->ie; ++i) {
                const Real r   = pmb->pcoord->x1v(i);
                const Real rho = prim(IDN, k, j, i);
                const Real vr  = prim(IVX, k, j, i);
                const Real vth = prim(IVY, k, j, i);
                const Real vph = prim(IVZ, k, j, i);

                const Real a_r_G = - gm_code / (r * r);
                const Real a_r_I   = - Omega2 * r + omega2 * r * st * st + 2.0 * orb.omega * vph * st;
                const Real a_theta_I = orb.omega * orb.omega * r * st * ct + 2 * orb.omega * vph * ct;
                const Real a_phi_I = - 2.0 * orb.omega * (vr * st + vth * ct) - orb.omegadot * r * st;

                Real a_r = a_r_G;
                Real a_theta = 0.0;
                Real a_phi = 0.0;
                if (NonInert) {
                    a_r = a_r + a_r_I;
                    a_theta = a_theta_I;
                    a_phi = a_phi_I;
                }

                cons(IM1, k, j, i) += dt * rho * a_r;
                cons(IM2, k, j, i) += dt * rho * a_theta;
                cons(IM3, k, j, i) += dt * rho * a_phi;
                cons(IEN, k, j, i) += dt * rho * (vr * a_r + vth * a_theta + vph * a_phi);
            }
        }
    }
}

//========================================================================================
//==================================== HELPERS ===========================================

//========================================================================================
// Helper: BH orbital state at time t  
//========================================================================================
BHOrbitState ComputeBHOrbitState(Real t) {
    BHOrbitState s;

    const Real phase = Omega_code * t + xi0;
    const Real X     = A_code * std::cos(phase);
    const Real Y     = B_code * std::sin(phase);
    s.R = std::sqrt(X*X + Y*Y);

    s.V_psi    = ell_BH_code / s.R;                                         

    s.V_R     = Omega_code * (B_code*B_code - A_code*A_code)            
                * std::sin(2.0 * phase) / (2.0 * s.R);

    s.omega    = ell_BH_code / (s.R * s.R);                                 

    s.omegadot = -2.0 * ell_BH_code * s.V_R / (s.R * s.R * s.R);          

    return s;
}

//========================================================================================
// Helper: gas velocity in BH frame at (r, theta, phi)  
//========================================================================================
void ComputeGasVelocity(Real r, Real theta, Real phi, const BHOrbitState &orb,
                        Real &vr, Real &vtheta, Real &vphi) {
    const Real st = std::sin(theta);
    const Real ct = std::cos(theta);
    const Real sp = std::sin(phi);
    const Real cp = std::cos(phi);

    vr   = - (orb.V_R  * cp + orb.V_psi * sp) * st;

    vtheta = - (orb.V_R  * cp + orb.V_psi * sp) * ct;

    vphi = - orb.omega * r * st + orb.V_R  * sp - orb.V_psi * cp;
}

//========================================================================================
// Helper: initial phase xi0 from inertial angle psi0
//   X0 = A cos(xi0) = R0 cos(psi0)
//   Y0 = B sin(xi0) = R0 sin(psi0)
//========================================================================================
Real ComputeXi0(Real psi0) {
    const Real xi0 = std::atan2(A_code * std::sin(psi0), B_code * std::cos(psi0));
    return xi0;
}

//========================================================================================
// Helper: output directory path
//========================================================================================
std::string GetSimulationDirectory(Real M_BH_Msun, Real B_pc, Real ecc_val, Real psi0_rad, 
                                    int nx1, int nx2, int nx3, Real rho_cluster, Real nu_iso_code,
                                    Real R_inner_code, Real R_outer_code, Real t_end,
                                    const std::string &base_path) {
    char subdir[256];
    std::snprintf(subdir, sizeof(subdir),
                  "mBH%.1f_B%.3f_ecc%.2f_psi0%.2f_nR%d_nT%d_nP%d_rho%.1e/nu%.0e_NonI%d_Rinn%.1e_Rout%.2f_tend%.2f",
                  M_BH_Msun, B_pc, ecc_val, psi0_rad, nx1, nx2, nx3, rho_cluster,
                  nu_iso_code, NonInert, R_inner_code, R_outer_code, t_end);
    return base_path + "/" + std::string(subdir);
}

//========================================================================================
// Helper: create output directory and set problem_id
//========================================================================================
std::string SetupOutputDirectory(ParameterInput *pin, Real M_BH_Msun, Real B_pc, Real ecc_val, Real psi0_rad, 
                                    int nx1, int nx2, int nx3, Real rho_cluster, Real nu_iso_code,
                                    Real R_inner_code, Real R_outer_code, Real t_end,
                                    const std::string &out_dir_override) {
    // expand leading ~ to home directory
    std::string dir_expanded = out_dir_override;
    if (!dir_expanded.empty() && dir_expanded[0] == '~') {
        const char *home = std::getenv("HOME");
        if (home) {
            dir_expanded = std::string(home) + dir_expanded.substr(1);
        } else {
            std::cerr << "Warning: HOME not set, cannot expand ~ in output_dir\n";
        }
    }
    const std::string dir = dir_expanded.empty()
        ? GetSimulationDirectory(M_BH_Msun, B_pc, ecc_val, psi0_rad,
                                 nx1, nx2, nx3, rho_cluster, nu_iso_code, R_inner_code, R_outer_code, t_end)
        : GetSimulationDirectory(M_BH_Msun, B_pc, ecc_val, psi0_rad,
                                 nx1, nx2, nx3, rho_cluster, nu_iso_code, R_inner_code, R_outer_code, t_end,
                                 dir_expanded);
    if (Globals::my_rank == 0) {
        const int status = std::system(("mkdir -p " + dir).c_str());
        if (status != 0)
            std::cerr << "Warning: could not create directory " << dir << std::endl;
        else
            std::cout << "Output directory: " << dir << std::endl;
    }
    pin->SetString("job", "problem_id", dir + "/sim");
    return dir;
}

//========================================================================================
// Helper: write parameters to JSON
//========================================================================================
void WriteSimulationParameters(const std::string& dir, Real M_BH_Msun, Real B_pc, Real ecc_val, 
                                Real psi0_rad, Real R_inner_code, Real R_outer_code, Real R_circ_code, Real Dt_inflow_yr,
                                int nx1, int nx2, int nx3, Real rho_cluster_cgs, Real rho_gas_cgs, Real epsilon, 
                                Real T_gas_K, Real mu, 
                                Real c_s_cgs, Real Omega_cgs, Real time_unit_cgs,
                                Real length_unit_cgs, Real gm_code_val, Real nu_iso_code, 
                                Real xi0_val, Real t_end) {
    if (Globals::my_rank != 0) return;

    const Real A_pc      = B_pc / std::sqrt(1.0 - ecc_val * ecc_val);
    const Real Myr_cgs   = 1.0e6 * yr_cgs;

    const std::string path = dir + "/parameters.json";

    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: could not write " << path << std::endl;
        return;
    }
    f.precision(16);

    f << "{\n";
    f << "  \"physical_input\": {\n";
    f << "    \"M_BH_Msun\":         " << M_BH_Msun                      << ",\n";
    f << "    \"B_pc\":              " << B_pc                           << ",\n";
    f << "    \"ecc\":               " << ecc_val                        << ",\n";
    f << "    \"psi0_rad\":          " << psi0_rad                       << ",\n";
    f << "    \"nx1\":               " << nx1                            << ",\n";
    f << "    \"nx2\":               " << nx2                            << ",\n";
    f << "    \"nx3\":               " << nx3                            << ",\n";
    f << "    \"rho_cluster_cgs\":   " << rho_cluster_cgs                << ",\n";
    f << "    \"rho_gas_cgs\":       " << rho_gas_cgs                    << ",\n";
    f << "    \"epsilon\":           " << epsilon                        << ",\n";
    f << "    \"T_gas_K\":           " << T_gas_K                        << ",\n";
    f << "    \"mu\":                " << mu                             << ",\n";
    f << "    \"NonInert\":          " << (NonInert ? "true" : "false")  << "\n";
    f << "  },\n";

    f << "  \"physical_constants_cgs\": {\n";
    f << "    \"G_cgs\":     " << G_cgs     << ",\n";
    f << "    \"M_sun_cgs\": " << M_sun_cgs << ",\n";
    f << "    \"k_B_cgs\":   " << k_B_cgs   << ",\n";
    f << "    \"m_p_cgs\":   " << m_p_cgs   << ",\n";
    f << "    \"pc_cgs\":    " << pc_cgs    << ",\n";
    f << "    \"yr_cgs\":    " << yr_cgs    << ",\n";
    f << "    \"au_cgs\":    " << au_cgs    << "\n";
    f << "  },\n";

    f << "  \"derived_quantities_cgs\": {\n";
    f << "    \"A_pc\":            " << A_pc                            << ",\n";
    f << "    \"c_s_cgs\":         " << c_s_cgs                         << ",\n";
    f << "    \"R_circ_code\":     " << R_circ_code                     << ",\n";
    f << "    \"Dt_inflow_yr\":    " << Dt_inflow_yr                    << ",\n";
    f << "    \"Omega_rads\":      " << Omega_cgs                      << ",\n";
    f << "    \"Omega_Myr\":       " << Omega_cgs * Myr_cgs            << "\n";
    f << "  },\n";

    f << "  \"unit_system_cgs\": {\n";
    f << "    \"length_unit_cgs\":   " << length_unit_cgs      << ",\n";
    f << "    \"velocity_unit_cgs\": " << c_s_cgs              << ",\n";
    f << "    \"time_unit_cgs\":     " << time_unit_cgs        << ",\n";
    f << "    \"time_unit_yr\":      " << time_unit_cgs / yr_cgs   << ",\n";
    f << "    \"density_unit_cgs\":  " << rho_gas_cgs          << "\n";
    f << "  },\n";

    f << "  \"code_units\": {\n";
    f << "    \"gm_code\":         " << gm_code_val                                   << ",\n";
    f << "    \"B_code\":          " << B_pc  * pc_cgs / length_unit_cgs              << ",\n";
    f << "    \"A_code\":          " << A_pc  * pc_cgs / length_unit_cgs              << ",\n";
    f << "    \"Omega_code\":      " << Omega_cgs * time_unit_cgs                     << ",\n";
    f << "    \"xi0_rad\":         " << xi0_val                                       << ",\n";
    f << "    \"R_inner_code\":    " << R_inner_code                                  << ",\n";
    f << "    \"R_outer_code\":    " << R_outer_code                                  << ",\n";
    f << "    \"nu_iso_code\":     " << nu_iso_code                                   << ",\n";
    f << "    \"gamma_gas\":       " << gamma_gas                                     << "\n";
    f << "  }\n";
    f << "}\n";

    f.close();
    std::cout << "Parameters written to: " << path << std::endl;
}
