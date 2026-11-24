# Transmon Qubit Simulation

Numerical simulation of a superconducting transmon qubit, built up step by step in Jupyter notebooks: from the static Hamiltonian, to microwave driving, to shaped control pulses.

![hero figure](figures/hero.png)

## Contents

| Notebook | Topic |
|---|---|
| [`1_transmon_hamiltonian.ipynb`](notebooks/1_transmon_hamiltonian.ipynb) | Charge-basis Hamiltonian, spectrum, charge dispersion, anharmonicity, physical parameters |
| [`2_microwave_drive.ipynb`](notebooks/2_microwave_drive.ipynb) | Time-dependent driving, Rabi oscillations, amplitude scan, off-resonant drive, leakage |
| [`3_gaussian_pulse.ipynb`](notebooks/3_gaussian_pulse.ipynb) | Gaussian pulse envelope, pi-pulse calibration, fidelity, rotating-frame phase |
| `4_resonator_coupling.ipynb` *(planned)* | Qubit-resonator coupling, dispersive shift |
| `5_dephasing.ipynb` *(planned)* | Lindblad master equation, T1 and T2 |

Shared code lives in [`transmon_func.py`](transmon_func.py):

| Function | Purpose |
|---|---|
| `transmon_hamiltonian` | Builds the charge-basis Hamiltonian matrix |
| `get_energy_levels` | Diagonalizes it and returns the energies and the charge operator in the energy basis |
| `gaussian_envelope` | Gaussian pulse envelope, truncated at 5 sigma |
| `driven_hamiltonian` | Static Hamiltonian plus the pulsed microwave drive |
| `schrodinger_equation` | Right-hand side passed to the ODE solver |


## 1. Static transmon Hamiltonian

The transmon Hamiltonian is

```math
\hat{H}_0
=
4 E_C \left( \hat{n} - n_g \right)^2
-
E_J \cos \hat{\varphi}
```

Writing the cosine in terms of charge-translation operators,

```math
-E_J \cos \hat{\varphi}
=
-\frac{E_J}{2}
\left(
e^{i \hat{\varphi}}
+
e^{-i \hat{\varphi}}
\right)
```

```math
e^{i \hat{\varphi}} \, | N \rangle
=
| N + 1 \rangle
```

```math
e^{-i \hat{\varphi}} \, | N \rangle
=
| N - 1 \rangle
```

After truncating the charge states to $N = -N_{\max}, \ldots, N_{\max}$, the Hamiltonian is a tridiagonal matrix:

```math
H_{nn}
=
4 E_C \left( n - n_g \right)^2
```

```math
H_{n,n+1}
=
H_{n+1,n}
=
-\frac{E_J}{2}
```

The lowest levels are compared against the analytic (Duffing oscillator) approximation from Koch et al. (2007):

```math
E_m
\approx
-E_J
+
\sqrt{8 E_J E_C} \left( m + \frac{1}{2} \right)
-
\frac{E_C}{12} \left( 6 m^2 + 6 m + 3 \right)
```

**Charge dispersion** is the sensitivity of each level to the offset charge:

```math
\epsilon_m
=
\left|
E_m \left( n_g = 0 \right)
-
E_m \left( n_g = 0.5 \right)
\right|
```

**Anharmonicity** is the difference between the 1-2 and 0-1 transitions:

```math
\alpha
=
\left( E_2 - E_1 \right)
-
\left( E_1 - E_0 \right)
=
E_2 - 2 E_1 + E_0
```

**Physical parameters.** The circuit parameters are converted to energies with

```math
E_C
=
\frac{e^2}{2 C}
```

```math
E_J
=
\frac{\hbar I_c}{2 e}
```

Parameters used: $C = 70$ fF, $I_c = 40$ nA, giving $E_C / h = 0.28$ GHz, $E_J / h = 19.9$ GHz, $E_J / E_C = 71.8$.

**Results:**
- 0-1 transition frequency: 6.34 GHz
- Anharmonicity: -0.31 GHz
- Agreement with the Duffing approximation: levels 0 and 1 match it closely already at the minimum cutoff (error of 0.1% at N = 5), while higher levels move away from it. This is expected, since the approximation expands the cosine potential only to fourth order in the phase, which is accurate for the lowest levels when $E_J / E_C$ is large but breaks down higher up the well.

![cutoff convergence](figures/01_cutoff_convergence.png)

![charge dispersion](figures/01_charge_dispersion.png)

![anharmonicity](figures/01_anharmonicity.png)


## 2. Microwave drive and Rabi oscillations

The gate charge is driven at the microwave frequency:

```math
n_g(t)
=
n_g^{(0)}
+
n_g^{(\mathrm{ac})} \cos \left( \omega_d t \right)
```

Expanding the Hamiltonian and dropping terms that only contribute a global phase, the drive Hamiltonian is

```math
H_{\mathrm{drive}}(t)
=
-8 E_C \, n_g^{(\mathrm{ac})} \, \hat{n} \, \cos \left( \omega_d t \right)
```

The charge operator is transformed into the basis of the lowest 6 energy eigenstates:

```math
n_{jk}
=
\langle j | \hat{n} | k \rangle
```

which shows which levels the drive couples strongly (the nearest-neighbour elements dominate).

The state evolves according to the time-dependent Schrodinger equation:

```math
\frac{d}{dt} | \psi(t) \rangle
=
-\frac{i}{\hbar}
H(t)
| \psi(t) \rangle
```

integrated with an adaptive Runge-Kutta solver (DOP853) with the maximum step limited to 1/40 of a drive period.

For a weak resonant drive, the two-level approximation predicts

```math
P_1(t)
=
\sin^2 \left( \frac{\Omega_R \, t}{2} \right)
```

with Rabi frequency

```math
\Omega_R
=
\frac{8 E_C \, n_g^{(\mathrm{ac})}}{\hbar}
\left| \langle 0 | \hat{n} | 1 \rangle \right|
```

Leakage out of the qubit subspace is tracked as

```math
P_{\mathrm{leak}}(t)
=
1
-
P_0(t)
-
P_1(t)
```

**Results:**
- Simulated Rabi oscillations vs the analytic prediction agree to 0.0005%
- Rabi frequency scales with drive amplitude (scanned $n_g^{(\mathrm{ac})} = 0.0005, 0.002, 0.01$)
- Maximum leakage out of the {0, 1} subspace: 0.0003 (at $n_g^{(\mathrm{ac})} = 0.001$)
- Drive detuned to 1.2 times the 0-1 frequency: level 1 population stays below 5e-6.

![resonant drive populations](figures/02_populations.png)

![amplitude scan](figures/02_amplitude_scan.png)


## 3. Gaussian pulse

A Gaussian envelope, truncated at 5 sigma, multiplies the drive:

```math
A(t)
=
\exp \left( -\frac{(t - t_0)^2}{2 \sigma^2} \right)
```

The width is chosen so the pulse area gives a rotation of pi:

```math
\Omega_R \, \sigma \, \sqrt{2 \pi}
=
\pi
```

```math
\sigma
=
\frac{\sqrt{\pi}}{\sqrt{2} \, \Omega_R}
```

The fidelity is measured against the target state $|1\rangle$:

```math
F
=
\left|
\langle 1 | \psi_{\mathrm{final}} \rangle
\right|^2
```

and the relative phase between the two qubit levels is evaluated in the frame rotating at the drive frequency:

```math
\phi
=
\arg \left(
\frac{c_1(T)}{c_0(T)}
\, e^{i \omega_d T}
\right)
```

**Results:**
- Final fidelity: 0.9999
- Final populations $P_0$ and $P_1$: 0.0001 and 0.9999
- Final leakage: 4e-7
- Rotating-frame phase: 0.009
- Population plot included at the top


## Validation

- Spectrum agrees with the Duffing approximation for the lowest levels at large $E_J / E_C$
- Charge dispersion decays exponentially with $\sqrt{E_J / E_C}$, as predicted by Koch et al.
- Rabi frequency matches the analytic formula
- Gaussian pi pulse reaches F close to 1, with the remaining error (about 1e-4) coming from residual ground-state population rather than leakage
- Cross-check against `scqubits`: 0-1 frequency agrees to 0.005%


## Planned work

- [ ] Coupling to a readout resonator and the dispersive shift
- [ ] Dephasing and relaxation with a Lindblad master equation (T1, T2, Ramsey, echo)
- [ ] DRAG pulse to reduce leakage to the second excited state


## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

The notebooks import from `transmon_func.py`, so launch Jupyter from the repository root, or keep the notebooks next to that file.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── transmon_func.py     # shared Hamiltonian, pulse and ODE functions
├── notebooks/
│   ├── 1_transmon_hamiltonian.ipynb
│   ├── 2_microwave_drive.ipynb
│   └── 3_gaussian_pulse.ipynb
└── figures/             # images used in this README
```

## References

1. J. Koch et al., "Charge-insensitive qubit design derived from the Cooper pair box," *Phys. Rev. A* **76**, 042319 (2007).
2. P. Krantz et al., "A quantum engineer's guide to superconducting qubits," *Appl. Phys. Rev.* **6**, 021318 (2019).