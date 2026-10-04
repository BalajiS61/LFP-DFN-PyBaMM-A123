# Report notes - LFP DFN case study (A123 ANR26650M1B)

Running log of verified findings and decisions. Each section is filled in when
its step is complete and checked.

---

## Step 0 - Data preparation

**Files and formats**
- `Charge_1C/2C/3C/4C.mat`: top-level vectors `Time`, `Current`, `Voltage` (1 s sampling).
- `Discharge_1C.mat`, `HPPC.mat`: struct `meas` with `Time`, `Current`, `Voltage`, `Ah`, `Power`, `Battery_Temp_degC`, ...
- Sign convention in all files: charge > 0, discharge < 0. PyBaMM uses discharge > 0, so the current is flipped once in `data_loader.py`.

**Measured capacities**
| Test | Charge throughput |
|---|---|
| C1 / C2 / C3 / C4 (charged in) | 2.423 / 2.447 / 2.456 / 2.452 Ah |
| D1 (discharged) | 2.451 Ah (cycler counter 2.450 Ah) |
| HPPC (net discharged at the end) | 2.432 Ah (cycler counter 2.438 Ah) |
| HPPC (deepest point of discharge) | about 2.48 Ah (cycler counter minimum 2.484 Ah); the final charge pulse returns ~50 mAh |

Usable capacity is about 2.45 Ah, slightly below the 2.5 Ah rating. Higher C-rate charges start from lower voltage (2.83-2.87 V vs 2.94 V at 1C), so they put in slightly more charge.

**Temperature**: 22.8-24.6 degC in all logged tests, so an isothermal model is justified (as stated in the assignment).

**Issue found and fixed - HPPC current representation**
- The cycler logs sparsely during the slow 0.75 A (0.3C) discharge steps between pulse sets (gaps up to 267 s). The sample after each gap already holds -0.75 A, so current flowed during the whole gap.
- Linear interpolation of the current loses 0.313 Ah (13 %) over the test; forward hold loses 0.620 Ah; **backward zero-order hold matches the cycler's Ah counter within 6 mAh (0.25 %)**.
- Decision: the model receives a backward zero-order-hold step profile for all datasets. Verified: the step profile carries exactly the same charge as the data for all six tests.
- 333 duplicate timestamps in HPPC (and 1 in D1) carried identical currents and were removed.

---

## Step 1 - Model and parameter set

- **Model**: PyBaMM DFN (`pybamm.lithium_ion.DFN()`), isothermal. PyBaMM 26.9.0, IDAKLU solver.
- **Parameter set**: `Prada2013`. Justification: it is PyBaMM's only LFP/graphite parameter set (the other built-in sets - e.g. Chen2020, Ecker2015, Marquis2019, OKane2022, Mohtat2020, ORegan2022 - are NMC, NCA or LCO cathodes with an OCV of roughly 3.6-4.2 V and cannot represent an LFP cell with its ~3.3 V plateau), and it and was parametrised on an A123 26650 cell (ANR26650M1A, 2.3 Ah), the predecessor of the ANR26650M1B (2.5 Ah). Same manufacturer, chemistry and format, so it is the closest available starting point.
- **Current input**: measured current as a step-shaped interpolant; solver breakpoints at every current step so the adaptive solver cannot step over a pulse.
- **Initial state**: set from the measured initial rest voltage (3.4471 V) using the parameter set's own voltage limits (2.0-3.6 V). Voltage limits are widened to 1.0-4.5 V only afterwards, for event handling, so they do not change the SOC definition. (An earlier version widened the limits first, which silently redefined SOC 0/100 %; this was found and corrected.)

---

## Step 2 - Baseline HPPC simulation (un-tuned Prada2013)

**Electrode balance at the start of HPPC**
- Anode stoichiometry 0.8042, cathode stoichiometry 0.0089.
- Electrode capacities: Q_n = 2.907 Ah, Q_p = 3.292 Ah.
- Dischargeable from this state: **2.338 Ah** (anode-limited), vs **2.432 Ah** measured in the test.

**Result**: the simulation stops at 51,369 s (14.3 h, **56.6 % of the test**) on the minimum-voltage event during a 37.5 A (15C) discharge pulse, after 1.599 Ah discharged.

**Root-cause analysis of the early stop** (diagnostic script `step2b_diagnose_stop.py`)
| Candidate cause | Evidence | Verdict |
|---|---|---|
| Wrong current input | model and measured charge both 1.599 Ah at the stop | ruled out |
| Wrong initial state | start corresponds to fully charged (2.338 Ah available vs 2.3 Ah nominal); forcing SOC = 1.0 stops at the same time | ruled out |
| Electrolyte depletion | minimum c_e 739 of 1200 mol/m3 | ruled out |
| Capacity exhausted | 0.74 Ah still in the cell at the stop (bulk anode stoichiometry 0.8042 - 1.599/2.907 = 0.254) | not the cause of the stop |
| **Graphite solid diffusion too slow** | anode particle **surface** stoichiometry 0.005 everywhere while the **bulk** is 0.254 | **cause of the stop** |

Conclusion: Prada2013's graphite diffusivity is too low for this cell's 15C pulses; the particle surface empties while lithium remains inside, and the voltage collapses.

**Baseline metrics** (error = simulated - measured, over the simulated part, final 60 s collapse excluded)
| Metric | Value |
|---|---|
| RMSE | 37.3 mV |
| MAE | 29.9 mV |
| Max abs error | 206.8 mV |
| Mean error (bias) | -19.6 mV (full simulated part) |
| RMSE at rest / under load | 32.9 / 67.2 mV |

Including the final collapse: RMSE 44.2 mV, max 1638 mV (single end-of-run point at 1.108 V vs 2.746 V measured).

Initial-state comparison: forcing SOC = 1.0 gives RMSE 64.0 mV (without collapse), so the measurement-based (voltage) initial state is kept as the official baseline.

**Why these error metrics**
- **RMSE** (primary): standard for model-vs-measurement voltage fits, in volts so directly interpretable, and it penalises large deviations more than small ones. That matters here because the large errors occur during high-rate pulses, which are exactly the dynamics a DFN should capture. It is also the quantity minimised in least-squares optimisation, so training and validation use a consistent measure.
- **MAE**: robust companion to RMSE; a large RMSE/MAE ratio signals that error is concentrated in short events (pulses) rather than spread evenly.
- **Max absolute error**: worst case, relevant for voltage-limit and safety use of the model.
- **Mean error (bias)**: separates a systematic offset (OCV / capacity mismatch) from dynamic error.
- **RMSE split into rest and load**: rest error reflects thermodynamics (OCV, capacity, SOC drift); load error reflects kinetics, transport and resistance. This split points directly to which parameter groups to tune.

**Observations from the plots**
- Rest-voltage offset grows with depth of discharge (about -15 mV early to about -60 mV at 14 h): model SOC drifts relative to the cell, so the capacity and OCV shape differ.
- Error histogram is multi-modal (peaks around -50, -25, +5 mV): each mode is a different SOC level with a different rest offset, i.e. OCV mismatch.
- Pulse errors of +/-100-150 mV; in the zoomed pulse the model under-predicts the voltage drop during the 37.5 A pulse (+100 mV) and relaxes too slowly afterwards (error drifts from +15 to -20 mV over about 500 s): kinetics and solid diffusion.
- First ~1.2 h (high SOC): largest charge-pulse errors, so the top end of the OCV / stoichiometry window is off.

**Implications for Step 3 (parameter priorities)**
1. Negative electrode solid diffusivity (cause of the stop and pulse relaxation errors).
2. Capacity / stoichiometry window (2.338 vs 2.432 Ah; SOC drift).
3. OCP alignment (32.9 mV rest RMSE).
4. Kinetics / resistance (67.2 mV load RMSE).

---

## Step 3a - Residual analysis: when and why residuals appear

Script: `step3a_residual_analysis.py` (uses the Task 2 baseline, final 60 s collapse excluded).
Coverage: 30,873 samples to 51,299 s, **SOC 100 % -> 36 % only** (the baseline stops there).
The analysis will be repeated on the full SOC range once the model completes the test.
SOC reference: measured HPPC throughput, 2.432 Ah.

**1. Error by operating phase**
| Phase | Share | RMSE | Mean | Max [mV] |
|---|---|---|---|---|
| Long rest (OCV-like) | 16.5 % | 28.2 | -23.3 | 59.7 |
| Relaxation (< 300 s after load) | 58.4 % | 33.9 | -23.6 | 188.0 |
| Low-rate discharge (0.75 A) | 5.6 % | 58.7 | -53.8 | 115.3 |
| Low-rate charge | 7.9 % | 33.2 | +18.8 | 123.0 |
| Discharge pulse | 7.5 % | 61.3 | -6.8 | 206.8 |
| Charge pulse | 4.3 % | 26.0 | +17.1 | 121.3 |

**2. Error by SOC**: rest offset grows from -8.5 mV (90-100 %) to -60.0 mV (30-40 %); the 30-40 % bin is the worst (RMSE 63.6 mV) as the model approaches its graphite diffusion limit.

**3. End-of-rest (OCV-like) offsets**: -16.4, -15.3, -23.5, -33.2, -21.2, -23.8, -55.6 mV at SOC 0.948 ... 0.384. The model is systematically below the measured OCV, and the gap widens at low SOC.

**4. Pulse resistance** (30 valid 10 s constant-current pulses; 6 voltage-limited charge pulses and 8 single-sample current transitions excluded; R_inst at 0.11 s)
| Pulse | Measured R_inst / R_end | Model R_inst / R_end | Measured growth | Model growth [mOhm] |
|---|---|---|---|---|
| Discharge 5C | 7.48 / 14.27 | 13.96 / 16.07 | 6.8 | 2.1 |
| Discharge 10C | 8.04 / 13.17 | 9.37 / 11.66 | 5.1 | 2.3 |
| Discharge 15C | 7.92 / 12.50 | 7.46 / 9.61 | 4.6 | 2.2 |
| Charge 4C | 5.91 / 12.51 | 15.78 / 18.12 | 6.6 | 2.3 |
(growth = R_end - R_inst, i.e. the time-dependent part of the polarization over 10 s)

**Interpretation - when and why**
| When | Why (physics) | Evidence |
|---|---|---|
| At rest, all SOC, growing at low SOC | OCV curve and capacity / stoichiometry window of Prada2013 differ from this cell; model SOC drifts (2.338 vs 2.432 Ah) | rest offsets -16 -> -56 mV; multi-modal error histogram |
| During low-rate steps (0.75 A) | charge-transfer resistance too large (exchange current density too low) | -53.8 mV discharge / +18.8 mV charge vs -23 mV rest offset |
| At pulse onset | measured fast response is ohmic (flat 7.5-8.0 mOhm across 5-15C); model response is dominated by non-linear Butler-Volmer kinetics (14.0 -> 7.5 mOhm as current rises) | R_inst vs C-rate |
| During pulses and relaxation | time-dependent (transport) polarization 2-3x too small in the model | growth 4.6-6.8 mOhm measured vs 2.1-2.3 mOhm model; slow error drift after pulses |
| High-rate pulses at low SOC | graphite solid diffusion cannot supply the surface: surface empties, voltage collapses | Step 2 diagnosis (surface 0.005 vs bulk 0.254) |
| First ~1.2 h (high SOC) | charge pulses hit the 3.6 V limit in the cell; top of the OCV / stoichiometry window misplaced in the model | largest charge-pulse errors there; 6 voltage-limited charge pulses |

---

## Step 3 - Parameter selection and grouping

Principle: tune in order of physical dependency. Thermodynamics (capacity, OCV) first, because an
optimizer fitting kinetics on a wrong SOC/OCV basis would compensate the offset with unphysical
kinetic or transport values.

| Priority | Group | Parameters | Evidence | Method |
|---|---|---|---|---|
| 1 | Thermodynamics / capacity | electrode stoichiometry window (initial concentrations, cyclable lithium), electrode capacities (active material fractions) | 2.338 vs 2.432 Ah; SOC drift; rest offsets | manual / direct calibration (Step 3c) |
| 1 | OCP | positive (LFP) and negative (graphite) OCP alignment to the measured OCV | rest offsets -16 to -56 mV, plateau slope mismatch | manual, measured pseudo-OCV (Step 3b) |
| 2 | Kinetics | negative and positive reaction rate constants (exchange current density) | non-linear model R_inst; low-rate polarization | optimizer (Step 4) |
| 3 | Ohmic | lumped contact resistance | flat measured R_inst of 7.5-8.0 mOhm | optimizer (Step 4) |
| 4 | Transport | negative and positive particle diffusivities | 2-3x too small polarization growth; graphite surface depletion | optimizer (Step 4) |

**Kept at Prada2013 values (not optimized)**, with reasons:
- Particle radii: with voltage data only, diffusivity and radius are not separately identifiable (the time constant is R^2/D), so the radius is fixed and the diffusivity fitted.
- Electrolyte properties (conductivity, diffusivity, transference number): electrolyte depletion was ruled out (minimum 739 of 1200 mol/m3), and adding them would create strong correlation with contact resistance and kinetics.
- Geometry, maximum concentrations, thermal parameters: physical constants of the design or not active (isothermal).

Identifiability note: contact resistance and exchange current density both affect the pulse onset;
they are separated by their different current dependence (ohmic is linear, charge transfer is not),
which is why the training data must span several C-rates (1C-4C charge + 1C discharge).

---

## Step 3b - Measured OCV vs Prada2013 OCV

Script: `step3b_ocv_analysis.py` (helpers in `ocv_tools.py`).

**Measured OCV points (13, full SOC range)**
- HPPC start (rested, fully charged): 3.4471 V at SOC 1.
- 12 HPPC end-of-rest points (rests of 1800 s) from SOC 0.948 to 0.025. The rests are not fully relaxed (end slopes 3.6-14.4 mV/h, steeper at low SOC), so an exponential relaxation V_inf + B exp(-t/tau) was fitted to each rest tail (first 120 s skipped) and extrapolated. All 12 fits valid, tau = 509-691 s, correction +0.3 to +1.2 mV. The raw end-of-rest voltages were therefore already within about 1 mV of equilibrium.
- SOC by coulomb counting (within 6 mAh of the cycler counter).

**Charge-branch OCV / hysteresis**: the 1C-4C charge files end during the CV phase of the CC-CV charge (voltage held at 3.600-3.601 V, current still 3.7-5.2 mA), with no rest. The charge-branch OCV and the LFP hysteresis therefore cannot be measured from this data. (A first version of the check treated the CV phase as a rest and reported a spurious 154 mV "hysteresis"; this was found and corrected by requiring |I| < 5 mA and a relaxing voltage.) All OCV points are on the discharge branch, consistent with the HPPC protocol.

**1C pseudo-OCV** (average of 1C charge and discharge): median half-gap 81 mV, up to 226 mV from the rest points, so it is used as a shape reference only.

**Prada2013 vs measured OCV**
- Prada window: graphite x 0.0176 -> 0.8100, LFP y 0.7035 -> 0.0038 (only 70 % of the LFP used); window capacity 2.303 Ah vs 2.432 Ah measured (ratio 0.947).

| Region | Prada - measured | Cause |
|---|---|---|
| SOC 100 % | +153 mV | Prada's SOC 1 is defined at 3.6 V; the rested full cell is at 3.447 V, so the top of the window is misplaced |
| SOC 95-39 % | -15 to -37 mV (RMSE 24.2 mV) | plateau too low and shape differs (window placement; graphite OCP shape) |
| SOC < 30 % | -67 to -642 mV | Prada's graphite empties too early (x -> 0.018) |

---

## Step 3c - Manual calibration of capacity and OCP (electrode balancing)

Script: `step3c_ocv_calibration.py`. Method: discharging from full moves both electrodes along a
lithium-conserving line, x = x100 - Ah/Q_n, y = y100 + Ah/Q_p, OCV = U_p(y) - U_n(x).
Fitted to the 13 verified OCV points (rests sigma 3 mV, start point sigma 20 mV), with the physical
constraints that no electrode is exhausted before 2.46 Ah and that active material + porosity <= 1.

**Identifiability of Q_p (parameter analysis)**: a first fit with Q_p free drove Q_p to the geometry
limit (positive active + porosity = 1.000, N/P 0.54) - unphysical. Cause: the LFP OCP is flat on its
plateau, so the OCV carries almost no information on Q_p. A profile scan (Q_p fixed, other parameters
refitted) showed the RMSE flat at 6.3-7.3 mV from 2.9 to 5.0 Ah but rising to 11 mV at 2.65 Ah and
149 mV at 2.41 Ah: the OCV gives a lower bound (about 3.1 Ah within 1 mV of the best fit) and no upper
bound. Q_p was therefore fixed at the Prada2013 value (3.292 Ah), which lies inside the data-consistent
range.

**End of discharge**: an initial assumption that the OCV at the measured end lies on a knee (2.75 V) was
removed: the rest at 2.371 Ah relaxes back to 3.176 V, so the final drop to 2.5 V under load is dynamic
polarization, not an OCV feature (typical for LFP). The calibrated OCV at the measured end is 3.150 V.

**Graphite OCP (external resource)**: candidates from Prada2013, Chen2020 and Ecker2015. PyBaMM's
Prada2013 set uses the Chen2020 graphite OCP (identical function, detected automatically).
| Graphite OCP | RMSE rest | Max rest | Note |
|---|---|---|---|
| Prada2013 (= Chen2020) | 11.8 mV | 19.4 mV | Q_n at the geometry bound - rejected |
| **Ecker2015** | **7.0 mV** | 13.6 mV | free fit - selected |
Assumption: graphite thermodynamics are largely cell-independent, so a graphite OCP measured on another
commercial cell (Ecker et al. 2015, Kokam) is a justified substitute; the remaining +/-11 mV structure
(e.g. +13.6 mV at SOC 0.69) reflects small differences in staging positions.

**Calibrated thermodynamic parameters**
| Quantity | Prada2013 | Calibrated |
|---|---|---|
| Graphite x at full / at 2.432 Ah | 0.810 / - | 0.9662 / 0.0803 |
| LFP y at full / at 2.432 Ah | 0.0038 / - | 0.0095 / 0.7481 |
| Q_n | 2.907 Ah | 2.745 Ah |
| Q_p | 3.292 Ah | 3.292 Ah (fixed) |
| Negative active material fraction | 0.5800 | 0.5476 |
| Positive active material fraction | 0.3740 | 0.3740 |
| N/P ratio | 0.883 | 0.834 |
| Usable capacity (2.432 Ah test) | 2.303 Ah window | 2.652 Ah available, anode-limited, 220 mAh margin |
| OCV RMSE at rest points | 229.6 mV (full range) / 24.2 mV (SOC 30-95 %) | 7.0 mV |
Volume check: negative 0.548 + 0.360 = 0.908, positive 0.374 + 0.426 = 0.800 (both <= 1).

---

## Step 3d - HPPC with calibrated thermodynamics

Rest error over the simulated part fell from 32.8 to 16.6 mV, confirming the OCV calibration. But the run
stopped at 2.3 h on the maximum-voltage event during a 20 A charge pulse. Diagnosis (`step2b_diagnose_stop.py
--model calibrated`): graphite surface stoichiometry 0.9999 while the bulk was 0.928 (charge conservation),
electrolyte 1037 of 1200 mol/m3, LFP surface 0.023-0.029 -> graphite solid diffusion limitation at the top end.
Together with the baseline stop (surface 0.005 vs bulk 0.254 at the bottom end), this shows the Prada2013
graphite diffusivity is too low for this cell in both directions; the calibration only moved the graphite
window upward (x100 = 0.966), exposing the limit at high SOC.

(A comparison bug - comparing different time windows - was found and corrected: comparisons now use the
common window of both runs with each run's final 60 s excluded.)

---

## Step 3e - Graphite diffusivity sensitivity (manual transport tuning)

Script: `step3e_diffusivity_scan.py`. Calibrated thermodynamics, D_n scaled x1-x300 from Prada's
3.0e-15 m2/s (constant). Common window for a fair comparison: 0-2.26 h.
| D_n factor | D_n [m2/s] | Coverage | RMSE common | Rest | Load | RMSE own part | Max |
|---|---|---|---|---|---|---|---|
| 1 | 3.0e-15 | 9.0 % | 45.8 | 16.6 | 91.5 | 45.8 | 194.9 |
| 3 | 9.0e-15 | 10.5 % | 43.0 | 16.3 | 85.3 | 44.2 | 172.4 |
| **10** | **3.0e-14** | **100 %** | 42.0 | 16.3 | 83.2 | **43.1** | 350.7 |
| 30 | 9.0e-14 | 100 % | 41.9 | 16.5 | 82.7 | 48.1 | 358.6 |
| 100 | 3.0e-13 | 100 % | 41.9 | 16.6 | 82.7 | 50.9 | 399.0 |
| 300 | 9.0e-13 | 100 % | 41.9 | 16.6 | 82.8 | 51.7 | 418.6 |
(errors in mV)

- x10 is the smallest factor that completes the full 25 h test; 3.0e-14 m2/s is within the range
  typically reported for graphite (~1e-14 to 1e-13 m2/s), whereas Prada's 3e-15 is below it.
- Above x10 the common-window error is flat, while the full-test error and max error increase: too-fast
  diffusion degrades the low-SOC behaviour. D_n is therefore identifiable mainly from the low-SOC part.
- Remaining error is dynamic (load 83 mV vs rest 16 mV) -> kinetics and contact resistance (Task 4).

**Manually tuned model (end of Task 3)**: Step 3c thermodynamics + D_n = 3.0e-14 m2/s. First model to
simulate the full HPPC test: RMSE 43.1 mV over 100 % of the test (baseline: 37.3 mV over the easier
first 56.6 % only).
**Task 4 bounds from this scan**: D_n in about [1e-14, 3e-13] m2/s.


**Note on the SOC axis (Steps 3a/3b)**: SOC in the residual and OCV plots uses the net HPPC throughput
(2.432 Ah) as reference, which is why the deepest point appears at about -2 % SOC. The calibration in 3c
works in Ah and is unaffected; the "no electrode exhausted" condition also holds at the deepest point
(2.48 Ah: graphite stoichiometry 0.966 - 2.48/2.745 = 0.063, about 170 mAh margin).

---

## Step 4a - Training data check (Task 3 model)

Initial states: charge tests start at full minus the charge put in until the CV cut-off (2.423-2.456 Ah
below full, graphite x0 = 0.071-0.083); D1 starts full. The OCV-inverted initial voltage suggests a state
~0.13 Ah deeper, as expected for a cell still relaxing after the preceding discharge (voltage below OCV);
charge counting is used.

| Set | Coverage | RMSE | Mean error | Runtime default / coarse |
|---|---|---|---|---|
| C1 | 100 % | 118.9 | -69.8 | 35.5 / 21.3 s |
| C2 | 100 % | 136.0 | -91.9 | 34.3 / 21.0 s |
| C3 | 100 % | 143.4 | -105.8 | 34.3 / 20.6 s |
| C4 | 100 % | 146.4 | -117.2 | 38.9 / 23.7 s |
| D1 | 100 % | 119.9 | +36.4 | 1.9 / 1.5 s |
(errors in mV; training cost = mean RMSE 132.9 mV)

Mesh check: coarse mesh (x_n 10, x_s 6, x_p 10, r_n 12, r_p 12) vs PyBaMM default differs by at most
0.14 mV RMS / 3.4 mV max, so the coarse mesh is used during optimization and the default mesh for all
final results.

**Error localisation (`step4_optimize.py --diagnose`)** - a first reading of the mean errors suggested
LFP hysteresis (charge error -69.8 mV vs discharge +36.4 mV at 1C). A one-state hysteresis term was
tested and the hypothesis was REJECTED: fitted on the constant-current parts only, the mean errors are
already +5, +12, +8, -1 mV (C1-C4) and -4 mV (D1), and the best contact resistance and hysteresis
amplitude are both ~0. The apparent asymmetry came from averaging over other phases.

Where the training error actually is (Task 3 model):
| Phase | Error | Observation |
|---|---|---|
| Constant-current charge / discharge | RMSE 38-54 mV, mean -4 to +12 mV | already good |
| CV phase at 3.6 V (incl. low-current CV tail) | -160 to -200 mV | cell held at 3.600 V while current tapers; model relaxes to ~3.45 V |
| Start of each charge test | +200 to +380 mV | cell starts at 2.83-2.94 V, still relaxing from the preceding discharge; model starts relaxed at its OCV (~3.15 V) |
| End of 1C discharge + rest after | +180 to +590 mV | cell plunges to 2.5 V and recovers slowly; model dips only to ~3.0 V |

Interpretation: the CV hold, the end-of-discharge plunge and the slow recovery all indicate large,
slowly relaxing concentration gradients at the ends of the range - together with the 2-3x too small
polarization growth in HPPC pulses this points to slow solid-state transport in LFP (D_p), which the
optimizer can fit. The start of the charge tests is different: the cell's relaxation history from the
preceding discharge is unknown to the model (uniform initial state), so no parameter can represent it.

**Decisions for Step 4**
- The first 0.15 Ah of each charge test (about 6 % of capacity; < 4 min at 1C, < 1 min at 4C) is excluded
  from the cost - the initial-history window. CV phases, end of discharge and rests stay in the cost.
  Final errors are reported on the full curves as well.
- D_p bounds widened to 0.001-1000 x Prada2013 (evidence points to slower LFP transport).
- Contact resistance and hysteresis kept as free linear terms (solved exactly by variable projection);
  if they stay ~0 this is reported as "not supported by the data".