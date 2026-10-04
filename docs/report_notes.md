# Report notes - LFP DFN case study (A123 ANR26650M1B)

Running log of verified findings and decisions. Each section is filled in when
its step is complete and checked.

---

## Step 1 - Data preparation

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

---

## Step 4 - Optimization

### Method assessment: gradient-based vs non-gradient
| | Gradient-based (trust-region least squares, Levenberg-Marquardt) | Non-gradient (Nelder-Mead, Powell, CMA-ES, PSO, Bayesian) |
|---|---|---|
| Evaluations needed | tens (uses the residual Jacobian) | hundreds to thousands |
| Cost here (~30-60 s per evaluation of all 5 training sets) | ~0.5-1.5 h | ~4-25 h |
| Global search | no - needs a good start | yes (CMA-ES, PSO) |
| Robustness to noisy / failing simulations | lower | higher |
| By-products | Jacobian -> parameter uncertainty and correlation | none |
Choice: gradient-based `scipy.optimize.least_squares` (method 'trf', bounds) from the physics-based
Task 3 starting point, with log10-scaled parameters. The extensive manual analysis (Steps 3a-3e)
provides a starting point close to the optimum, which removes the main weakness of local methods;
the Jacobian gives parameter uncertainties and correlations. Robustness measures: bounds from Step 3,
failed / stopped simulations penalised instead of crashing, checkpointing after every evaluation.

### Pipeline (step4_optimize.py)
- Training data: Charge 1C, 2C, 3C, 4C and Discharge 1C, simulated in parallel processes.
- Residual: model - measured voltage per sample, each set weighted equally (1/sqrt(N)).
- Variable projection: series contact resistance (and an optional hysteresis term) enter the voltage
  linearly and are solved exactly by bounded linear least squares for every DFN evaluation.
- Coarse mesh during optimization (max 0.14 mV RMS difference to the default mesh, Step 4a),
  default mesh for all final results.

### Problems found and fixed during Step 4
1. Jacobian step degenerate: scipy's relative finite-difference step collapses to ~1e-8 at a
   parameter value of 0 (the start in log space), far below the solver tolerance -> zero Jacobian,
   the optimizer never moved. Fixed with an explicit Jacobian using an absolute step of 0.03 decades
   (~7 %); verified on synthetic data with a known optimum away from the start (recovered exactly).
2. Hysteresis hypothesis rejected (see Step 4a notes): the term only absorbed CV-phase error; removed.
3. Memory: 5 parallel workers on this laptop exceeded RAM (85 % used, disk paging) and the evaluation
   time rose from ~30 s to ~250 s; 3 workers with other applications closed restored ~50 s.

### Run A - four parameters (k_n, k_p, D_n, D_p), all data except the initial history window
Converged (ftol) after 34 evaluations. Training RMSE 129.7 -> 123.8 mV (-4.5 %) only.
k_n 0.589 x, k_p 1.41 x (+/- 0.55 dex: not identifiable), D_n 1.61e-14 m2/s, D_p 0.417 x Prada2013;
correlations k_n-k_p -0.67, D_n-D_p -0.78; contact resistance ~0.
Assessment: not acceptable as final. The remaining error is the CV phase (cell held at 3.600 V while
the current tapers to ~5 mA, i.e. an almost-equilibrium voltage ~150 mV above the rested full-charge
OCV of 3.447 V) - a slow top-of-charge relaxation that a constant-diffusivity DFN cannot produce.
Chasing it pushed D_n below the 3e-14 m2/s that Task 3e showed is needed to complete HPPC.

### Model-structure tests (step4b_dp_shape_probe.py) - state-dependent solid diffusivity
Form: D(s) = D0 * 10^-(a exp(-s/w)) near an end of the stoichiometry range, w = 0.05.
Base: Run A values, k_p fixed 1, D_n 3e-14. Errors in mV.

LFP near full delithiation (a_top) - targets the CV hold:
| a_top | C1 CV RMSE | C4 |
|---|---|---|
| 0 | 161.9 | completes, CV 152.7 |
| 0.1 | 160.7 | completes, CV 155.3 |
| 0.25 | 158.1 | stops at 25 % (voltage overshoot) |
| 0.5 | 150.4 | stops at 24 % |
| 1-3 | - | stops at 22-55 % (C1 and C4) |
Rejected: too weak at 1C and too strong at 4C simultaneously. A diffusion limitation scales with
current, but the CV hold occurs at a few mA, i.e. it is a slow relaxation, not polarization. Likely
cause: LFP phase-transformation / many-particle effects, not represented in the DFN.

LFP near full lithiation (a_bot): no effect at all - in the calibrated window LFP only reaches y ~ 0.75
at the end of discharge.

Graphite near empty (b_bot) - targets the end-of-discharge plunge and slow recovery:
| b_bot | D1 total | D1 last 5 % | D1 rest after | D1 CC part |
|---|---|---|---|---|
| 0 | 119.8 | 308.2 | 306.9 | 48.9 |
| 1 | 115.7 | 287.8 | 299.5 | 48.8 |
| 2 | 104.6 | 236.0 | 275.8 | 48.6 |
| 3 | 64.8 | 210.9 | - (run stops at 91 %) | 48.3 |
Accepted: steady improvement exactly where intended, constant-current part and charge tests unaffected.
Physical finding: the top-of-charge behaviour is controlled by LFP near full delithiation, the
end-of-discharge behaviour by graphite near full delithiation.

### Final-model design (decided before validation)
- Fitted: k_n, D_n, D_p and b_bot (graphite diffusivity reduction near empty); k_p fixed at Prada2013.
- Cost window: constant-current parts of the charge tests (from 0.15 Ah) + complete 1C discharge.
  The CV phase is excluded (structural limit, shown by Run A and the a_top probe) and its error is
  reported separately.
- Run A is kept as a sensitivity result (effect of including the CV phase in the fit).

### Training-only final fit (results/step4_final) - failed validation
k_n 0.92, D_n 2.45e-14 m2/s, D_p 0.39, b_bot 2.22, R_c ~0 (k_p fixed). Training 40.4 -> 26.2 mV
(D1 119.9 -> 53.3 mV). HPPC validation: coverage 79.8 % (minimum-voltage stop), 0-14.3 h window
RMSE 44.4 mV vs Task 3 37.6 mV, dynamic 48.4 vs 40.9 mV -> worse than Task 3. Residual analysis:
instantaneous pulse resistance unchanged (model 14.8 -> 7.6 mOhm non-linear, cell flat ~7.5-8 mOhm).
Cause: constant-current training data carry almost no information on the sub-second kinetic/ohmic
response and never reach 15C. Physical variables: anode potential vs Li/Li+ down to -139 mV (HPPC)
and -110 mV for 881 s (4C charge): lithium-plating risk region (not modelled).

### Split fit: HPPC odd SOC levels in training, even levels held out (results/step4_split)
12 SOC levels; training 22,438 HPPC samples, held out 26,046. Fitted k_n, k_p, D_n (>= 3e-14), D_p, b_bot.
Result: k_n 1.16, k_p 10.5, D_n 3.12e-14, D_p 0.37, b_bot 2.95, R_c 2.10 mOhm; max correlation 0.43
(D_n-b_bot) - pulse data made k_p and R_c identifiable and removed the strong correlations.
Training: D1 119.9 -> 51.9 mV, C4 20.8 -> 17.7 mV, HPPC training levels 42.1 -> 41.8 mV.

Mesh sensitivity: HPPC completed with the coarse optimization mesh but stopped at 77.9 % on the default
mesh (held-out 36.9 vs 46.1 mV therefore not comparable). Near b_bot ~3 the graphite diffusivity falls
steeply near empty; the coarse radial mesh smooths it, the default mesh resolves it. The Step 4a mesh
check held for the Task 3 model, not after adding b_bot -> refit on the default mesh (results/step4_split_fine).

### Fine-mesh refits: cost cliff and b_bot feasibility
- Split fit on the default mesh (results/step4_split_fine): the optimizer never moved (final = start).
  Jacobian probes showed a cliff: D_n 3.15e-14 -> rms 69.2 mV (HPPC stops, penalty), D_n 3.38e-14 ->
  51.0 mV. A gradient method assumes a smooth cost; its larger steps failed and it shrank back to the
  start. Fixes: early stops penalised over ALL samples a run cannot simulate (incl. held-out levels -
  only the fact that it stopped is used, never held-out voltages), and every evaluation line reports
  stopped tests.
- With the full penalty, even D_n 3.6e-14 / b_bot 2.3 stopped -> feasibility scan
  (step4c_bbot_feasibility.py), default mesh, k_n 1.16, k_p 10.5, D_p 0.37:
  | D_n | b_bot | HPPC coverage | D1 RMSE |
  |---|---|---|---|
  | 3.6e-14 | 0.0 | 100 % | 122.0 mV |
  | 3.6e-14 | 0.5 | 90.5 % | 120.7 mV |
  | 3.6e-14 | 1.0 | 90.4 % | 118.8 mV |
  | 3.6e-14 | 1.5 | 85.0 % | 115.7 mV |
  | 3.6e-14 | 2.0 | 85.0 % | 110.3 mV |
  | 6.0e-14 | 2.0 | 90.4 % | 119.4 mV |
  Any b_bot > 0 empties the graphite surface during 15C pulses at low SOC. b_bot is REJECTED for the final
  model: it improves the slow 1C end of discharge but cannot coexist with 15C pulses. Structural limit of a
  single-particle-size DFN (a particle-size distribution would let small particles carry pulses while large
  ones produce the late plunge).

### Final model design
- Fitted: k_n, k_p, D_n (>= 3.5e-14, start 3.6e-14 = verified feasible point), D_p; contact resistance
  solved exactly; no b_bot, no hysteresis.
- Training: Charge 1C-4C (CC parts from 0.15 Ah), Discharge 1C (complete), HPPC odd SOC levels (dynamic parts).
- Independent validation: HPPC even SOC levels. Full-HPPC metrics are partly training data (odd levels)
  and partly calibration data (rests, Step 3c) - only the even-level dynamic error is fully independent.
- Default (validation) mesh during optimization; run: results/step4_final_model.

### Final model and validation (results/step4_final_model, results/step5_final_model)
Final parameters: k_n 1.082 x (+/-0.01 dex), k_p 9.52 x (+/-0.42), D_n 3.5e-14 m2/s (at feasibility bound),
D_p 1.773 x (+/-0.20), R_c 2.09 mOhm; correlations < 0.4. Flat cost (k_p 0.14-57, D_p 0.7-2.3 -> cost 54.4-54.6 mV).
Run resumed once after a thermal shutdown. Coverage 100 % on all six tests.
Training RMSE (Task 3 -> final): C1 20.8->21.2, C2 20.7->21.8, C3 19.7->20.3, C4 20.8->17.4, D1 119.9->120.6,
HPPC odd levels 42.1->39.3 mV.

HPPC validation [mV]          RMSE  RMSE_tw   MAE    max   mean  rest  dynamic
full test   Task 3            43.1   22.5    29.8  350.7  10.0  11.7   46.8
            final             40.9   21.6    29.4  293.9   8.7  11.6   44.4
0-14.3 h    Task 2            37.3   30.1    29.9  206.8 -18.9  28.2   38.8
            Task 3            37.6   20.3    26.5  200.8   6.5  10.0   40.9
            final             35.1   19.2    25.7  165.7   4.3  10.0   38.1
held-out    Task 3            46.1   27.4    33.2  350.7  13.9  16.4   48.6
            final             44.5   26.8    33.3  293.9  13.3  16.3   46.9
training lv final             39.3   19.5    29.1  287.5   5.2   8.7   41.6
(Subset time weights computed on the full time grid - an earlier version computed them on the subset and
gave 53-55 mV; fixed.)

Physical variables (final): anode potential vs Li/Li+ at separator: HPPC -122.3 mV (276 s < 0 V),
4C charge -93.7 mV (791 s < 0 V). Likely overestimated: model charge-pulse resistance 16.8 vs 5.9 mOhm
measured; isothermal; no plating reaction.