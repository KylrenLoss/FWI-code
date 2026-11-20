# User Guide

This guide describes the acquisition geometry, the inputs and outputs, the configurable parameters and the expected behaviour of `ti.cu` and `tifwi.cu`. For step-by-step examples, see the [Tutorial](TUTORIAL.md).

## 1. Workflow

| Step | Program | Initial model | Data used in inversion | Output |
| --- | --- | --- | --- | --- |
| 1 | `ti` | `model_bin/mar2_lin_501_141_4400.bin` | Kaiser-windowed integrated data | `output/vp_inv_final.bin` |
| 2 | `tifwi` | `output/vp_inv_final.bin` | original (non-integrated) data, three frequency stages | `output2/vp_inv_final.bin` |

In both programs, the observed data are synthesized internally from the true model `model_bin/mar2_ext_501_141.bin`; no precomputed shot gathers are needed. Both programs solve the 2D acoustic wave equation with a finite-difference scheme that is second order in time and eighth order in space, with a hybrid Higdon absorbing boundary. The model is updated with a preconditioned conjugate-gradient method (PRP formula).

## 2. Geometry

| Quantity | Value |
| --- | --- |
| Target model | 461 × 141 grid points (4600 m × 1400 m) |
| Lateral edge extension | 20 points per side → 501 × 141 |
| Absorbing boundary | 34 points per side → 569 × 209 simulation grid |
| Grid spacing | 10 m |
| Water layer (not updated) | top 19 grid points (190 m) |
| Shots | 93, spacing 50 m, within the target region |
| Receivers | 501 fixed receivers, spacing 10 m (461 in the target region, 20 in each extension) |
| Source / receiver depth | 10 m |
| Recording | 4 s, 1 ms sampling (4001 samples) |

### Note on the receiver count

The manuscript describes the 461 receivers within the target region. To reduce boundary artifacts, the code extends the model by 20 edge-replicated grid points (200 m) on each side and places 20 additional receivers in each extension, giving 501 receivers per shot. All 501 traces are used in both inversion steps, and all results are displayed within the original 461 × 141 target region. The 34-point absorbing layer contains no receivers.

## 3. Parameters

All parameters are compile-time macros near the top of each `.cu` file. Edit them and recompile to change them. Parameters shared by both programs must be kept consistent in the two files.

### 3.1 Step 1 (`ti.cu`)

| Parameter | Value | Macro |
| --- | --- | --- |
| Time step / number of samples | 1 ms / 4001 (4 s) | `DT`, `NT` |
| Ricker peak frequency / delay | 10 Hz / 0.7 s | `F0`, `T0` |
| Integration passes | 6 | `N_TI_ITER` |
| Integration window length | 140 | `TI_TW_HALF = 70` (half window length) |
| Kaiser β | 6.0 (0 gives a rectangular window) | `TI_KAISER_BETA` |
| Window normalization | sum of weights = 1 | `TI_WIN_NORM` |
| Iterations | 50 | `K_ITER` |

The source wavelet and the observed data pass through the same integration, and both are then normalized by the peak amplitude of the integrated wavelet.

### 3.2 Step 2 (`tifwi.cu`)

| Stage | Ricker peak frequency | Iterations | Gradient smoothing | Wavelet low-pass |
| --- | ---: | ---: | --- | --- |
| 1 | 10 Hz | 50 | on | none |
| 2 | 20 Hz | 50 | off | none |
| 3 | 30 Hz | 50 | off | 35 Hz (5 Hz cosine half-width) |

The stages are defined by the arrays `STAGE_F0`, `STAGE_ITER`, `STAGE_SMOOTH`, `STAGE_LP_FC` and the macros `N_STAGES`, `LP_TAPER`. The Ricker delay is `T0 = 0.2 s`. Each stage starts from the result of the previous stage, re-synthesizes the observed data with that stage's wavelet, and restarts the conjugate gradient from steepest descent.

### 3.3 Parameters in both files

| Group | Macros |
| --- | --- |
| Model size | `NX_T`, `NZ_C`, `ACQ_EXT`, `NW`, `DX`, `DZ` |
| Absorbing boundary | `L_PAD`, `B_CAL`, `HIG_A`, `HIG_B` |
| Acquisition | `N_SHOTS`, `DSX`, `REC_EXT`, `DRX`, `J_SRC`, `J_REC` |
| Velocity bounds | `VMIN_CLAMP` (1500 m/s), `VMAX_CLAMP` (4700 m/s) |
| Gradient preconditioning | `PRECOND_ILLUM`, `ILLUM_USE_REC`, `ILLUM_EPS`, `PRECOND_VPOW`, `VPOW_P`, `SMOOTH_RADIUS`, `SMOOTH_SIGMA`; `PRECOND_SMOOTH` in `ti.cu` |
| Gradient mask | `MASK_TAPER_X`, `MASK_TAPER_Z` |
| Output | `SAVE_EVERY` (5), `SAVE_GRAD` (1 = also save gradients and illumination) |

## 4. Input files

Raw little-endian float32, no header, velocity in m/s, 501 (x) × 141 (z), depth index fastest: sample `(i, j)` is at `i * 141 + j` (0-based). Each file is 282,564 bytes. The absorbing boundary is filled by edge replication inside the programs.

- `model_bin/mar2_ext_501_141.bin`: true model, a portion of the Marmousi-II model (Martin et al., 2006) extended by 20 edge-replicated columns on each side.
- `model_bin/mar2_lin_501_141_4400.bin`: linear initial model.

Martin, G.S., Wiley, R., Marfurt, K.J., 2006. Marmousi2: An elastic upgrade for Marmousi. The Leading Edge 25 (2), 156–166.

## 5. Output files

```text
output/                        # ti
├── vp_inv_final.bin           # final TI-FWI model (input of tifwi)
├── vp_inv_iter_NN.bin         # model every 5 iterations
├── obs_data.bin               # observed data
├── obs_data_int.bin           # integrated, normalized observed data
├── wavelet_orig.bin           # Ricker wavelet (4001 samples)
├── wavelet_int.bin            # integrated, normalized wavelet
├── obj_history.bin            # objective function, one value per iteration
├── alpha_history.bin          # step length
├── beta_history.bin           # PRP coefficient
├── gradnorm_history.bin       # L2 norm of the raw gradient
└── grad_raw_iter_NN.bin, grad_prec_iter_NN.bin, illum_src_iter_NN.bin

output2/                       # tifwi
├── stage1_10Hz/, stage2_20Hz/, stage3_30Hz/
│   ├── obs_data.bin, wavelet.bin
│   ├── vp_inv_iter_NN.bin, vp_inv_final.bin
│   ├── obj_history.bin, alpha_history.bin, beta_history.bin, gradnorm_history.bin
│   ├── rmse_target_history.bin   # RMSE in the target region, water included
│   ├── rmse_rock_history.bin     # RMSE in the target region, water excluded
│   └── grad_raw_iter_NN.bin, grad_prec_iter_NN.bin, illum_src_iter_NN.bin
└── vp_inv_final.bin           # final model
```

| File type | Format |
| --- | --- |
| Velocity models, gradients, illumination | float32, 569 × 209 including the absorbing boundary, depth fastest (`i * 209 + j`), 475,684 bytes |
| Shot data | float32, ordered `[shot][time][receiver]` with receiver fastest, 93 × 4001 × 501 values (about 746 MB) |
| Wavelets | float32, 4001 values |
| History files | float32, one value per iteration |

All files are overwritten when a program is run again. Rename `output/` or `output2/` first if you want to keep earlier results.

## 6. Expected behaviour

**Step 1 (`./ti`)** prints the grid, acquisition and integration settings, the GPU name and free memory, and the memory it allocates. It then generates the observed data shot by shot, applies the trace integration to the wavelet and to the observed data, and runs 50 iterations. Each iteration prints one line:

```text
  iter  k/50  obj=...  alpha=...  beta=...  ||g||=...  (time s)
```

The objective function should decrease over the iterations. The final model `output/vp_inv_final.bin`, cropped to the target region, should recover the large-scale structure of the Marmousi-II model, as in Fig. 7(b) of the manuscript.

**Step 2 (`./tifwi`)** prints the stage settings and the RMSE of the initial model, then runs the three stages. Each iteration prints the objective function and the RMSE in the target region:

```text
  iter  k/50  obj=...  alpha=...  beta=...  ||g||=...  RMSE(target)=...  RMSE(rock)=...  (time s)
```

The RMSE should decrease within each stage. The final model `output2/vp_inv_final.bin` should reconstruct the detailed structure of the target region, as in Fig. 7(c) of the manuscript.

Small numerical differences from the manuscript figures are expected on different GPUs and CUDA versions, mainly because of `-use_fast_math`.

The progress line shows the estimated remaining time of each iteration. The water layer is not updated, and velocities are clipped to 1500–4700 m/s after every update.

If a program stops with `Cannot open ...`, check that it was started from the repository root and that the required input file exists. If it stops with a CUDA out-of-memory error, use a GPU with more memory.
