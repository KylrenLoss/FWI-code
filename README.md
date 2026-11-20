# FWI-code: Kaiser-Windowed Local Time-Window Integration for Full Waveform Inversion

CUDA C++ implementation and a full-bandwidth Marmousi-II synthetic example for the manuscript:

> Xie, C., You, X., Yu, S., Song, P., Wang, S., Wang, X., Qin, Z. *Full Waveform Inversion Based on Kaiser-Windowed Local Time-Window Integration*. Submitted to *Computers & Geosciences*.

The example runs two inversion steps in sequence:

1. **`ti.cu` — Kaiser-windowed TI-FWI.** Inverts the integrated data starting from a linear velocity model and produces a background velocity model.
2. **`tifwi.cu` — multiscale conventional FWI.** Uses the result of step 1 as the initial model and refines it in three frequency stages.

The two programs are standalone; `tifwi.cu` only reads the output file of `ti.cu`.

## Scope

This repository reproduces the full-bandwidth Marmousi-II experiment in Section 3.1 of the manuscript, i.e. the results in **Fig. 7(b)** (TI-FWI) and **Fig. 7(c)** (conventional FWI initialized with the TI-FWI result).

Not included: the conventional-FWI baseline from the linear model (Fig. 7(a)), the experiment with frequencies below 4 Hz removed (Section 3.2), and the South China Sea field-data experiment (Section 4). The field data are proprietary and cannot be redistributed.

## Repository layout

```text
FWI-code/
├── README.md
├── LICENSE
├── .gitignore                 # excludes output/ and output2/
├── ti.cu
├── tifwi.cu
└── model_bin/
    ├── mar2_ext_501_141.bin       # true model (used to synthesize observed data)
    └── mar2_lin_501_141_4400.bin  # linear initial model for ti.cu
```

`output/` and `output2/` are created at runtime. Input and output paths are relative and hard-coded, so **run both programs from the repository root**.

## Requirements

- Linux (the code uses POSIX `mkdir`; native Windows builds need minor changes)
- NVIDIA GPU; the main allocations are one stored wavefield of about 1.9 GB (569 × 209 × 4001 float32) and three shot-data buffers of about 0.75 GB each (93 × 4001 × 501 float32), so a GPU with 8 GB or more of memory is recommended
- NVIDIA driver and CUDA Toolkit (`nvcc`) with a supported host C/C++ compiler; the code was developed and tested with CUDA 11.8
- Several GB of free disk space for the generated outputs

## Compilation

```bash
nvcc -O3 -arch=sm_89 -use_fast_math ti.cu    -o ti    -lm
nvcc -O3 -arch=sm_89 -use_fast_math tifwi.cu -o tifwi -lm
```

`sm_89` targets Ada Lovelace GPUs. Replace it with the compute capability of your GPU (e.g. `sm_80` for A100, `sm_86` for RTX 30-series). `-use_fast_math` may cause small numerical differences across GPUs and CUDA versions.

## Running the example

### Step 1: TI-FWI

```bash
./ti
```

Settings (defined as macros at the top of `ti.cu`):

| Parameter | Value | Macro |
| --- | --- | --- |
| Time step / number of samples | 1 ms / 4001 (4 s) | `DT`, `NT` |
| Ricker peak frequency / delay | 10 Hz / 0.7 s | `F0`, `T0` |
| Integration passes | 6 | `N_TI_ITER` |
| Integration window | 141 samples (140 ms) | `TI_TW_HALF = 70` |
| Kaiser β | 6.0 | `TI_KAISER_BETA` |
| Window normalization | sum of weights = 1 | `TI_WIN_NORM` |
| Iterations | 50 | `K_ITER` |

The source wavelet and the observed data are passed through the same Kaiser-windowed integration before inversion. The final model is written to `output/vp_inv_final.bin` and is required by step 2.

### Step 2: multiscale conventional FWI

```bash
./tifwi
```

`tifwi` reads `output/vp_inv_final.bin` and runs three stages; each stage starts from the previous result and re-synthesizes the observed data from the true model with that stage's wavelet (delay `T0 = 0.2 s`). No trace integration is applied in this step.

| Stage | Ricker peak frequency | Iterations | Gradient smoothing | Wavelet low-pass |
| --- | ---: | ---: | --- | --- |
| 1 | 10 Hz | 50 | on | none |
| 2 | 20 Hz | 50 | off | none |
| 3 | 30 Hz | 50 | off | 35 Hz (5 Hz cosine half-width) |

The final model is written to `output2/vp_inv_final.bin`.

## Geometry

| Quantity | Value |
| --- | --- |
| Target model | 461 × 141 grid points (4600 m × 1400 m) |
| Lateral edge extension | 20 points per side → 501 × 141 |
| Absorbing boundary | 34 points per side → 569 × 209 simulation grid |
| Grid spacing | 10 m |
| Shots | 93, spacing 50 m, within the target region |
| Receivers | 501 fixed receivers, spacing 10 m (461 in the target region, 20 in each extension) |
| Source / receiver depth | 10 m |
| Finite-difference scheme | 2nd order in time, 8th order in space; hybrid Higdon absorbing boundary |

All 501 traces are used in both steps. The lateral extension is part of the physical model; the 34-point absorbing layer contains no receivers.

## Configuration

All parameters are compile-time macros near the top of each `.cu` file; edit and recompile to change them. The most relevant ones:

- Integration: `N_TI_ITER`, `TI_TW_HALF`, `TI_KAISER_BETA`, `TI_WIN_NORM` (`ti.cu`)
- Inversion: `K_ITER`, `VMIN_CLAMP`, `VMAX_CLAMP`, `SAVE_EVERY`, `SAVE_GRAD`
- Gradient preconditioning: `PRECOND_ILLUM`, `ILLUM_USE_REC`, `ILLUM_EPS`, `PRECOND_VPOW`, `VPOW_P`, `PRECOND_SMOOTH`, `SMOOTH_RADIUS`, `SMOOTH_SIGMA`
- Multiscale stages: `N_STAGES`, `STAGE_F0`, `STAGE_ITER`, `STAGE_SMOOTH`, `STAGE_LP_FC`, `LP_TAPER` (`tifwi.cu`)
- Acquisition: `N_SHOTS`, `DSX`, `REC_EXT`, `DRX`

Setting `TI_KAISER_BETA = 0` gives a rectangular window, which can be used to compare window functions.

## Input and output formats

### Input models

Raw little-endian float32, no header, velocity in m/s, 501 (x) × 141 (z), depth index fastest: sample `(i, j)` is at `i * 141 + j` (0-based). Each file is 282,564 bytes. The absorbing boundary is filled by edge replication inside the programs.

The true model is a portion of the Marmousi-II model (Martin et al., 2006), extended by 20 edge-replicated columns on each side. The initial model is a linear velocity model.

Martin, G.S., Wiley, R., Marfurt, K.J., 2006. Marmousi2: An elastic upgrade for Marmousi. The Leading Edge 25 (2), 156–166.

### Outputs

```text
output/                        # ti
├── vp_inv_final.bin           # final TI-FWI model (input of tifwi)
├── vp_inv_iter_NN.bin         # model every 5 iterations
├── obs_data.bin               # observed data
├── obs_data_int.bin           # integrated, normalized observed data
├── wavelet_orig.bin, wavelet_int.bin
├── obj_history.bin            # objective function, 50 values
└── grad_*, illum_*, alpha/beta/gradnorm_history.bin

output2/                       # tifwi
├── stage1_10Hz/, stage2_20Hz/, stage3_30Hz/
│   ├── vp_inv_final.bin
│   ├── obj_history.bin
│   ├── rmse_target_history.bin   # RMSE in the target region, water included
│   └── rmse_rock_history.bin     # RMSE in the target region, water excluded
└── vp_inv_final.bin           # final model
```

- Velocity models: float32, 569 × 209 including the absorbing boundary, depth fastest (`i * 209 + j`), 475,684 bytes.
- Shot data: float32, ordered `[shot][time][receiver]` with receiver fastest, 93 × 4001 × 501 values (about 746 MB).
- Other diagnostic files are listed in the header comment of each `.cu` file.

## Plotting the target region

Remove the 34-point absorbing boundary and the 20-point lateral extensions to obtain the 461 × 141 target region. MATLAB example:

```matlab
fid = fopen('output/vp_inv_final.bin', 'rb', 'ieee-le');   % or output2/vp_inv_final.bin
V = fread(fid, [209, 569], 'single=>single');
fclose(fid);

V_target = V(35:175, 55:515);   % 141 (z) x 461 (x)

imagesc((0:460) * 10, (0:140) * 10, V_target);
axis image; colorbar;
xlabel('Distance (m)'); ylabel('Depth (m)');
```

`output/vp_inv_final.bin` corresponds to Fig. 7(b) and `output2/vp_inv_final.bin` to Fig. 7(c).

## License

This code is released under the MIT License, see `LICENSE`.

## Citation

If you use this code, please cite the manuscript above (DOI to be added on publication).

## Contact

- Shengqi Yu: yushengqi@hrbeu.edu.cn
- Zhiliang Qin: qin_zhiliang@126.com
