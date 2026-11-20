# Tutorial

This tutorial shows four typical uses of the code. Run all commands from the repository root. Parameter definitions are in the [User Guide](USER_GUIDE.md).

1. [Reading and plotting the results](#1-reading-and-plotting-the-results)
2. [Using your own velocity model](#2-using-your-own-velocity-model)
3. [Changing the integration parameters](#3-changing-the-integration-parameters)
4. [Changing the multiscale stages](#4-changing-the-multiscale-stages)

## 1. Reading and plotting the results

The output models are stored on the full 569 × 209 simulation grid, which includes the 34-point absorbing boundary and the 20-point lateral extensions. Remove both to obtain the 461 × 141 target region. MATLAB example:

```matlab
fid = fopen('output/vp_inv_final.bin', 'rb', 'ieee-le');   % or output2/vp_inv_final.bin
V = fread(fid, [209, 569], 'single=>single');               % rows = depth, columns = distance
fclose(fid);

V_target = V(35:175, 55:515);                               % 141 (z) x 461 (x)

imagesc((0:460) * 10, (0:140) * 10, V_target);
axis image; colormap(jet); colorbar;
xlabel('Distance (m)'); ylabel('Depth (m)');
```

The same indexing applies to intermediate models (e.g. `output/vp_inv_iter_05.bin`) and to the result of a single stage (e.g. `output2/stage1_10Hz/vp_inv_final.bin`). The history files (`obj_history.bin`, `rmse_rock_history.bin`, etc.) are plain float32 vectors with one value per iteration and can be read with `fread(fid, Inf, 'single')`.

## 2. Using your own velocity model

### 2.1 Prepare the model files

Each program expects the target model extended by `ACQ_EXT` (default 20) edge-replicated columns on each side, stored as raw little-endian float32 with depth fastest. Starting from a target model `v` of size `nz × nx` in m/s, in MATLAB:

```matlab
ACQ_EXT = 20;
v_ext = [repmat(v(:,1), 1, ACQ_EXT), v, repmat(v(:,end), 1, ACQ_EXT)];   % nz x (nx + 40)
fid = fopen('model_bin/my_true.bin', 'wb', 'ieee-le');
fwrite(fid, v_ext, 'float32');   % MATLAB is column-major, so depth is stored fastest
fclose(fid);
```

Prepare the initial model in the same way, e.g. `model_bin/my_init.bin`.

### 2.2 Edit the macros in both `.cu` files

Make the same changes in `ti.cu` and `tifwi.cu`:

| Macro | Set to |
| --- | --- |
| `NX_T` | number of columns `nx` of the target model |
| `NZ_C` | number of rows `nz` |
| `DX`, `DZ` | grid spacing (m) |
| `NW` | number of grid points in the water layer (not updated); 0 if there is none |
| `N_SHOTS`, `DSX` | number and spacing of shots; all shots must lie in the target region, i.e. `(N_SHOTS - 1) * DSX <= (NX_T - 1) * DX` |
| `DRX` | receiver spacing; keep equal to `DX` |
| `VMIN_CLAMP`, `VMAX_CLAMP` | velocity range of the model |
| `NT`, `DT` | number of time samples and time step |

The default configuration uses `VMAX_CLAMP * DT / DX = 0.47` with `DX = DZ`. This ratio is not a general stability limit. When changing `DT`, `DX`, `DZ` or the velocity range, check the stability condition of the 2D finite-difference scheme (second order in time, eighth order in space) for the new values.

### 2.3 Change the file paths

- `ti.cu`: replace `./model_bin/mar2_ext_501_141.bin` (true model) and `./model_bin/mar2_lin_501_141_4400.bin` (initial model) in the `read_model_padded` calls.
- `tifwi.cu`: replace `./model_bin/mar2_ext_501_141.bin` (true model) in the `read_model_padded` call.

### 2.4 Check the memory

GPU memory grows with the model size and recording length:

- stored wavefield: `(NX_T + 2*ACQ_EXT + 2*L_PAD) * (NZ_C + 2*L_PAD) * NT * 4` bytes
- each of the three shot-data buffers: `N_SHOTS * (NX_T + 2*REC_EXT) * NT * 4` bytes

Recompile both programs and run them as in the Quick start of the README.

## 3. Changing the integration parameters

The integration is applied only in step 1, so only `ti.cu` needs to be changed.

| Goal | Change |
| --- | --- |
| Longer or shorter window | `TI_TW_HALF` (half window length; window length is `2*TI_TW_HALF`) |
| More or fewer integration passes | `N_TI_ITER` |
| Rectangular window instead of Kaiser | `TI_KAISER_BETA 0.0` |
| Stronger or weaker sidelobe suppression | larger or smaller `TI_KAISER_BETA` |

For example, to compare the Kaiser and rectangular windows:

```bash
./ti                              # Kaiser window (default)
mv output output_kaiser

# edit ti.cu: #define TI_KAISER_BETA 0.0
nvcc -O3 -arch=sm_89 -use_fast_math ti.cu -o ti -lm
./ti                              # rectangular window
mv output output_rect
```

Then compare `output_kaiser/wavelet_int.bin` and `output_rect/wavelet_int.bin` (4001 samples each) and their spectra, as well as the final models and `obj_history.bin` of the two runs. Before running `./tifwi`, make sure the desired result is located at `output/vp_inv_final.bin`.

## 4. Changing the multiscale stages

The stages of step 2 are set by arrays near the top of `tifwi.cu`. For example, two stages with peak frequencies of 10 and 20 Hz and a 25 Hz low-pass in the second stage:

```c
#define N_STAGES     2
static const float STAGE_F0[N_STAGES]     = { 10.0f, 20.0f };
static const int   STAGE_ITER[N_STAGES]   = { 50,    50    };
static const int   STAGE_SMOOTH[N_STAGES] = { 1,     0     };
static const float STAGE_LP_FC[N_STAGES]  = { 0.0f,  25.0f };
```

- `STAGE_SMOOTH = 1` enables Gaussian smoothing of the gradient in that stage.
- `STAGE_LP_FC` is the center frequency of the zero-phase low-pass applied to the wavelet; 0 disables it. The transition half-width is `LP_TAPER` (5 Hz).
- Output directories are named after the peak frequency, e.g. `output2/stage2_20Hz/`.

Higher peak frequencies require a finer grid (`DX`, `DZ`) to avoid numerical dispersion.
