# FWI-code: Kaiser-Windowed Local Time-Window Integration for Full Waveform Inversion

CUDA C++ implementation and a full-bandwidth Marmousi-II synthetic example for the manuscript:

> Xie, C., You, X., Yu, S., Song, P., Wang, S., Wang, X., Qin, Z. *Full Waveform Inversion Based on Kaiser-Windowed Local Time-Window Integration*. Submitted to *Computers & Geosciences*.

The example runs two inversion steps in sequence:

1. **`ti.cu` — Kaiser-windowed TI-FWI.** Inverts the integrated data starting from a linear velocity model and produces a background velocity model.
2. **`tifwi.cu` — multiscale conventional FWI.** Uses the result of step 1 as the initial model and refines it in three frequency stages.

The two programs are standalone; `tifwi.cu` only reads the output file of `ti.cu`.

## Documentation

- [User Guide](docs/USER_GUIDE.md): acquisition geometry, input and output formats, all configurable parameters, and expected behaviour.
- [Tutorial](docs/TUTORIAL.md): reading and plotting the results, using your own velocity model, changing the integration parameters, and changing the multiscale stages.

## Scope

This repository reproduces the full-bandwidth Marmousi-II experiment in Section 3.1 of the manuscript, i.e. the results in **Fig. 7(b)** (TI-FWI) and **Fig. 7(c)** (conventional FWI initialized with the TI-FWI result).

Not included: the conventional-FWI baseline from the linear model (Fig. 7(a)), the experiment with frequencies below 4 Hz removed (Section 3.2), and the South China Sea field-data experiment (Section 4). The field data are proprietary and cannot be redistributed.

The code places 501 receivers per shot, whereas the manuscript describes the 461 receivers within the target region; see [Note on the receiver count](docs/USER_GUIDE.md#note-on-the-receiver-count).

## Repository layout

```text
FWI-code/
├── README.md
├── LICENSE
├── .gitignore                     # excludes output/ and output2/
├── ti.cu                          # step 1: Kaiser-windowed TI-FWI
├── tifwi.cu                       # step 2: multiscale conventional FWI
├── model_bin/
│   ├── mar2_ext_501_141.bin       # true model (used to synthesize observed data)
│   └── mar2_lin_501_141_4400.bin  # linear initial model for ti.cu
└── docs/
    ├── USER_GUIDE.md
    └── TUTORIAL.md
```

`output/` and `output2/` are created at runtime. Input and output paths are relative and hard-coded, so **run both programs from the repository root**.

## Requirements

- Linux (the code uses POSIX `mkdir`; native Windows builds need minor changes)
- NVIDIA GPU; the main allocations are one stored wavefield of about 1.9 GB and three shot-data buffers of about 0.75 GB each, so a GPU with 8 GB or more of memory is recommended
- NVIDIA driver and CUDA Toolkit (`nvcc`) with a supported host C/C++ compiler; the code was developed and tested with CUDA 11.8
- Several GB of free disk space for the generated outputs

## Compilation

```bash
nvcc -O3 -arch=sm_89 -use_fast_math ti.cu    -o ti    -lm
nvcc -O3 -arch=sm_89 -use_fast_math tifwi.cu -o tifwi -lm
```

`sm_89` targets Ada Lovelace GPUs. Replace it with the compute capability of your GPU (e.g. `sm_80` for A100, `sm_86` for RTX 30-series). `-use_fast_math` may cause small numerical differences across GPUs and CUDA versions.

## Quick start

```bash
./ti        # step 1, writes output/vp_inv_final.bin
./tifwi     # step 2, reads output/vp_inv_final.bin, writes output2/vp_inv_final.bin
```

Run `./tifwi` only after `./ti` has finished. Main settings of step 1: 10 Hz Ricker wavelet, 6 Kaiser-windowed integration passes, window length 140, Kaiser β = 6.0, 50 iterations. Step 2 runs three stages with 10, 20 and 30 Hz Ricker wavelets, 50 iterations each, without trace integration. Full parameter lists are in the [User Guide](docs/USER_GUIDE.md).

To view the results, see [Reading and plotting the results](docs/TUTORIAL.md#1-reading-and-plotting-the-results).

## License

This code is released under the MIT License, see `LICENSE`.

## Citation

If you use this code, please cite the manuscript above (DOI to be added on publication).

## Contact

- Shengqi Yu: yushengqi@hrbeu.edu.cn
- Zhiliang Qin: qin_zhiliang@126.com
