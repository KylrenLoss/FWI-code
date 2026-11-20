/*
 * ti.cu - Trace-Integration Full Waveform Inversion (TI-FWI), CUDA implementation
 * ------------------------------------------------------------------------
 * The source wavelet and the observed data are passed through the same number
 * of Kaiser-weighted sliding-window trace integrations to boost low frequencies,
 * followed by acoustic FWI to mitigate cycle skipping. The final model can be
 * used as the initial model of tifwi.cu.
 *
 * Inputs (./model_bin/, float32, NX_C x NZ_C, depth fastest):
 *   mar2_ext_501_141.bin        true model (used to synthesize observed data)
 *   mar2_lin_501_141_4400.bin   initial model
 *
 * Outputs (./output/, float32):
 *   obs_data.bin              raw observed data, stored as [shot][time][receiver],
 *                             receiver index fastest
 *   obs_data_int.bin          trace-integrated, normalized observed data (used in inversion)
 *   wavelet_orig.bin          raw Ricker wavelet (NT)
 *   wavelet_int.bin           trace-integrated, normalized wavelet (used in inversion)
 *   vp_inv_iter_NN.bin        velocity model every SAVE_EVERY iterations
 *                             (extended grid NZ_E x NX_E, incl. padding)
 *   vp_inv_final.bin          final velocity model (same layout)
 *   grad_raw_iter_NN.bin      raw gradient, hard mask only  (SAVE_GRAD = 1)
 *   grad_prec_iter_NN.bin     preconditioned gradient       (SAVE_GRAD = 1)
 *   illum_src_iter_NN.bin     source-side illumination      (SAVE_GRAD = 1 and PRECOND_ILLUM = 1)
 *   obj_history.bin           objective function            (K_ITER)
 *   alpha_history.bin         step length                   (K_ITER)
 *   beta_history.bin          PRP coefficient               (K_ITER)
 *   gradnorm_history.bin      L2 norm of the raw gradient   (K_ITER)
 *
 * Build: nvcc -O3 -arch=sm_89 -use_fast_math ti.cu -o ti -lm
 * Run:   ./ti
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>

/* ============================= Parameters ============================= */

/* ---- Grid and model ----
 * Model files are little-endian float32 holding the physical region only
 * (NX_C x NZ_C), depth fastest (v[i*NZ_C + j]); the absorbing-boundary
 * padding is filled by edge replication in this program. */
#define NX_T    461                  /* lateral grid points of the target region */
#define NZ_C    141                  /* vertical grid points of the physical region */
#define ACQ_EXT 20                   /* replicated grid points on each side of the target */
#define NX_C    (NX_T + 2*ACQ_EXT)   /* = 501, lateral grid points of the physical region */
#define NW      19                   /* water-layer grid points (physical j = 0..18) */

/* ---- Absorbing boundary: L_PAD layers on each side ----
 * The outermost N_FD layers are kept at zero; the B_CAL layers inside them
 * form the FD / Higdon blending zone. */
#define L_PAD   34
#define B_CAL   30
#define N_FD    4                    /* half-width of the 8th-order stencil */

#define NX_E    (NX_C + 2*L_PAD)     /* = 569           */
#define NZ_E    (NZ_C + 2*L_PAD)     /* = 209           */
#define NXZ     ((size_t)NX_E * NZ_E)/* = 118921        */

#define DX      10.0f
#define DZ      10.0f
#define DT      0.001f
#define NT      4001                /* number of time steps (4 s) */

#define F0      10.0f               /* Ricker peak frequency (Hz) */
#define T0      0.7f                /* Ricker time delay (s) */

/* ---- Acquisition: fixed full-spread receivers shared by all shots ----
 * i is the column index on the extended grid; the target region is i = 54 .. 514:
 *   receivers i = I_REC0 .. I_REC0 + N_RECS - 1 = 34 .. 534 (501 traces)
 *   sources   i = I_ACQ0 + round(s*DSX/DX)      = 54 .. 514 (93 shots) */
#define N_SHOTS  93
#define DSX      50.0f              /* shot spacing (m) */
#define REC_EXT  20                 /* receiver extension beyond each side of the target */
#define N_RECS   (NX_T + 2*REC_EXT) /* = 501                              */
#define DRX      10.0f              /* receiver spacing (m) */
#define I_ACQ0   (L_PAD + ACQ_EXT)  /* = 54, left column of the target region */
#define I_REC0   (I_ACQ0 - REC_EXT) /* = 34, left column of the receiver spread */

#if REC_EXT > ACQ_EXT
#error "REC_EXT must not exceed ACQ_EXT: receivers would fall inside the absorbing boundary"
#endif
/* Source and receiver depth: physical j = 1 (z = 10 m) */
#define J_SRC    (1 + L_PAD)
#define J_REC    (1 + L_PAD)

#define K_ITER       50             /* number of iterations */
#define SAVE_EVERY   5              /* save the model every SAVE_EVERY iterations */

#define SAVE_GRAD    1              /* 1 = also save gradient and illumination */

/* Velocity bounds applied after each model update (m/s) */
#define VMIN_CLAMP   1500.0f
#define VMAX_CLAMP   4700.0f

/* ---- Trace integration: N_TI_ITER passes, window length 2*TI_TW_HALF+1 = 141 samples ---- */
#define N_TI_ITER    6
#define TI_TW_HALF   70
#define TI_KAISER_BETA  6.0         /* Kaiser window beta, 0 = rectangular */
#define TI_WIN_NORM     1           /* 1 = normalize window to sum(h) = 1 */

/* Higdon coefficients (alfa1 = alfa2 = 0): A = 2, B = 1 */
#define HIG_A   2.0f
#define HIG_B   1.0f

/* ---- Gradient preconditioning (applied in order) ----
 *   (1) illumination normalization: g <- g / (I + ILLUM_EPS),
 *       I = source-side sum(p^2) x receiver-side sum(lambda^2),
 *       each normalized by its max(I*M) (M is the gradient mask)
 *   (2) v^p weighting: g <- g * (v / VMIN_CLAMP)^VPOW_P
 *   (3) separable Gaussian smoothing inside the mask (normalized convolution) */
#define PRECOND_ILLUM    1
#define ILLUM_USE_REC    1          /* 1 = source x receiver; 0 = source only */
#define ILLUM_EPS        1e-3f      /* stabilization constant */
#define PRECOND_VPOW     1
#define VPOW_P           1.0f
#define PRECOND_SMOOTH   1
#define SMOOTH_RADIUS    3          /* Gaussian kernel radius (grid points) */
#define SMOOTH_SIGMA     1.5f       /* Gaussian sigma (grid points) */

/* ---- Cosine taper width of the gradient mask (grid points, 0 = no taper) ---- */
#define MASK_TAPER_Z     0          /* vertical taper below the sea floor */
#define MASK_TAPER_X     20         /* lateral taper at both ends of the physical region */

#define IDX(i,j)  ((size_t)(i)*NZ_E + (j))  /* column-major, z fastest (same as MATLAB) */

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error: %s @ %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
    exit(1);} } while(0)

/* =========================== Host utilities =========================== */
static void make_dir(const char* p) {
    struct stat st = {0};
    if (stat(p, &st) == -1) mkdir(p, 0700);
}

static void read_bin(const char* path, float* h_data, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    size_t r = fread(h_data, sizeof(float), n, fp);
    if (r != n) { fprintf(stderr, "Short read %s: %zu/%zu\n", path, r, n); exit(1); }
    fclose(fp);
}

static void write_bin(const char* path, const float* h_data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fwrite(h_data, sizeof(float), n, fp);
    fclose(fp);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void show_progress(const char* prefix, int cur, int total, double elapsed) {
    int width = 28;
    float frac = (total > 0) ? (float)cur / total : 0.0f;
    int filled = (int)(frac * width);
    printf("\r%s [", prefix);
    for (int i = 0; i < width; i++) {
        if (i < filled) putchar('=');
        else if (i == filled) putchar('>');
        else putchar(' ');
    }
    double eta = (frac > 1e-6) ? elapsed * (1.0 - frac) / frac : 0.0;
    printf("] %5.1f%% (%d/%d) | %.1fs / ETA %.1fs    ",
           frac*100, cur, total, elapsed, eta);
    fflush(stdout);
}

/* ============================ CUDA kernels ============================ */

/* 8th-order spatial FD time step; the outermost N_FD layers are set to zero;
 * FD values inside the absorbing band are also stored in u1_* */
__global__ void k_forward(float* __restrict__ Pn,
                          const float* __restrict__ Pc,
                          const float* __restrict__ Po,
                          const float* __restrict__ v,
                          float* u1_l,  float* u1_r,  float* u1_u,  float* u1_d,
                          float* u1_lu, float* u1_ru, float* u1_ld, float* u1_rd)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX_E || j >= NZ_E) return;
    size_t idx = IDX(i,j);

    if (i < N_FD || i >= NX_E-N_FD || j < N_FD || j >= NZ_E-N_FD) {
        Pn[idx] = 0.0f;
        return;
    }

    const float c0 = -205.0f/72.0f;
    const float c1 =  8.0f/5.0f;
    const float c2 = -1.0f/5.0f;
    const float c3 =  8.0f/315.0f;
    const float c4 = -1.0f/560.0f;

    float pc = Pc[idx];

    float pxx = c0*pc
              + c1*(Pc[IDX(i+1,j)] + Pc[IDX(i-1,j)])
              + c2*(Pc[IDX(i+2,j)] + Pc[IDX(i-2,j)])
              + c3*(Pc[IDX(i+3,j)] + Pc[IDX(i-3,j)])
              + c4*(Pc[IDX(i+4,j)] + Pc[IDX(i-4,j)]);
    pxx /= (DX*DX);

    float pzz = c0*pc
              + c1*(Pc[IDX(i,j+1)] + Pc[IDX(i,j-1)])
              + c2*(Pc[IDX(i,j+2)] + Pc[IDX(i,j-2)])
              + c3*(Pc[IDX(i,j+3)] + Pc[IDX(i,j-3)])
              + c4*(Pc[IDX(i,j+4)] + Pc[IDX(i,j-4)]);
    pzz /= (DZ*DZ);

    float v2dt2 = v[idx] * v[idx] * DT * DT;
    float pn = 2.0f*pc - Po[idx] + v2dt2 * (pxx + pzz);
    Pn[idx] = pn;

    if (i <= L_PAD-1 && j >= L_PAD && j < NZ_E-L_PAD)
        u1_l[NZ_C*(i - N_FD) + j - L_PAD] = pn;
    if (i >= NX_E-L_PAD && j >= L_PAD && j < NZ_E-L_PAD)
        u1_r[NZ_C*(i - (NX_E-L_PAD)) + j - L_PAD] = pn;
    if (j <= L_PAD-1 && i >= L_PAD && i < NX_E-L_PAD)
        u1_u[B_CAL*(i - L_PAD) + j - N_FD] = pn;
    if (j >= NZ_E-L_PAD && i >= L_PAD && i < NX_E-L_PAD)
        u1_d[B_CAL*(i - L_PAD) + j - (NZ_E-L_PAD)] = pn;
    if (i <= L_PAD-1 && j <= L_PAD-1)
        u1_lu[B_CAL*(i - N_FD) + j - N_FD] = pn;
    if (i >= NX_E-L_PAD && j <= L_PAD-1)
        u1_ru[B_CAL*(i - (NX_E-L_PAD)) + j - N_FD] = pn;
    if (i <= L_PAD-1 && j >= NZ_E-L_PAD)
        u1_ld[B_CAL*(i - N_FD) + j - (NZ_E-L_PAD)] = pn;
    if (i >= NX_E-L_PAD && j >= NZ_E-L_PAD)
        u1_rd[B_CAL*(i - (NX_E-L_PAD)) + j - (NZ_E-L_PAD)] = pn;
}

/* ------------------ 2nd-order Higdon hybrid absorbing boundary ------------------
 * Each time step consists of three kernels:
 *   k_forward  interior FD update; FD solution in the absorbing band -> u1_*
 *   k_higdon   Higdon solution in the absorbing band -> u2_*
 *   k_hybrid   P = alfa*u1 + (1-alfa)*u2
 * One-way Higdon scheme (left side as an example, c = v*dt/dx):
 *   P^{k+1}[i] = (2 - A*c - B*c^2) * P^k[i] + c*(A + 2*B*c) * P^k[i+1]
 *              - B*c^2 * P^k[i+2] + (A*c - 1) * P^{k-1}[i] - A*c * P^{k-1}[i+1]
 * Corner regions use the average of the x- and z-direction results.
 * ------------------------------------------------------------------------------ */

/* One-way Higdon: p0/q0 are this point at steps k / k-1; p1, p2, q1 are neighbours toward the interior */
__device__ static inline float hig1(float c,
                                    float p0, float p1, float p2,
                                    float q0, float q1)
{
    return (2.0f - HIG_A*c - HIG_B*c*c) * p0
         + c * (HIG_A + 2.0f*HIG_B*c)   * p1
         - HIG_B*c*c                    * p2
         + (HIG_A*c - 1.0f)             * q0
         - HIG_A*c                      * q1;
}

/* Higdon solution in the absorbing band (four sides + four corners), written to u2_* */
__global__ void k_higdon(const float* __restrict__ Pc,
                         const float* __restrict__ Po,
                         const float* __restrict__ v,
                         float* u2_l,  float* u2_r,  float* u2_u,  float* u2_d,
                         float* u2_lu, float* u2_ru, float* u2_ld, float* u2_rd)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX_E || j >= NZ_E) return;

    size_t idx = IDX(i,j);
    float cx = v[idx] * DT / DX;
    float cz = v[idx] * DT / DZ;

    /* ---- left ---- */
    if (i >= L_PAD-B_CAL && i <= L_PAD-1 && j >= L_PAD && j < NZ_C+L_PAD) {
        u2_l[NZ_C*(i - N_FD) + j - L_PAD] =
            hig1(cx, Pc[idx], Pc[IDX(i+1,j)], Pc[IDX(i+2,j)],
                     Po[idx], Po[IDX(i+1,j)]);
    }
    /* ---- right ---- */
    if (i >= NX_C+L_PAD && i < NX_C+L_PAD+B_CAL && j >= L_PAD && j < NZ_C+L_PAD) {
        u2_r[NZ_C*(i - (NX_E-L_PAD)) + j - L_PAD] =
            hig1(cx, Pc[idx], Pc[IDX(i-1,j)], Pc[IDX(i-2,j)],
                     Po[idx], Po[IDX(i-1,j)]);
    }
    /* ---- top ---- */
    if (i >= L_PAD && i < NX_C+L_PAD && j >= L_PAD-B_CAL && j <= L_PAD-1) {
        u2_u[B_CAL*(i - L_PAD) + j - N_FD] =
            hig1(cz, Pc[idx], Pc[IDX(i,j+1)], Pc[IDX(i,j+2)],
                     Po[idx], Po[IDX(i,j+1)]);
    }
    /* ---- bottom ---- */
    if (i >= L_PAD && i < NX_C+L_PAD && j >= NZ_C+L_PAD && j < NZ_C+L_PAD+B_CAL) {
        u2_d[B_CAL*(i - L_PAD) + j - (NZ_E-L_PAD)] =
            hig1(cz, Pc[idx], Pc[IDX(i,j-1)], Pc[IDX(i,j-2)],
                     Po[idx], Po[IDX(i,j-1)]);
    }
    /* ---- top-left corner ---- */
    if (i >= N_FD && i <= L_PAD-1 && j >= N_FD && j <= L_PAD-1) {
        u2_lu[B_CAL*(i - N_FD) + j - N_FD] = 0.5f *
            ( hig1(cx, Pc[idx], Pc[IDX(i+1,j)], Pc[IDX(i+2,j)],
                       Po[idx], Po[IDX(i+1,j)])
            + hig1(cz, Pc[idx], Pc[IDX(i,j+1)], Pc[IDX(i,j+2)],
                       Po[idx], Po[IDX(i,j+1)]) );
    }
    /* ---- top-right corner ---- */
    if (i >= NX_C+L_PAD && i < NX_C+L_PAD+B_CAL && j >= N_FD && j <= L_PAD-1) {
        u2_ru[B_CAL*(i - (NX_E-L_PAD)) + j - N_FD] = 0.5f *
            ( hig1(cx, Pc[idx], Pc[IDX(i-1,j)], Pc[IDX(i-2,j)],
                       Po[idx], Po[IDX(i-1,j)])
            + hig1(cz, Pc[idx], Pc[IDX(i,j+1)], Pc[IDX(i,j+2)],
                       Po[idx], Po[IDX(i,j+1)]) );
    }
    /* ---- bottom-left corner ---- */
    if (i >= N_FD && i <= L_PAD-1 && j >= NZ_C+L_PAD && j < NZ_C+L_PAD+B_CAL) {
        u2_ld[B_CAL*(i - N_FD) + j - (NZ_E-L_PAD)] = 0.5f *
            ( hig1(cx, Pc[idx], Pc[IDX(i+1,j)], Pc[IDX(i+2,j)],
                       Po[idx], Po[IDX(i+1,j)])
            + hig1(cz, Pc[idx], Pc[IDX(i,j-1)], Pc[IDX(i,j-2)],
                       Po[idx], Po[IDX(i,j-1)]) );
    }
    /* ---- bottom-right corner ---- */
    if (i >= NX_C+L_PAD && i < NX_C+L_PAD+B_CAL && j >= NZ_C+L_PAD && j < NZ_C+L_PAD+B_CAL) {
        u2_rd[B_CAL*(i - (NX_E-L_PAD)) + j - (NZ_E-L_PAD)] = 0.5f *
            ( hig1(cx, Pc[idx], Pc[IDX(i-1,j)], Pc[IDX(i-2,j)],
                       Po[idx], Po[IDX(i-1,j)])
            + hig1(cz, Pc[idx], Pc[IDX(i,j-1)], Pc[IDX(i,j-2)],
                       Po[idx], Po[IDX(i,j-1)]) );
    }
}

/* Blend inside the absorbing band: P = alfa*u1(FD) + (1-alfa)*u2(Higdon) */
__global__ void k_hybrid(float* Pn, const float* __restrict__ alfa,
                         const float* u1_l,  const float* u1_r,
                         const float* u1_u,  const float* u1_d,
                         const float* u1_lu, const float* u1_ru,
                         const float* u1_ld, const float* u1_rd,
                         const float* u2_l,  const float* u2_r,
                         const float* u2_u,  const float* u2_d,
                         const float* u2_lu, const float* u2_ru,
                         const float* u2_ld, const float* u2_rd)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX_E || j >= NZ_E) return;

    size_t idx = IDX(i,j);
    float a = alfa[idx];
    int p;

    if (i >= N_FD && i <= L_PAD-1 && j >= L_PAD && j < NZ_E-L_PAD) {
        p = NZ_C*(i - N_FD) + j - L_PAD;
        Pn[idx] = u1_l[p]*a + u2_l[p]*(1.0f - a);
    }
    if (i >= NX_E-L_PAD && i < NX_E-N_FD && j >= L_PAD && j < NZ_E-L_PAD) {
        p = NZ_C*(i - (NX_E-L_PAD)) + j - L_PAD;
        Pn[idx] = u1_r[p]*a + u2_r[p]*(1.0f - a);
    }
    if (j >= N_FD && j <= L_PAD-1 && i >= L_PAD && i < NX_E-L_PAD) {
        p = B_CAL*(i - L_PAD) + j - N_FD;
        Pn[idx] = u1_u[p]*a + u2_u[p]*(1.0f - a);
    }
    if (j >= NZ_E-L_PAD && j < NZ_E-N_FD && i >= L_PAD && i < NX_E-L_PAD) {
        p = B_CAL*(i - L_PAD) + j - (NZ_E-L_PAD);
        Pn[idx] = u1_d[p]*a + u2_d[p]*(1.0f - a);
    }
    if (i >= N_FD && i <= L_PAD-1 && j >= N_FD && j <= L_PAD-1) {
        p = B_CAL*(i - N_FD) + j - N_FD;
        Pn[idx] = u1_lu[p]*a + u2_lu[p]*(1.0f - a);
    }
    if (i >= NX_E-L_PAD && i < NX_E-N_FD && j >= N_FD && j <= L_PAD-1) {
        p = B_CAL*(i - (NX_E-L_PAD)) + j - N_FD;
        Pn[idx] = u1_ru[p]*a + u2_ru[p]*(1.0f - a);
    }
    if (i >= N_FD && i <= L_PAD-1 && j >= NZ_E-L_PAD && j < NZ_E-N_FD) {
        p = B_CAL*(i - N_FD) + j - (NZ_E-L_PAD);
        Pn[idx] = u1_ld[p]*a + u2_ld[p]*(1.0f - a);
    }
    if (i >= NX_E-L_PAD && i < NX_E-N_FD && j >= NZ_E-L_PAD && j < NZ_E-N_FD) {
        p = B_CAL*(i - (NX_E-L_PAD)) + j - (NZ_E-L_PAD);
        Pn[idx] = u1_rd[p]*a + u2_rd[p]*(1.0f - a);
    }
}

/* Inject point source: P += v^2 * dt^2 * w(t) */
__global__ void k_add_source(float* P, const float* v, const float* wavelet,
                             size_t idx_src, int t)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float vsrc = v[idx_src];
        P[idx_src] += vsrc * vsrc * DT * DT * wavelet[t];
    }
}

/* Record the wavefield at receivers (traces with index < 0 are set to 0) */
__global__ void k_record(const float* P, float* trace, const int* idx_recs)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N_RECS) return;
    int ir = idx_recs[r];
    trace[r] = (ir >= 0) ? P[ir] : 0.0f;
}

/* Inject residuals at receivers (adjoint source) */
__global__ void k_inject_residual(float* P, const float* v, const float* res,
                                   const int* idx_recs)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N_RECS) return;
    int ir = idx_recs[r];
    if (ir >= 0) {
        float vrec = v[ir];
        atomicAdd(&P[ir], vrec * vrec * DT * DT * res[r]);
    }
}

/* Store the full wavefield of time step t */
__global__ void k_save_slice(float* wavefield, const float* P, int t)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NXZ) return;
    wavefield[(size_t)t * NXZ + idx] = P[idx];
}

/* c = a - b */
__global__ void k_subtract(float* c, const float* a, const float* b, size_t n)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) c[t] = a[t] - b[t];
}

/* Gradient cross-correlation: g += (2/v^3) * d2P/dt2 * lambda, where d2P/dt2 is a
 * finite difference of three consecutive forward wavefields; with PRECOND_ILLUM,
 * illumination is accumulated as well: I_src += p^2, I_rec += lambda^2 */
__global__ void k_correlate(float* g, float* I_src, float* I_rec,
                             const float* P_prev, const float* P_curr,
                             const float* P_next, const float* lam, const float* v)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX_E || j >= NZ_E) return;
    size_t idx = IDX(i,j);
    float pc  = P_curr[idx];
    float lm  = lam[idx];
    float ptt = (P_next[idx] + P_prev[idx] - 2.0f*pc) / (DT*DT);
    float vv = v[idx];
    float v3 = vv * vv * vv;
    g[idx] += (2.0f / v3) * ptt * lm;

#if PRECOND_ILLUM
    I_src[idx] += pc * pc;
    I_rec[idx] += lm * lm;
#else
    (void)I_src; (void)I_rec;
#endif
}

/* Hard mask: zero where M < 0.5 (only used to save the raw gradient) */
__global__ void k_mask_grad(float* g_tilde, const float* g, const float* M)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NXZ) return;
    g_tilde[t] = (M[t] < 0.5f) ? 0.0f : g[t];
}

/* ====================== Gradient preconditioning ====================== */

/* out = a * b */
__global__ void k_mul(float* out, const float* a, const float* b, size_t n)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    out[t] = a[t] * b[t];
}

/* Main preconditioner: zero where M <= 0, otherwise illumination normalization
 * and v^p weighting (the tapered mask is applied at the end) */
__global__ void k_precond_grad(float* g_out, const float* g,
                                const float* I_s, const float* I_r,
                                const float* v,   const float* M,
                                float inv_Is, float inv_Ir, float eps_I)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NXZ) return;
    if (M[t] <= 0.0f) { g_out[t] = 0.0f; return; }

    float val = g[t];

#if PRECOND_ILLUM
    float I = I_s[t] * inv_Is;              /* normalize to [0,1] */
  #if ILLUM_USE_REC
    I *= I_r[t] * inv_Ir;
  #endif
    val /= (I + eps_I);
#else
    (void)I_s; (void)I_r; (void)inv_Is; (void)inv_Ir; (void)eps_I;
#endif

#if PRECOND_VPOW
    val *= powf(v[t] / VMIN_CLAMP, VPOW_P);
#else
    (void)v;
#endif

    g_out[t] = val;
}

/* Gaussian smoothing along x; truncated kernel renormalized at grid edges */
__global__ void k_smooth_x(float* out, const float* in, int R, float inv2s2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX_E || j >= NZ_E) return;
    float s = 0.0f, wsum = 0.0f;
    for (int m = -R; m <= R; ++m) {
        int ii = i + m;
        if (ii < 0 || ii >= NX_E) continue;
        float w = expf(-(float)(m*m) * inv2s2);
        s    += w * in[IDX(ii,j)];
        wsum += w;
    }
    out[IDX(i,j)] = (wsum > 0.0f) ? (s / wsum) : 0.0f;
}

/* Gaussian smoothing along z */
__global__ void k_smooth_z(float* out, const float* in, int R, float inv2s2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX_E || j >= NZ_E) return;
    float s = 0.0f, wsum = 0.0f;
    for (int m = -R; m <= R; ++m) {
        int jj = j + m;
        if (jj < 0 || jj >= NZ_E) continue;
        float w = expf(-(float)(m*m) * inv2s2);
        s    += w * in[IDX(i,jj)];
        wsum += w;
    }
    out[IDX(i,j)] = (wsum > 0.0f) ? (s / wsum) : 0.0f;
}

/* Finish the normalized convolution: g = num / den * M */
__global__ void k_smooth_finish(float* g_out, const float* num,
                                 const float* den, const float* M)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NXZ) return;
    g_out[t] = (den[t] > 1e-8f) ? (num[t] / den[t]) * M[t] : 0.0f;
}

/* Indicator of the mask support (1 where M > 0); after smoothing it is the
 * denominator of the normalized convolution */
__global__ void k_indicator(float* out, const float* M)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NXZ) return;
    out[t] = (M[t] > 0.0f) ? 1.0f : 0.0f;
}

/* Apply the tapered mask directly when smoothing is disabled */
__global__ void k_apply_taper(float* g, const float* M)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NXZ) return;
    g[t] *= M[t];
}

/* Search direction d = -g~ + beta*d_prev */
__global__ void k_search_dir(float* d, const float* g_tilde, const float* d_prev,
                              float beta)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NXZ) return;
    d[t] = -g_tilde[t] + beta * d_prev[t];
}

/* Model update v += alpha*d, clamped to [VMIN_CLAMP, VMAX_CLAMP] */
__global__ void k_update_v(float* v, const float* d, float alpha)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NXZ) return;
    float vn = v[t] + alpha * d[t];
    if (vn < VMIN_CLAMP) vn = VMIN_CLAMP;
    if (vn > VMAX_CLAMP) vn = VMAX_CLAMP;
    v[t] = vn;
}

/* Perturbed model v_pert = v + eps*d */
__global__ void k_make_pert(float* v_pert, const float* v, const float* d, float eps)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NXZ) return;
    v_pert[t] = v[t] + eps * d[t];
}

/* Fill the padding by replicating edge values of the physical region */
__global__ void k_pad_edge(float* v)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NXZ) return;
    int i = (int)(t / NZ_E);
    int j = (int)(t % NZ_E);
    int ii = i, jj = j;

    if      (ii <  L_PAD)              ii = L_PAD;
    else if (ii >  L_PAD + NX_C - 1)   ii = L_PAD + NX_C - 1;
    if      (jj <  L_PAD)              jj = L_PAD;
    else if (jj >  L_PAD + NZ_C - 1)   jj = L_PAD + NZ_C - 1;

    if (ii != i || jj != j) v[t] = v[IDX(ii,jj)];
}

/* Reduction: result += sum(a*b) */
__global__ void k_dot(const float* a, const float* b, float* result, size_t n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (gid < n) ? a[gid] * b[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

/* Reduction: result += sum(a^2) */
__global__ void k_norm2(const float* a, float* result, size_t n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < n) ? a[gid] : 0.0f;
    sdata[tid] = v * v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

/* Atomic max for float */
__device__ static inline float atomicMaxFloat(float* addr, float val) {
    int* iaddr = (int*)addr;
    int old = *iaddr, assumed;
    do {
        assumed = old;
        float current = __int_as_float(assumed);
        float candidate = fmaxf(current, val);
        old = atomicCAS(iaddr, assumed, __float_as_int(candidate));
    } while (assumed != old);
    return __int_as_float(old);
}

/* Reduction: result = max(result, max|a|) */
__global__ void k_max_abs(const float* a, float* result, size_t n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (gid < n) ? fabsf(a[gid]) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }
    if (tid == 0) atomicMaxFloat(result, sdata[0]);
}

/* out = (a - b) * inv_eps */
__global__ void k_diff_scale(float* out, const float* a, const float* b,
                              float inv_eps, size_t n)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) out[t] = (a[t] - b[t]) * inv_eps;
}

/* ============================ Host helpers ============================ */

/* Ricker wavelet (peak frequency F0, delay T0) */
static void compute_wavelet(float* h_w) {
    for (int k = 0; k < NT; k++) {
        float t = k * DT - T0;
        float arg = (float)M_PI * F0 * t;
        h_w[k] = (1.0f - 2.0f*arg*arg) * expf(-arg*arg);
    }
}

/* Gradient mask: 1 below the water layer inside the physical region (optional
 * cosine tapers), 0 in the water layer and the padding */
static void compute_mask(float* h_M) {
    for (int i = 0; i < NX_E; i++) {
        for (int j = 0; j < NZ_E; j++) {
            int in_x  = (i >= L_PAD) && (i < L_PAD + NX_C);
            int in_z  = (j >= L_PAD) && (j < L_PAD + NZ_C);
            float w = 0.0f;
            if (in_x && in_z) {
                w = 1.0f;
                /* water layer (physical j < NW) set to 0 */
                int dz = j - (L_PAD + NW);      /* grid index below the sea floor */
                if (dz < 0) w = 0.0f;
#if MASK_TAPER_Z > 0
                /* vertical cosine taper below the sea floor */
                else if (dz < MASK_TAPER_Z)
                    w *= 0.5f * (1.0f - cosf((float)M_PI * dz / MASK_TAPER_Z));
#endif
#if MASK_TAPER_X > 0
                /* lateral cosine taper at both ends of the physical region */
                if (w > 0.0f) {
                    int dl = i - L_PAD;
                    int dr = (L_PAD + NX_C - 1) - i;
                    int dm = (dl < dr) ? dl : dr;
                    if (dm < MASK_TAPER_X)
                        w *= 0.5f * (1.0f - cosf((float)M_PI * (dm + 1)
                                                 / (MASK_TAPER_X + 1)));
                }
#endif
            }
            h_M[IDX(i,j)] = w;
        }
    }
}

/* Blending weight alfa: 1 in the physical region and the outermost N_FD layers;
 * cos(0.5*pi*flag/B_CAL) in the flag-th layer outside the physical region
 * (flag = 1..B_CAL); corners take the larger flag of the two directions. */
static void compute_alfa(float* h_a) {
    for (size_t t = 0; t < NXZ; t++) h_a[t] = 1.0f;

    for (int i = N_FD; i < NX_E - N_FD; i++) {
        for (int j = N_FD; j < NZ_E - N_FD; j++) {
            for (int flag = 1; flag <= B_CAL; flag++) {
                float w = cosf(0.5f * (float)M_PI * (float)flag / (float)B_CAL);

                if (i == L_PAD - flag &&
                    j >= L_PAD - flag && j <= NZ_C + L_PAD - 1 + flag)
                    h_a[IDX(i,j)] = w;

                if (i == NX_C + L_PAD - 1 + flag &&
                    j >= L_PAD - flag && j <= NZ_C + L_PAD - 1 + flag)
                    h_a[IDX(i,j)] = w;

                if (j == L_PAD - flag &&
                    i > L_PAD - flag && i < NX_C + L_PAD - 1 + flag)
                    h_a[IDX(i,j)] = w;

                if (j == NZ_C + L_PAD - 1 + flag &&
                    i > L_PAD - flag && i < NX_C + L_PAD - 1 + flag)
                    h_a[IDX(i,j)] = w;
            }
        }
    }
}

/* Read the physical model (NX_C x NZ_C) into the extended grid; padding filled by edge replication */
static void read_model_padded(const char* path, float* h_v) {
    size_t n_core = (size_t)NX_C * NZ_C;
    float* buf = (float*)malloc(n_core * sizeof(float));
    if (!buf) { fprintf(stderr, "malloc failed in read_model_padded\n"); exit(1); }
    read_bin(path, buf, n_core);

    for (size_t t = 0; t < NXZ; t++) h_v[t] = 0.0f;
    for (int i = 0; i < NX_C; i++)
        for (int j = 0; j < NZ_C; j++)
            h_v[IDX(i + L_PAD, j + L_PAD)] = buf[(size_t)i*NZ_C + j];
    free(buf);

    /* fill padding by edge replication */
    for (int i = 0; i < NX_E; i++) {
        for (int j = 0; j < NZ_E; j++) {
            int ii = i, jj = j;
            if      (ii <  L_PAD)            ii = L_PAD;
            else if (ii >  L_PAD + NX_C - 1) ii = L_PAD + NX_C - 1;
            if      (jj <  L_PAD)            jj = L_PAD;
            else if (jj >  L_PAD + NZ_C - 1) jj = L_PAD + NZ_C - 1;
            if (ii != i || jj != j) h_v[IDX(i,j)] = h_v[IDX(ii,jj)];
        }
    }
}

/* Acquisition: source i = I_ACQ0 + round(s*DSX/DX), receiver i = I_REC0 + round(r*DRX/DX),
 * identical for all shots; receivers outside the physical region get index -1. */
static void compute_geometry(int* h_idx_srcs, int* h_idx_recs_all) {
    for (int s = 0; s < N_SHOTS; s++) {
        int i_src = I_ACQ0 + (int)roundf((float)s * DSX / DX);
        h_idx_srcs[s] = (int)IDX(i_src, J_SRC);
        for (int r = 0; r < N_RECS; r++) {
            int i_rec = I_REC0 + (int)roundf((float)r * DRX / DX);
            int valid = (i_rec >= L_PAD) && (i_rec < L_PAD + NX_C);
            h_idx_recs_all[s*N_RECS + r] = valid ? (int)IDX(i_rec, J_REC) : -1;
        }
    }
}

/* ============================ Trace integration (host) ============================
 * Weighted sliding-window integration y(t) = sum_{m=-Tw}^{Tw} h(m)*x(t-m), with zero
 * padding at both ends, repeated n_iter times. h is a Kaiser window
 * h(m) = I0(beta*sqrt(1-(m/Tw)^2)) / I0(beta); beta = 0 gives a rectangular window.
 * MATLAB equivalent:
 *   h = kaiser(2*Tw+1, beta).';      % h = h / sum(h) when TI_WIN_NORM = 1
 *   for k = 1:n_iter, y = conv(y, h, 'same'); end
 */

/* Zeroth-order modified Bessel function of the first kind I0(x), series expansion */
static double ti_bessel_i0(double x)
{
    double half = 0.5 * x;
    double term = 1.0, sum = 1.0;
    for (int k = 1; k < 64; k++) {
        double f = half / (double)k;
        term *= f * f;
        sum  += term;
        if (term < 1e-17 * sum) break;
    }
    return sum;
}

/* Kaiser window of length 2*Tw+1 (rectangular when beta <= 0), normalized to
 * sum = 1 when normalize != 0. The caller frees the returned array. */
static float* ti_make_window(int Tw, double beta, int normalize)
{
    int L = 2 * Tw + 1;
    float* h = (float*)malloc((size_t)L * sizeof(float));
    if (!h) { fprintf(stderr, "ti_make_window: malloc failed\n"); exit(1); }

    double denom = (beta > 0.0) ? ti_bessel_i0(beta) : 1.0;
    double sum   = 0.0;
    for (int m = -Tw; m <= Tw; m++) {
        double w;
        if (beta > 0.0 && Tw > 0) {
            double r = (double)m / (double)Tw;
            double a = 1.0 - r * r;
            if (a < 0.0) a = 0.0;
            w = ti_bessel_i0(beta * sqrt(a)) / denom;
        } else {
            w = 1.0;                       /* rectangular window */
        }
        h[m + Tw] = (float)w;
        sum += w;
    }
    if (normalize && sum > 0.0) {
        for (int i = 0; i < L; i++) h[i] = (float)((double)h[i] / sum);
    }
    return h;
}

/* n-pass weighted sliding-window integration of one trace (in place); h has length 2*Tw+1, centered at h[Tw] */
static void integrate_trace(float* trace, float* tmp, const float* h,
                             int N, int Tw, int n_iter)
{
    for (int iter = 0; iter < n_iter; iter++) {
        /* left end: window truncated */
        int t_lo = (Tw < N) ? Tw : N;
        for (int t = 0; t < t_lo; t++) {
            int hi = t + Tw;  if (hi > N - 1) hi = N - 1;
            float acc = 0.0f;
            for (int k = 0; k <= hi; k++) acc += h[k - t + Tw] * trace[k];
            tmp[t] = acc;
        }
        /* interior: full window inside the trace */
        int t_hi = N - Tw;                 /* interior range [Tw, N-Tw-1] */
        for (int t = t_lo; t < t_hi; t++) {
            const float* xp = trace + t - Tw;
            float acc = 0.0f;
            for (int m = 0; m <= 2 * Tw; m++) acc += h[m] * xp[m];
            tmp[t] = acc;
        }
        /* right end: window truncated */
        int t_start = (t_hi > t_lo) ? t_hi : t_lo;
        for (int t = t_start; t < N; t++) {
            int lo = t - Tw;  if (lo < 0) lo = 0;
            float acc = 0.0f;
            for (int k = lo; k < N; k++) acc += h[k - t + Tw] * trace[k];
            tmp[t] = acc;
        }
        memcpy(trace, tmp, N * sizeof(float));
    }
}

/* n-pass trace integration of the wavelet (in place) */
static void integrate_wavelet_n(float* wavelet, int nt, int Tw, int n_iter) {
    if (n_iter <= 0 || Tw <= 0) return;
    float* tmp = (float*)malloc(nt * sizeof(float));
    float* h   = ti_make_window(Tw, TI_KAISER_BETA, TI_WIN_NORM);
    integrate_trace(wavelet, tmp, h, nt, Tw, n_iter);
    free(tmp); free(h);
}

/* n-pass trace integration of all observed traces (in place); data stored as
 * [shot][time][receiver], receiver index fastest */
static void integrate_obs_n(float* h_obs, int nshot, int nrec, int nt,
                             int Tw, int n_iter)
{
    if (n_iter <= 0 || Tw <= 0) return;

    float* trace = (float*)malloc(nt * sizeof(float));
    float* tmp   = (float*)malloc(nt * sizeof(float));
    float* h     = ti_make_window(Tw, TI_KAISER_BETA, TI_WIN_NORM);

    for (int s = 0; s < nshot; s++) {
        for (int r = 0; r < nrec; r++) {
            float* base = h_obs + (size_t)s * nt * nrec + r;

            for (int k = 0; k < nt; k++) trace[k] = base[k * nrec];

            integrate_trace(trace, tmp, h, nt, Tw, n_iter);

            for (int k = 0; k < nt; k++) base[k * nrec] = trace[k];
        }
        if ((s+1) % 10 == 0 || s == nshot-1) {
            printf("\r  integrating obs traces: %3d/%3d shots", s+1, nshot);
            fflush(stdout);
        }
    }
    printf("\n");

    free(trace); free(tmp); free(h);
}

/* =========================== Global buffers =========================== */
static float *d_v_curr, *d_v_obs_model, *d_v_pert;
static float *d_g, *d_g_tilde, *d_g_tilde_prev;
static float *d_d, *d_d_prev;
static float *d_M;
static float *d_Msm;                        /* smooth(1[M>0]), normalized-convolution denominator */
static float *d_I_src, *d_I_rec;            /* source / receiver illumination */
static float *d_sm1, *d_sm2;                /* scratch buffers for smoothing */
static float *d_alfa;                       /* Higdon/FD blending weights */
/* Absorbing-band buffers: u1 = FD solution, u2 = Higdon solution */
static float *d_u1_l,  *d_u1_r,  *d_u1_u,  *d_u1_d;
static float *d_u1_lu, *d_u1_ru, *d_u1_ld, *d_u1_rd;
static float *d_u2_l,  *d_u2_r,  *d_u2_u,  *d_u2_d;
static float *d_u2_lu, *d_u2_ru, *d_u2_ld, *d_u2_rd;

#define ABC_U1_ARGS d_u1_l,d_u1_r,d_u1_u,d_u1_d,d_u1_lu,d_u1_ru,d_u1_ld,d_u1_rd
#define ABC_U2_ARGS d_u2_l,d_u2_r,d_u2_u,d_u2_d,d_u2_lu,d_u2_ru,d_u2_ld,d_u2_rd
static float *d_wavelet;
static float *d_P0, *d_P1, *d_P2;
static float *d_L0, *d_L1, *d_L2;
static float *d_wavefield;                  /* forward wavefield, NT x NXZ */
static float *d_obs_all, *d_syn_all, *d_syn_pert_all;
static int   *d_idx_recs;                   /* receiver indices of the current shot */
static int   *h_idx_recs_all;
static int   *h_idx_srcs;
static float *d_red;                        /* reduction result */

/* Forward modeling of one shot: records receiver data if d_syn_shot != NULL,
 * stores the full wavefield if save_wavefield = 1 */
static void forward_one_shot(float* d_v, int s,
                              float* d_syn_shot /* NT*N_RECS or NULL */,
                              int save_wavefield /* 0/1 */)
{
    dim3 block2(32, 8);
    dim3 grid2((NZ_E + 31)/32, (NX_E + 7)/8);
    int nblk1 = (int)((NXZ + 255) / 256);
    int nblkR = (N_RECS + 255) / 256;

    /* receiver indices of this shot */
    CK(cudaMemcpy(d_idx_recs, h_idx_recs_all + s*N_RECS,
                   N_RECS*sizeof(int), cudaMemcpyHostToDevice));

    int idx_src = h_idx_srcs[s];

    CK(cudaMemset(d_P0, 0, NXZ*sizeof(float)));
    CK(cudaMemset(d_P1, 0, NXZ*sizeof(float)));
    CK(cudaMemset(d_P2, 0, NXZ*sizeof(float)));

    for (int k = 0; k < NT; k++) {
        /* 1) interior FD */
        k_forward<<<grid2, block2>>>(d_P2, d_P1, d_P0, d_v, ABC_U1_ARGS);
        /* 2) Higdon solution in the absorbing band */
        k_higdon <<<grid2, block2>>>(d_P1, d_P0, d_v, ABC_U2_ARGS);
        /* 3) blend in the absorbing band */
        k_hybrid <<<grid2, block2>>>(d_P2, d_alfa, ABC_U1_ARGS, ABC_U2_ARGS);
        /* 4) add source */
        k_add_source<<<1,1>>>(d_P2, d_v, d_wavelet, (size_t)idx_src, k);

        if (d_syn_shot != NULL) {
            k_record<<<nblkR, 256>>>(d_P2, d_syn_shot + (size_t)k*N_RECS, d_idx_recs);
        }
        if (save_wavefield) {
            k_save_slice<<<nblk1, 256>>>(d_wavefield, d_P2, k);
        }

        /* rotate: P0 <- P1, P1 <- P2 */
        float* tmp = d_P0; d_P0 = d_P1; d_P1 = d_P2; d_P2 = tmp;
    }
}

/* Adjoint of one shot: back-propagate residuals and cross-correlate with the
 * forward wavefield to accumulate the gradient (and illumination) */
static void adjoint_one_shot(float* d_v, int s, const float* d_residual_shot)
{
    dim3 block2(32, 8);
    dim3 grid2((NZ_E + 31)/32, (NX_E + 7)/8);
    int nblkR = (N_RECS + 255) / 256;

    /* receiver indices of this shot */
    CK(cudaMemcpy(d_idx_recs, h_idx_recs_all + s*N_RECS,
                   N_RECS*sizeof(int), cudaMemcpyHostToDevice));

    CK(cudaMemset(d_L0, 0, NXZ*sizeof(float)));
    CK(cudaMemset(d_L1, 0, NXZ*sizeof(float)));
    CK(cudaMemset(d_L2, 0, NXZ*sizeof(float)));

    for (int k = NT - 1; k >= 0; k--) {
        /* adjoint propagation (same operator as forward) */
        k_forward<<<grid2, block2>>>(d_L2, d_L1, d_L0, d_v, ABC_U1_ARGS);
        k_higdon <<<grid2, block2>>>(d_L1, d_L0, d_v, ABC_U2_ARGS);
        k_hybrid <<<grid2, block2>>>(d_L2, d_alfa, ABC_U1_ARGS, ABC_U2_ARGS);

        /* inject residual of time step k */
        k_inject_residual<<<nblkR, 256>>>(d_L2, d_v,
            d_residual_shot + (size_t)k*N_RECS, d_idx_recs);

        /* Stored forward slice k corresponds to u^{k+1}, so slices k-2, k-1, k give
         * the second time derivative aligned with residual time k, which is then
         * cross-correlated with the adjoint wavefield */
        if (k >= 2) {
            float* P_prev = d_wavefield + (size_t)(k-2)*NXZ;
            float* P_curr = d_wavefield + (size_t)(k-1)*NXZ;
            float* P_next = d_wavefield + (size_t) k    *NXZ;
            k_correlate<<<grid2, block2>>>(d_g, d_I_src, d_I_rec,
                                            P_prev, P_curr, P_next, d_L2, d_v);
        }

        /* rotate */
        float* tmp = d_L0; d_L0 = d_L1; d_L1 = d_L2; d_L2 = tmp;
    }
}

/* Host wrappers for reductions */
static float reduce_dot(const float* d_a, const float* d_b, size_t n) {
    CK(cudaMemset(d_red, 0, sizeof(float)));
    int nblk = (int)((n + 255) / 256);
    k_dot<<<nblk, 256>>>(d_a, d_b, d_red, n);
    float h_r;
    CK(cudaMemcpy(&h_r, d_red, sizeof(float), cudaMemcpyDeviceToHost));
    return h_r;
}
static float reduce_norm2(const float* d_a, size_t n) {
    CK(cudaMemset(d_red, 0, sizeof(float)));
    int nblk = (int)((n + 255) / 256);
    k_norm2<<<nblk, 256>>>(d_a, d_red, n);
    float h_r;
    CK(cudaMemcpy(&h_r, d_red, sizeof(float), cudaMemcpyDeviceToHost));
    return h_r;
}
static float reduce_max(const float* d_a, size_t n) {
    float zero = 0.0f;
    CK(cudaMemcpy(d_red, &zero, sizeof(float), cudaMemcpyHostToDevice));
    int nblk = (int)((n + 255) / 256);
    k_max_abs<<<nblk, 256>>>(d_a, d_red, n);
    float h_r;
    CK(cudaMemcpy(&h_r, d_red, sizeof(float), cudaMemcpyDeviceToHost));
    return h_r;
}

/* Masked maximum max|field*M|, used for illumination normalization */
static float masked_max(const float* d_field) {
    int nblk = (int)((NXZ + 255) / 256);
    k_mul<<<nblk, 256>>>(d_sm1, d_field, d_M, NXZ);
    return reduce_max(d_sm1, NXZ);
}

/* Separable Gaussian smoothing inside the mask (normalized convolution), in place */
static void smooth_masked(float* d_field) {
    dim3 block2(32, 8);
    dim3 grid2((NZ_E + 31)/32, (NX_E + 7)/8);
    int   nblk   = (int)((NXZ + 255) / 256);
    float inv2s2 = 1.0f / (2.0f * SMOOTH_SIGMA * SMOOTH_SIGMA);
    k_smooth_x<<<grid2, block2>>>(d_sm1, d_field, SMOOTH_RADIUS, inv2s2);
    k_smooth_z<<<grid2, block2>>>(d_sm2, d_sm1,   SMOOTH_RADIUS, inv2s2);
    k_smooth_finish<<<nblk, 256>>>(d_field, d_sm2, d_Msm, d_M);
}

/* ================================ Main ================================ */
int main(int argc, char** argv) {
    printf("\n========================================================\n");
    printf("  Trace-Integration FWI (CUDA)\n");
    printf("  Grid    : %d (z) x %d (x) extended, %d (z) x %d (x) physical\n",
           NZ_E, NX_E, NZ_C, NX_C);
    printf("  Acq     : %d shots, %d receivers (fixed spread)\n", N_SHOTS, N_RECS);
    printf("  Time    : NT=%d, dt=%.4f s\n", NT, DT);
    printf("  Wavelet : Ricker f0=%.1f Hz\n", F0);
    printf("  K_iter  : %d\n", K_ITER);
    printf("  TI      : n=%d, window=%d samples, Kaiser beta=%.1f\n",
           N_TI_ITER, 2*TI_TW_HALF+1, (double)TI_KAISER_BETA);
    printf("========================================================\n\n");

    make_dir("./output");

    /* GPU info */
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("GPU      : %s (%.1f GB)\n", prop.name,
           prop.totalGlobalMem / 1e9);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Free mem : %.2f / %.2f GB\n\n",
           free_mem/1e9, total_mem/1e9);

    /* ============================================================
       STEP 1: allocate GPU memory
       ============================================================ */
    printf(">> Allocating GPU memory ...\n");
    size_t bytes_NXZ      = NXZ * sizeof(float);
    size_t bytes_wavefield = (size_t)NT * NXZ * sizeof(float);
    size_t bytes_data     = (size_t)N_SHOTS * N_RECS * NT * sizeof(float);

    printf("   wavefield buffer : %.2f GB\n", bytes_wavefield/1e9);
    printf("   obs/syn arrays   : 3 x %.2f GB\n", bytes_data/1e9);

    CK(cudaMalloc(&d_v_curr,       bytes_NXZ));
    CK(cudaMalloc(&d_v_obs_model,  bytes_NXZ));
    CK(cudaMalloc(&d_v_pert,       bytes_NXZ));
    CK(cudaMalloc(&d_g,            bytes_NXZ));
    CK(cudaMalloc(&d_g_tilde,      bytes_NXZ));
    CK(cudaMalloc(&d_g_tilde_prev, bytes_NXZ));
    CK(cudaMalloc(&d_d,            bytes_NXZ));
    CK(cudaMalloc(&d_d_prev,       bytes_NXZ));
    CK(cudaMalloc(&d_M,            bytes_NXZ));
    CK(cudaMalloc(&d_Msm,          bytes_NXZ));
    CK(cudaMalloc(&d_I_src,        bytes_NXZ));
    CK(cudaMalloc(&d_I_rec,        bytes_NXZ));
    CK(cudaMalloc(&d_sm1,          bytes_NXZ));
    CK(cudaMalloc(&d_sm2,          bytes_NXZ));
    CK(cudaMalloc(&d_alfa,         bytes_NXZ));
    {   /* absorbing-band buffers */
        size_t b_lr = (size_t)B_CAL * NZ_C * sizeof(float);
        size_t b_ud = (size_t)NX_C  * B_CAL * sizeof(float);
        size_t b_cn = (size_t)B_CAL * B_CAL * sizeof(float);
        CK(cudaMalloc(&d_u1_l,  b_lr)); CK(cudaMalloc(&d_u2_l,  b_lr));
        CK(cudaMalloc(&d_u1_r,  b_lr)); CK(cudaMalloc(&d_u2_r,  b_lr));
        CK(cudaMalloc(&d_u1_u,  b_ud)); CK(cudaMalloc(&d_u2_u,  b_ud));
        CK(cudaMalloc(&d_u1_d,  b_ud)); CK(cudaMalloc(&d_u2_d,  b_ud));
        CK(cudaMalloc(&d_u1_lu, b_cn)); CK(cudaMalloc(&d_u2_lu, b_cn));
        CK(cudaMalloc(&d_u1_ru, b_cn)); CK(cudaMalloc(&d_u2_ru, b_cn));
        CK(cudaMalloc(&d_u1_ld, b_cn)); CK(cudaMalloc(&d_u2_ld, b_cn));
        CK(cudaMalloc(&d_u1_rd, b_cn)); CK(cudaMalloc(&d_u2_rd, b_cn));
        CK(cudaMemset(d_u1_l, 0, b_lr)); CK(cudaMemset(d_u2_l, 0, b_lr));
        CK(cudaMemset(d_u1_r, 0, b_lr)); CK(cudaMemset(d_u2_r, 0, b_lr));
        CK(cudaMemset(d_u1_u, 0, b_ud)); CK(cudaMemset(d_u2_u, 0, b_ud));
        CK(cudaMemset(d_u1_d, 0, b_ud)); CK(cudaMemset(d_u2_d, 0, b_ud));
        CK(cudaMemset(d_u1_lu,0, b_cn)); CK(cudaMemset(d_u2_lu,0, b_cn));
        CK(cudaMemset(d_u1_ru,0, b_cn)); CK(cudaMemset(d_u2_ru,0, b_cn));
        CK(cudaMemset(d_u1_ld,0, b_cn)); CK(cudaMemset(d_u2_ld,0, b_cn));
        CK(cudaMemset(d_u1_rd,0, b_cn)); CK(cudaMemset(d_u2_rd,0, b_cn));
    }
    CK(cudaMalloc(&d_wavelet,      NT*sizeof(float)));
    CK(cudaMalloc(&d_P0,           bytes_NXZ));
    CK(cudaMalloc(&d_P1,           bytes_NXZ));
    CK(cudaMalloc(&d_P2,           bytes_NXZ));
    CK(cudaMalloc(&d_L0,           bytes_NXZ));
    CK(cudaMalloc(&d_L1,           bytes_NXZ));
    CK(cudaMalloc(&d_L2,           bytes_NXZ));
    CK(cudaMalloc(&d_wavefield,    bytes_wavefield));
    CK(cudaMalloc(&d_obs_all,      bytes_data));
    CK(cudaMalloc(&d_syn_all,      bytes_data));
    CK(cudaMalloc(&d_syn_pert_all, bytes_data));
    CK(cudaMalloc(&d_idx_recs,     N_RECS*sizeof(int)));
    CK(cudaMalloc(&d_red,          sizeof(float)));

    /* ============================================================
       STEP 2: wavelet, gradient mask, blending weights, acquisition geometry
       ============================================================ */
    printf(">> Building wavelet, mask, ABC weights, geometry ...\n");

    float* h_buf = (float*)malloc(bytes_NXZ);

    float h_wavelet[NT];
    compute_wavelet(h_wavelet);
    CK(cudaMemcpy(d_wavelet, h_wavelet, NT*sizeof(float), cudaMemcpyHostToDevice));

    compute_mask(h_buf);
    CK(cudaMemcpy(d_M, h_buf, bytes_NXZ, cudaMemcpyHostToDevice));

    /* Normalized-convolution denominator d_Msm = smooth(1[M>0]); the mask is fixed, computed once */
    {
        dim3 block2(32, 8);
        dim3 grid2((NZ_E + 31)/32, (NX_E + 7)/8);
        int   nblk   = (int)((NXZ + 255) / 256);
        float inv2s2 = 1.0f / (2.0f * SMOOTH_SIGMA * SMOOTH_SIGMA);
        k_indicator<<<nblk, 256>>>(d_sm2, d_M);
        k_smooth_x<<<grid2, block2>>>(d_sm1, d_sm2, SMOOTH_RADIUS, inv2s2);
        k_smooth_z<<<grid2, block2>>>(d_Msm, d_sm1, SMOOTH_RADIUS, inv2s2);
        CK(cudaDeviceSynchronize());
    }

    compute_alfa(h_buf);
    CK(cudaMemcpy(d_alfa, h_buf, bytes_NXZ, cudaMemcpyHostToDevice));

    h_idx_srcs = (int*)malloc(N_SHOTS*sizeof(int));
    h_idx_recs_all = (int*)malloc(N_SHOTS*N_RECS*sizeof(int));
    compute_geometry(h_idx_srcs, h_idx_recs_all);

    /* ============================================================
       STEP 3: read true and initial models
       ============================================================ */
    printf(">> Reading model_bin/mar2_ext_501_141.bin (true model) ...\n");
    read_model_padded("./model_bin/mar2_ext_501_141.bin", h_buf);
    CK(cudaMemcpy(d_v_obs_model, h_buf, bytes_NXZ, cudaMemcpyHostToDevice));

    printf(">> Reading model_bin/mar2_lin_501_141_4400.bin (initial model) ...\n");
    read_model_padded("./model_bin/mar2_lin_501_141_4400.bin", h_buf);
    CK(cudaMemcpy(d_v_curr, h_buf, bytes_NXZ, cudaMemcpyHostToDevice));

    free(h_buf);

    /* ============================================================
       STEP 4: synthesize observed data from the true model
       ============================================================ */
    printf("\n>> Generating observed data (true model) ...\n");
    double t_start = now_sec();
    for (int s = 0; s < N_SHOTS; s++) {
        forward_one_shot(d_v_obs_model, s,
                          d_obs_all + (size_t)s * N_RECS * NT,
                          /*save_wavefield=*/0);
        show_progress("  obs gen", s+1, N_SHOTS, now_sec() - t_start);
    }
    printf("\n");

    /* save observed data */
    {
        float* h_obs = (float*)malloc(bytes_data);
        CK(cudaMemcpy(h_obs, d_obs_all, bytes_data, cudaMemcpyDeviceToHost));
        write_bin("./output/obs_data.bin", h_obs,
                   (size_t)N_SHOTS * N_RECS * NT);
        free(h_obs);
        printf("  saved: ./output/obs_data.bin (%.2f MB, raw)\n", bytes_data/1e6);
    }

    /* release the true model */
    CK(cudaFree(d_v_obs_model));

    /* ============================================================
       STEP 4.5: trace-integration preprocessing
       Trace integration is a linear time-invariant filter that commutes with
       forward modeling, so the wavelet and the observed data only need the
       same number of integrations. Both are then scaled by the peak of the
       integrated wavelet (the update alpha*d is invariant to this common scaling).
       ============================================================ */
    {
        /* save raw wavelet */
        write_bin("./output/wavelet_orig.bin", h_wavelet, NT);

        /* integrate wavelet */
        printf("\n>> Applying %d-times trace integration to source wavelet ...\n",
               N_TI_ITER);
        integrate_wavelet_n(h_wavelet, NT, TI_TW_HALF, N_TI_ITER);

        /* integrate observed data */
        float* h_obs = (float*)malloc(bytes_data);
        CK(cudaMemcpy(h_obs, d_obs_all, bytes_data, cudaMemcpyDeviceToHost));

        printf(">> Applying %d-times trace integration to obs data ...\n",
               N_TI_ITER);
        double t_int = now_sec();
        integrate_obs_n(h_obs, N_SHOTS, N_RECS, NT, TI_TW_HALF, N_TI_ITER);
        printf("  done in %.1f s\n", now_sec() - t_int);

        /* amplitude normalization: wavelet peak scaled to 1, observed data scaled by the same factor */
        float peak_w = 0.0f;
        for (int t = 0; t < NT; t++) {
            float a = fabsf(h_wavelet[t]);
            if (a > peak_w) peak_w = a;
        }
        const float amp_scale = (peak_w > 0.0f) ? (1.0f / peak_w) : 1.0f;
        printf(">> Amplitude normalization:\n");
        printf("   peak|int_wavelet| = %.3e   ->   scale = %.3e\n",
               peak_w, amp_scale);

        for (int t = 0; t < NT; t++) h_wavelet[t] *= amp_scale;
        const size_t n_obs_total = (size_t)N_SHOTS * N_RECS * NT;
        for (size_t i = 0; i < n_obs_total; i++) h_obs[i] *= amp_scale;

        /* save integrated wavelet and observed data */
        write_bin("./output/wavelet_int.bin",  h_wavelet, NT);
        write_bin("./output/obs_data_int.bin", h_obs, n_obs_total);
        printf("   saved: ./output/wavelet_int.bin, ./output/obs_data_int.bin (peak~1)\n");

        /* upload to GPU for the inversion */
        CK(cudaMemcpy(d_wavelet, h_wavelet, NT*sizeof(float),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_obs_all, h_obs, bytes_data,
                      cudaMemcpyHostToDevice));

        free(h_obs);
    }

    /* ============================================================
       STEP 5: FWI main loop
       ============================================================ */
    printf("\n>> Starting FWI iterations ...\n\n");

    float h_obj_hist[K_ITER]      = {0};
    float h_alpha_hist[K_ITER]    = {0};
    float h_beta_hist[K_ITER]     = {0};
    float h_gradnorm_hist[K_ITER] = {0};

    int nblk1 = (int)((NXZ + 255) / 256);
    size_t n_data_total = (size_t)N_SHOTS * N_RECS * NT;

    /* at k = 0, beta = 0 and d = -g~ */
    CK(cudaMemset(d_d_prev,       0, bytes_NXZ));
    CK(cudaMemset(d_g_tilde_prev, 0, bytes_NXZ));

    double t_iter_start = now_sec();

    for (int k = 0; k < K_ITER; k++) {
        double t_k = now_sec();
        char prefix[64];

        /* ---- 1. forward + adjoint, accumulate gradient ---- */
        CK(cudaMemset(d_g, 0, bytes_NXZ));
#if PRECOND_ILLUM
        /* re-accumulate illumination every iteration */
        CK(cudaMemset(d_I_src, 0, bytes_NXZ));
        CK(cudaMemset(d_I_rec, 0, bytes_NXZ));
#endif

        double obj = 0.0;

        snprintf(prefix, 64, "  it %2d/%d fwd+adj", k+1, K_ITER);
        for (int s = 0; s < N_SHOTS; s++) {
            /* forward modeling, store wavefield and synthetic data */
            forward_one_shot(d_v_curr, s,
                              d_syn_all + (size_t)s*N_RECS*NT,
                              /*save_wavefield=*/1);

            /* residual syn - obs, temporarily stored in d_syn_pert_all */
            size_t off = (size_t)s * N_RECS * NT;
            int nblkRes = (int)((N_RECS*NT + 255)/256);
            k_subtract<<<nblkRes, 256>>>(d_syn_pert_all + off,
                                          d_syn_all + off,
                                          d_obs_all + off,
                                          (size_t)N_RECS*NT);

            /* objective 0.5*||syn - obs||^2 */
            obj += 0.5 * reduce_norm2(d_syn_pert_all + off, (size_t)N_RECS*NT);

            /* adjoint propagation, accumulate gradient */
            adjoint_one_shot(d_v_curr, s, d_syn_pert_all + off);

            show_progress(prefix, s+1, N_SHOTS, now_sec() - t_k);
        }
        printf("\n");
        h_obj_hist[k] = (float)obj;

        /* ---- 2. preconditioning: illumination -> v^p weighting -> masked smoothing -> tapered mask ---- */
        {
            float inv_Is = 1.0f, inv_Ir = 1.0f;
#if PRECOND_ILLUM
            float Is_max = masked_max(d_I_src);
            float Ir_max = masked_max(d_I_rec);
            inv_Is = (Is_max > 1e-30f) ? 1.0f / Is_max : 0.0f;
            inv_Ir = (Ir_max > 1e-30f) ? 1.0f / Ir_max : 0.0f;
#endif
            k_precond_grad<<<nblk1, 256>>>(d_g_tilde, d_g, d_I_src, d_I_rec,
                                            d_v_curr, d_M,
                                            inv_Is, inv_Ir, ILLUM_EPS);
#if PRECOND_SMOOTH
            smooth_masked(d_g_tilde);
#else
            k_apply_taper<<<nblk1, 256>>>(d_g_tilde, d_M);
#endif
        }

        /* PRP: beta = g~ . (g~ - g~_prev) / ||g~_prev||^2 */
        float beta = 0.0f;
        if (k > 0) {
            float* d_diff = d_d;       /* d_d reused as scratch */
            int nblk_diff = (int)((NXZ+255)/256);
            k_diff_scale<<<nblk_diff, 256>>>(d_diff, d_g_tilde, d_g_tilde_prev,
                                              1.0f, NXZ);
            float num   = reduce_dot(d_g_tilde, d_diff, NXZ);
            float denom = reduce_norm2(d_g_tilde_prev, NXZ);
            if (denom > 1e-30f) beta = num / denom;
        }
        h_beta_hist[k] = beta;

        /* d = -g~ + beta*d_prev */
        k_search_dir<<<nblk1, 256>>>(d_d, d_g_tilde, d_d_prev, beta);

        /* L2 norm of the raw gradient d_g (before masking and preconditioning) */
        h_gradnorm_hist[k] = sqrtf(reduce_norm2(d_g, NXZ));

        /* ---- 3. step length by Born finite difference ---- */
        snprintf(prefix, 64, "  it %2d/%d step  ", k+1, K_ITER);

        float max_v = reduce_max(d_v_curr, NXZ);
        float max_d = reduce_max(d_d,      NXZ);
        /* perturbation eps such that max|eps*d| = max|v| / 100 */
        float eps_pert = (max_d > 1e-30f) ? max_v / (100.0f * max_d) : 0.0f;

        /* perturbed model */
        k_make_pert<<<nblk1, 256>>>(d_v_pert, d_v_curr, d_d, eps_pert);

        /* forward modeling with the perturbed model */
        double t_step = now_sec();
        for (int s = 0; s < N_SHOTS; s++) {
            forward_one_shot(d_v_pert, s,
                              d_syn_pert_all + (size_t)s*N_RECS*NT,
                              /*save_wavefield=*/0);
            show_progress(prefix, s+1, N_SHOTS, now_sec() - t_step);
        }
        printf("\n");

        /* delta = syn_pert - syn (not divided by eps) */
        int nblk_diff = (int)((n_data_total + 255) / 256);
        k_subtract<<<nblk_diff, 256>>>(d_syn_pert_all,
                                        d_syn_pert_all, d_syn_all,
                                        n_data_total);

        /* residual syn - obs */
        int nblkR = (int)((n_data_total + 255) / 256);
        k_subtract<<<nblkR, 256>>>(d_syn_all, d_syn_all, d_obs_all,
                                    n_data_total);

        /* alpha = -eps * <delta, syn - obs> / ||delta||^2 */
        float dot   = reduce_dot  (d_syn_pert_all, d_syn_all, n_data_total);
        float ndel2 = reduce_norm2(d_syn_pert_all, n_data_total);

        float alpha = (ndel2 > 1e-20f) ? (-eps_pert * dot / ndel2) : 0.0f;
        h_alpha_hist[k] = alpha;

        /* ---- 4. update model and replicate edges ---- */
        k_update_v<<<nblk1, 256>>>(d_v_curr, d_d, alpha);
        k_pad_edge<<<nblk1, 256>>>(d_v_curr);

        /* keep current g~ and d for the next iteration */
        CK(cudaMemcpy(d_g_tilde_prev, d_g_tilde, bytes_NXZ,
                       cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(d_d_prev,       d_d,       bytes_NXZ,
                       cudaMemcpyDeviceToDevice));

        /* ---- 5. output ---- */
        printf("  iter %2d/%d  obj=%.4e  alpha=%.4e  beta=%.4e  ||g||=%.4e  (%.1fs)\n",
               k+1, K_ITER, h_obj_hist[k], alpha, beta, h_gradnorm_hist[k],
               now_sec() - t_k);

        if (((k+1) % SAVE_EVERY) == 0 || k == K_ITER-1) {
            float* h_v = (float*)malloc(bytes_NXZ);
            CK(cudaMemcpy(h_v, d_v_curr, bytes_NXZ, cudaMemcpyDeviceToHost));
            char path[256];
            snprintf(path, 256, "./output/vp_inv_iter_%02d.bin", k+1);
            write_bin(path, h_v, NXZ);

#if SAVE_GRAD
            /* raw gradient (hard mask, using the idle buffer d_sm1) */
            k_mask_grad<<<nblk1, 256>>>(d_sm1, d_g, d_M);
            CK(cudaMemcpy(h_v, d_sm1, bytes_NXZ, cudaMemcpyDeviceToHost));
            snprintf(path, 256, "./output/grad_raw_iter_%02d.bin", k+1);
            write_bin(path, h_v, NXZ);

            /* preconditioned gradient */
            CK(cudaMemcpy(h_v, d_g_tilde, bytes_NXZ, cudaMemcpyDeviceToHost));
            snprintf(path, 256, "./output/grad_prec_iter_%02d.bin", k+1);
            write_bin(path, h_v, NXZ);

  #if PRECOND_ILLUM
            CK(cudaMemcpy(h_v, d_I_src, bytes_NXZ, cudaMemcpyDeviceToHost));
            snprintf(path, 256, "./output/illum_src_iter_%02d.bin", k+1);
            write_bin(path, h_v, NXZ);
  #endif
#endif
            free(h_v);
        }
    }

    /* ============================================================
       STEP 6: save final model and iteration histories
       ============================================================ */
    printf("\n>> Total FWI time: %.1f s = %.1f min\n",
           now_sec() - t_iter_start, (now_sec() - t_iter_start)/60.0);

    {
        float* h_v = (float*)malloc(bytes_NXZ);
        CK(cudaMemcpy(h_v, d_v_curr, bytes_NXZ, cudaMemcpyDeviceToHost));
        write_bin("./output/vp_inv_final.bin", h_v, NXZ);
        free(h_v);
    }
    write_bin("./output/obj_history.bin",      h_obj_hist,      K_ITER);
    write_bin("./output/alpha_history.bin",    h_alpha_hist,    K_ITER);
    write_bin("./output/beta_history.bin",     h_beta_hist,     K_ITER);
    write_bin("./output/gradnorm_history.bin", h_gradnorm_hist, K_ITER);

    /* ============================================================
       Release memory
       ============================================================ */
    CK(cudaFree(d_v_curr));
    CK(cudaFree(d_v_pert));
    CK(cudaFree(d_g));
    CK(cudaFree(d_g_tilde));
    CK(cudaFree(d_g_tilde_prev));
    CK(cudaFree(d_d));
    CK(cudaFree(d_d_prev));
    CK(cudaFree(d_M));
    CK(cudaFree(d_Msm));
    CK(cudaFree(d_I_src));
    CK(cudaFree(d_I_rec));
    CK(cudaFree(d_sm1));
    CK(cudaFree(d_sm2));
    CK(cudaFree(d_alfa));
    CK(cudaFree(d_u1_l));  CK(cudaFree(d_u2_l));
    CK(cudaFree(d_u1_r));  CK(cudaFree(d_u2_r));
    CK(cudaFree(d_u1_u));  CK(cudaFree(d_u2_u));
    CK(cudaFree(d_u1_d));  CK(cudaFree(d_u2_d));
    CK(cudaFree(d_u1_lu)); CK(cudaFree(d_u2_lu));
    CK(cudaFree(d_u1_ru)); CK(cudaFree(d_u2_ru));
    CK(cudaFree(d_u1_ld)); CK(cudaFree(d_u2_ld));
    CK(cudaFree(d_u1_rd)); CK(cudaFree(d_u2_rd));
    CK(cudaFree(d_wavelet));
    CK(cudaFree(d_P0));
    CK(cudaFree(d_P1));
    CK(cudaFree(d_P2));
    CK(cudaFree(d_L0));
    CK(cudaFree(d_L1));
    CK(cudaFree(d_L2));
    CK(cudaFree(d_wavefield));
    CK(cudaFree(d_obs_all));
    CK(cudaFree(d_syn_all));
    CK(cudaFree(d_syn_pert_all));
    CK(cudaFree(d_idx_recs));
    CK(cudaFree(d_red));
    free(h_idx_srcs);
    free(h_idx_recs_all);

    printf("\nDone.\n\n");
    return 0;
}