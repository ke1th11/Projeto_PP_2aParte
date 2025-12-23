
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

typedef struct { float x; int ix; float ux, uy, uz; } t_part;
typedef struct { float x, y, z; } t_fld;

static t_part *d_parts = NULL;
static t_fld  *d_E = NULL, *d_B = NULL, *d_J = NULL;
static int cuda_initialized = 0;

extern "C" void cuda_init(int np, int nx) {
    cudaMalloc(&d_parts, np * sizeof(t_part));
    cudaMalloc(&d_E, (nx + 1) * sizeof(t_fld));
    cudaMalloc(&d_B, (nx + 1) * sizeof(t_fld));
    cudaMalloc(&d_J, (nx + 1) * sizeof(t_fld)); // Alocar memória para a Corrente
}

extern "C" void cuda_final() {
    if (d_parts) cudaFree(d_parts);
    if (d_E)     cudaFree(d_E);
    if (d_B)     cudaFree(d_B);
    if (d_J)     cudaFree(d_J);
}

__global__ void spec_advance_kernel(t_part *parts, int np, const t_fld *E, const t_fld *B, t_fld *J, int nx, float tem, float dt_dx, float qnx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;

    // --- 1. BORIS PUSHER (O que já tinhas) ---
    float x = parts[i].x;
    int ix = parts[i].ix;
    float w1 = x; float w0 = 1.0f - w1;

    t_fld Ep = {w0*E[ix].x + w1*E[ix+1].x, w0*E[ix].y + w1*E[ix+1].y, w0*E[ix].z + w1*E[ix+1].z};
    t_fld Bp = {w0*B[ix].x + w1*B[ix+1].x, w0*B[ix].y + w1*B[ix+1].y, w0*B[ix].z + w1*B[ix+1].z};

    float ux = parts[i].ux + tem * Ep.x;
    float uy = parts[i].uy + tem * Ep.y;
    float uz = parts[i].uz + tem * Ep.z;

    float rg = rsqrtf(1.0f + ux*ux + uy*uy + uz*uz);
    float tx = tem * Bp.x * rg; float ty = tem * Bp.y * rg; float tz = tem * Bp.z * rg;
    float s = 2.0f / (1.0f + tx*tx + ty*ty + tz*tz);

    float ux_ = ux + (uy * tz - uz * ty);
    float uy_ = uy + (uz * tx - ux * tz);
    float uz_ = uz + (ux * ty - uy * tx);

    ux += (uy_ * tz - uz_ * ty) * s + tem * Ep.x;
    uy += (uz_ * tx - ux_ * tz) * s + tem * Ep.y;
    uz += (ux_ * ty - uy_ * tx) * s + tem * Ep.z;

    rg = rsqrtf(1.0f + ux*ux + uy*uy + uz*u

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

typedef struct { float x; int ix; float ux, uy, uz; } t_part;
typedef struct { float x, y, z; } t_fld;

static t_part *d_parts = NULL;
static t_fld  *d_E = NULL, *d_B = NULL, *d_J = NULL;
static int cuda_initialized = 0;

// --- FUNÇÕES EXPOSTAS PARA O C ---
extern "C" {

    void cuda_init(int np, int nx) {
        cudaMalloc(&d_parts, np * sizeof(t_part));
        cudaMalloc(&d_E, (nx + 1) * sizeof(t_fld));
        cudaMalloc(&d_B, (nx + 1) * sizeof(t_fld));
        cudaMalloc(&d_J, (nx + 1) * sizeof(t_fld));
    }

    void cuda_final() {
        if (d_parts) cudaFree(d_parts);
        if (d_E)     cudaFree(d_E);
        if (d_B)     cudaFree(d_B);
        if (d_J)     cudaFree(d_J);
        cuda_initialized = 0;
    }

    // ESTA É A FUNÇÃO QUE FALTAVA E CAUSAVA O ERRO DE LINKAGEM
    void showIterNumber(int iter) {
        // Pode ficar vazia, o main.c apenas precisa que ela exista
    }
}

__global__ void spec_advance_kernel(t_part *parts, int np, const t_fld *E, const t_fld *B, t_fld *J, int nx, float tem, float dt_dx, float qnx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;

    // --- 1. BORIS PUSHER ---
    float x = parts[i].x;
    int ix = parts[i].ix;

    // Proteção básica de memória
    if (ix < 0 || ix >= nx) return;

    float w1 = x; float w0 = 1.0f - w1;

    t_fld Ep = {w0*E[ix].x + w1*E[ix+1].x, w0*E[ix].y + w1*E[ix+1].y, w0*E[ix].z + w1*E[ix+1].z};
    t_fld Bp = {w0*B[ix].x + w1*B[ix+1].x, w0*B[ix].y + w1*B[ix+1].y, w0*B[ix].z + w1*B[ix+1].z};

    float ux = parts[i].ux + tem * Ep.x;
    float uy = parts[i].uy + tem * Ep.y;
    float uz = parts[i].uz + tem * Ep.z;

    float rg = rsqrtf(1.0f + ux*ux + uy*uy + uz*uz);
    float tx = tem * Bp.x * rg; float ty = tem * Bp.y * rg; float tz = tem * Bp.z * rg;
    float s = 2.0f / (1.0f + tx*tx + ty*ty + tz*tz);

    float ux_ = ux + (uy * tz - uz * ty);
    float uy_ = uy + (uz * tx - ux * tz);
    float uz_ = uz + (ux * ty - uy * tx);

    ux += (uy_ * tz - uz_ * ty) * s + tem * Ep.x;
    uy += (uz_ * tx - ux_ * tz) * s + tem * Ep.y;
    uz += (ux_ * ty - uy_ * tx) * s + tem * Ep.z;

    rg = rsqrtf(1.0f + ux*ux + uy*uy + uz*uz);

    // --- 2. DEPOSIÇÃO DE CORRENTE ---
    float vx = ux * rg;
    float vy = uy * rg;
    float vz = uz * rg;

    atomicAdd(&J[ix].y, qnx * vy * w0);
    atomicAdd(&J[ix+1].y, qnx * vy * w1);
    atomicAdd(&J[ix].z, qnx * vz * w0);
    atomicAdd(&J[ix+1].z, qnx * vz * w1);

    float dx = dt_dx * vx;
    float x1 = x + dx;
    int di = (int)floorf(x1);

    float x_mid = x + 0.5f * dx;
    atomicAdd(&J[ix].x, qnx * (dx/dt_dx) * (1.0f - x_mid));

    // --- 3. ATUALIZAR POSIÇÃO ---
    parts[i].x = x1 - di;
    parts[i].ix = ix + di;
    parts[i].ux = ux; parts[i].uy = uy; parts[i].uz = uz;
}

extern "C" void spec_advance_cuda(t_part *h_parts, int np, const t_fld *h_E, const t_fld *h_B, void *h_J, int nx, float tem, float dt_dx, float qnx) {
    if (!cuda_initialized) { cuda_init(np, nx); cuda_initialized = 1; }

    size_t size_p = np * sizeof(t_part);
    size_t size_f = (nx + 1) * sizeof(t_fld);

    cudaMemcpy(d_parts, h_parts, size_p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_f, cudaMemcpyHostToDevice);
    cudaMemset(d_J, 0, size_f);

    int tpb = 256;
    spec_advance_kernel<<<(np+tpb-1)/tpb, tpb>>>(d_parts, np, d_E, d_B, (t_fld*)d_J, nx, tem, dt_dx, qnx);

    cudaMemcpy(h_parts, d_parts, size_p, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_J, d_J, size_f, cudaMemcpyDeviceToHost);
}
