
#include <cuda_runtime.h>
#include <math.h>

typedef struct { float x; int ix; float ux, uy, uz; } t_part;
typedef struct { float x, y, z; } t_fld;

static t_part *d_parts = NULL;
static t_fld  *d_E = NULL, *d_B = NULL;
static t_fld  *d_J = NULL;
static int cuda_initialized = 0;

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
    if (d_J)     cudaFree(d_J);   // <<< FALTAVA
    cuda_initialized = 0;
}

void showIterNumber(int iter) {}   // exigido pelo main.c
}


__global__ void spec_advance_kernel(
    t_part *parts, int np, 
    const t_fld *E, const t_fld *B, t_fld *J, 
    int nx, float tem, float dt_dx, float qnx) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;

    float x = parts[i].x;
    int ix  = parts[i].ix;

    // Verificação de segurança sugerida
    if (ix < 0 || ix >= nx) return;

    float w1 = x;
    float w0 = 1.0f - w1;

    // --- 1. INTERPOLAÇÃO DOS CAMPOS ---
    t_fld Ep = {
        w0*E[ix].x + w1*E[ix+1].x,
        w0*E[ix].y + w1*E[ix+1].y,
        w0*E[ix].z + w1*E[ix+1].z
    };

    t_fld Bp = {
        w0*B[ix].x + w1*B[ix+1].x,
        w0*B[ix].y + w1*B[ix+1].y,
        w0*B[ix].z + w1*B[ix+1].z
    };

    // --- 2. BORIS PUSHER (VELOCIDADES) ---
    float ux = parts[i].ux + tem * Ep.x;
    float uy = parts[i].uy + tem * Ep.y;
    float uz = parts[i].uz + tem * Ep.z;

    float rg = rsqrtf(1.0f + ux*ux + uy*uy + uz*uz);

    float tx = tem * Bp.x * rg;
    float ty = tem * Bp.y * rg;
    float tz = tem * Bp.z * rg;

    float t2 = tx*tx + ty*ty + tz*tz;
    float s  = 2.0f / (1.0f + t2);

    float ux_ = ux + (uy * tz - uz * ty);
    float uy_ = uy + (uz * tx - ux * tz);
    float uz_ = uz + (ux * ty - uy * tx);

    ux += (uy_ * tz - uz_ * ty) * s + tem * Ep.x;
    uy += (uz_ * tx - ux_ * tz) * s + tem * Ep.y;
    uz += (ux_ * ty - uy_ * tx) * s + tem * Ep.z;

    rg = rsqrtf(1.0f + ux*ux + uy*uy + uz*uz);

    // --- 3. DEPOSIÇÃO DE CORRENTE (Sugestão do ChatGPT) ---
    float vx = ux * rg;
    float vy = uy * rg;
    float vz = uz * rg;

    // deslocamento normalizado
    float dx = dt_dx * vx;
    float x_new = x + dx;
    int di = (int)floorf(x_new);

    // posição média (Esirkepov simplificado 1D)
    float x_mid = x + 0.5f * dx;

    // Jx (CRÍTICO)
    atomicAdd(&J[ix].x,     qnx * dx * (1.0f - x_mid));
    atomicAdd(&J[ix + 1].x, qnx * dx * x_mid);

    // Jy
    atomicAdd(&J[ix].y,     qnx * vy * w0);
    atomicAdd(&J[ix + 1].y, qnx * vy * w1);

    // Jz
    atomicAdd(&J[ix].z,     qnx * vz * w0);
    atomicAdd(&J[ix + 1].z, qnx * vz * w1);

    // --- 4. ATUALIZAÇÃO DA POSIÇÃO ---
    parts[i].x  = x_new - di;
    parts[i].ix = ix + di;
    parts[i].ux = ux;
    parts[i].uy = uy;
    parts[i].uz = uz;
}

extern "C" void spec_advance_cuda(t_part *h_parts, int np, const t_fld *h_E, const t_fld *h_B, void *h_J, int nx, float tem, float dt_dx, float qnx) {
    if (!cuda_initialized) { cuda_init(np, nx); cuda_initialized = 1; }

    size_t size_p = np * sizeof(t_part);
    size_t size_f = (nx + 1) * sizeof(t_fld);

    // 1. Copiar dados do CPU para a GPU
    cudaMemcpy(d_parts, h_parts, size_p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_f, cudaMemcpyHostToDevice);

    // 2. Limpar o buffer de corrente na GPU antes de usar
    cudaMemset(d_J, 0, (nx + 1) * sizeof(t_fld));

    // 3. Configurar blocos e threads
    int tpb = 256;
    int bpg = (np + tpb - 1) / tpb;

    // 4. CHAMADA DO KERNEL (Aqui estavam os erros)
    // Precisamos passar d_J e qnx para bater certo com a nova definição
    spec_advance_kernel<<<bpg, tpb>>>(d_parts, np, d_E, d_B, (t_fld*)d_J, nx, tem, dt_dx, qnx);

    // 5. Esperar a GPU acabar e copiar de volta
    cudaDeviceSynchronize();
    cudaMemcpy(h_parts, d_parts, size_p, cudaMemcpyDeviceToHost);
    cudaMemcpy(
        (t_fld*)h_J,
        d_J + 1,                  // <<< DESLOCAMENTO CRÍTICO
        (nx + 1) * sizeof(t_fld),
        cudaMemcpyDeviceToHost
    );
}
