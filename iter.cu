/*
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Definir a estrutura exatamente como está no C para o NVCC não se perder
typedef struct {
    float x;
    int ix;
    float ux, uy, uz;
} t_part;

// Kernel que será executado na GPU
__global__ void spec_advance_kernel(t_part *parts, int np, float tem, float dt_dx) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < np) {
        float ux, uy, uz, u2, rg, dx, x1;
        
        // 1. Carregar dados
        ux = parts[i].ux;
        uy = parts[i].uy;
        uz = parts[i].uz;

        // 2. Cálculo de translação simples (Push)
        u2 = ux*ux + uy*uy + uz*uz;
        rg = 1.0f / rsqrtf(1.0f + u2); // rsqrtf é mais rápido em CUDA

        dx = dt_dx * ux * rg;
        x1 = parts[i].x + dx;

        // 3. Lógica de célula (equivalente ao ltrim/floor)
        int di = floorf(x1); 
        
        // 4. Guardar resultados
        parts[i].x = x1 - di;
        parts[i].ix += di;
    }
}

// Wrapper para o C chamar
extern "C" void spec_advance_cuda(t_part *h_parts, int np, float tem, float dt_dx) {
    t_part *d_parts;
    size_t size = np * sizeof(t_part);

    // Alocação e Cópia
    cudaMalloc((void**)&d_parts, size);
    cudaMemcpy(d_parts, h_parts, size, cudaMemcpyHostToDevice);

    // Lançamento do Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (np + threadsPerBlock - 1) / threadsPerBlock;

    spec_advance_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_parts, np, tem, dt_dx);

    // Check de erros (muito útil para debugging)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Cópia de volta e limpeza
    cudaMemcpy(h_parts, d_parts, size, cudaMemcpyDeviceToHost);
    cudaFree(d_parts);
}

extern "C" void showIterNumber(int iter) {
    // Esta função apenas serve para imprimir o número da iteração,
    // se quiseres que ela faça algo. Se não, deixa-a vazia.
    // printf("Iteration: %d\n", iter);
}
*/
/*
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Estruturas para compatibilidade com o ZPIC
typedef struct {
    float x;
    int ix;
    float ux, uy, uz;
} t_part;

typedef struct {
    float x, y, z;
} t_fld;

// Kernel com o Boris Pusher
__global__ void spec_advance_kernel(t_part *parts, int np,
                                   const t_fld *E, const t_fld *B,
                                   int nx, float tem, float dt_dx) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < np) {
        // 1. Obter posição e índices da partícula
        float x = parts[i].x;
        int ix = parts[i].ix;

        // 2. Interpolação Linear dos campos (E e B) no ponto da partícula
        // Simplificado para 1D (usando x e x+1)
        float w1 = x;
        float w0 = 1.0f - w1;

        t_fld Ep, Bp;
        Ep.x = w0 * E[ix].x + w1 * E[ix+1].x;
        Ep.y = w0 * E[ix].y + w1 * E[ix+1].y;
        Ep.z = w0 * E[ix].z + w1 * E[ix+1].z;

        Bp.x = w0 * B[ix].x + w1 * B[ix+1].x;
        Bp.y = w0 * B[ix].y + w1 * B[ix+1].y;
        Bp.z = w0 * B[ix].z + w1 * B[ix+1].z;

        // 3. BORIS PUSHER (O algoritmo real)
        float ux = parts[i].ux;
        float uy = parts[i].uy;
        float uz = parts[i].uz;

        // u- (Meio passo do campo elétrico)
        ux += tem * Ep.x;
        uy += tem * Ep.y;
        uz += tem * Ep.z;

        // Cálculo de Gamma (relativista)
        float u2 = ux*ux + uy*uy + uz*uz;
        float rg = rsqrtf(1.0f + u2);

        // Rotação Magnética (u- para u+)
        float tx = tem * Bp.x * rg;
        float ty = tem * Bp.y * rg;
        float tz = tem * Bp.z * rg;

        float t2 = tx*tx + ty*ty + tz*tz;
        float s = 2.0f / (1.0f + t2);

        float ux_ = ux + (uy * tz - uz * ty);
        float uy_ = uy + (uz * tx - ux * tz);
        float uz_ = uz + (ux * ty - uy * tx);

        ux += (uy_ * tz - uz_ * ty) * s;
        uy += (uz_ * tx - ux_ * tz) * s;
        uz += (ux_ * ty - uy_ * tx) * s;

        // u+ (Segundo meio passo do campo elétrico)
        ux += tem * Ep.x;
        uy += tem * Ep.y;
        uz += tem * Ep.z;

        // 4. Atualizar Posição (Push)
        u2 = ux*ux + uy*uy + uz*uz;
        rg = 1.0f / sqrtf(1.0f + u2);

        float dx = dt_dx * ux * rg;
        float x1 = x + dx;
        int di = floorf(x1);

        // 5. Guardar resultados
        parts[i].x = x1 - di;
        parts[i].ix += di;
        parts[i].ux = ux;
        parts[i].uy = uy;
        parts[i].uz = uz;
    }
}

// Wrapper atualizado para receber os campos
extern "C" void spec_advance_cuda(t_part *h_parts, int np, 
                                 void* h_E, void* h_B, int nx,
                                 float tem, float dt_dx) {
    t_part *d_parts;
    t_fld *d_E, *d_B;
    cudaError_t err;

    size_t size_p = np * sizeof(t_part);
    size_t size_f = (nx + 1) * sizeof(t_fld);

    // Verificar se a alocação funciona
    err = cudaMalloc(&d_parts, size_p);
    if (err != cudaSuccess) { 
        printf("ERRO CUDA (Malloc): %s\n", cudaGetErrorString(err)); 
        return; 
    }

    cudaMalloc(&d_E, size_f);
    cudaMalloc(&d_B, size_f);

    // Copiar Dados
    cudaMemcpy(d_parts, h_parts, size_p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_f, cudaMemcpyHostToDevice);

    // Lançar Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (np + threadsPerBlock - 1) / threadsPerBlock;
    spec_advance_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_parts, np, d_E, d_B, nx, tem, dt_dx);

    // VERIFICAR SE O KERNEL FALHOU
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERRO KERNEL: %s\n", cudaGetErrorString(err));
    }

    // Copiar de volta
    cudaMemcpy(h_parts, d_parts, size_p, cudaMemcpyDeviceToHost);
    
    // Debug para ver se mudou algo
    printf("Debug: Particula 0 x = %f, ix = %d\n", h_parts[0].x, h_parts[0].ix);

    cudaFree(d_parts);
    cudaFree(d_E);
    cudaFree(d_B);
}

extern "C" void showIterNumber(int iter) {
    // Pode ficar vazio, serve apenas para o Linker não reclamar
}
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

typedef struct { float x; int ix; float ux, uy, uz; } t_part;
typedef struct { float x, y, z; } t_fld;

// Ponteiros Globais Estáticos (Persistentes na GPU)
static t_part *d_parts = NULL;
static t_fld  *d_E = NULL, *d_B = NULL;

// --- FUNÇÃO DE INICIALIZAÇÃO (Chamar uma vez no início) ---
extern "C" void cuda_init(int np, int nx) {
    cudaMalloc(&d_parts, np * sizeof(t_part));
    cudaMalloc(&d_E, (nx + 1) * sizeof(t_fld));
    cudaMalloc(&d_B, (nx + 1) * sizeof(t_fld));
}

// --- FUNÇÃO DE LIMPEZA (Chamar uma vez no fim) ---
extern "C" void cuda_final() {
    if (d_parts) cudaFree(d_parts);
    if (d_E)     cudaFree(d_E);
    if (d_B)     cudaFree(d_B);
}

// O Teu Kernel (Mantido exatamente como tinhas)
__global__ void spec_advance_kernel(t_part *parts, int np, const t_fld *E, const t_fld *B, int nx, float tem, float dt_dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < np) {
        float x = parts[i].x;
        int ix = parts[i].ix;
        float w1 = x;
        float w0 = 1.0f - w1;

        t_fld Ep, Bp;
        Ep.x = w0 * E[ix].x + w1 * E[ix+1].x;
        Ep.y = w0 * E[ix].y + w1 * E[ix+1].y;
        Ep.z = w0 * E[ix].z + w1 * E[ix+1].z;

        Bp.x = w0 * B[ix].x + w1 * B[ix+1].x;
        Bp.y = w0 * B[ix].y + w1 * B[ix+1].y;
        Bp.z = w0 * B[ix].z + w1 * B[ix+1].z;

        float ux = parts[i].ux;
        float uy = parts[i].uy;
        float uz = parts[i].uz;

        ux += tem * Ep.x; uy += tem * Ep.y; uz += tem * Ep.z;

        float u2 = ux*ux + uy*uy + uz*uz;
        float rg = rsqrtf(1.0f + u2);

        float tx = tem * Bp.x * rg; float ty = tem * Bp.y * rg; float tz = tem * Bp.z * rg;
        float t2 = tx*tx + ty*ty + tz*tz;
        float s = 2.0f / (1.0f + t2);

        float ux_ = ux + (uy * tz - uz * ty);
        float uy_ = uy + (uz * tx - ux * tz);
        float uz_ = uz + (ux * ty - uy * tx);

        ux += (uy_ * tz - uz_ * ty) * s;
        uy += (uz_ * tx - ux_ * tz) * s;
        uz += (ux_ * ty - uy_ * tx) * s;

        ux += tem * Ep.x; uy += tem * Ep.y; uz += tem * Ep.z;

        u2 = ux*ux + uy*uy + uz*uz;
        rg = 1.0f / sqrtf(1.0f + u2);

        float dx = dt_dx * ux * rg;
        float x1 = x + dx;
        int di = floorf(x1);

        parts[i].x = x1 - di;
        parts[i].ix += di;
        parts[i].ux = ux;
        parts[i].uy = uy;
        parts[i].uz = uz;
    }
}

// Wrapper atualizado SEM Mallocs internos
extern "C" void spec_advance_cuda(t_part *h_parts, int np, void* h_E, void* h_B, int nx, float tem, float dt_dx) {
    size_t size_p = np * sizeof(t_part);
    size_t size_f = (nx + 1) * sizeof(t_fld);

    // Apenas Copiar
    cudaMemcpy(d_parts, h_parts, size_p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_f, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (np + threadsPerBlock - 1) / threadsPerBlock;
    spec_advance_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_parts, np, d_E, d_B, nx, tem, dt_dx);

    // Copiar de volta
    cudaMemcpy(h_parts, d_parts, size_p, cudaMemcpyDeviceToHost);
}

extern "C" void showIterNumber(int iter) {
    // Deixa vazio. Serve apenas para o compilador não reclamar.
}
