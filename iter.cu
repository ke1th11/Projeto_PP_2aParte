/*
#include <stdio.h>
#include <cuda.h>

// Este é o Kernel (corre na GPU)
__global__ void myKernel (int *iter) {
    printf("Olá da GPU! Iteração: %d\n", *iter);
}

// Esta é a função que o C vai conseguir chamar
extern "C" void showIterNumber(int i) {
    int *di;
    cudaMalloc((void**) &di, sizeof(int));
    cudaMemcpy(di, &i, sizeof(int), cudaMemcpyHostToDevice);

    myKernel<<<1,1>>>(di);

    cudaDeviceSynchronize(); // Garante que a GPU termina antes de continuar
    cudaFree(di);
}
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h> // Necessário para o float3 e make_float3
#include <math.h>

// 1. Estrutura de partícula alinhada com o particles.h
typedef struct {
    int ix;         
    float x;        
    float ux, uy, uz; 
} t_part_gpu;

// 2. Kernel atualizado usando tipos nativos
__global__ void spec_advance_kernel(t_part_gpu *parts, int np, float tem, float dt_dx) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < np) {
        // Exemplo de como usarias o float3 do CUDA aqui dentro no futuro:
        // float3 velocity = make_float3(parts[i].ux, parts[i].uy, parts[i].uz);
        
        float ux = parts[i].ux;
        float uy = parts[i].uy;
        float uz = parts[i].uz;

        // Cálculo da nova posição (Push)
        float rg = 1.0f / sqrtf(1.0f + ux*ux + uy*uy + uz*uz);
        float dx = dt_dx * rg * ux;
        
        float x1 = parts[i].x + dx;
        int di = floorf(x1); 
        
        parts[i].x = x1 - di;
        parts[i].ix += di;
    }
}

// 3. Wrapper para o C
extern "C" void spec_advance_cuda(t_part_gpu *h_parts, int np, float tem, float dt_dx) {
    t_part_gpu *d_parts;
    size_t size = np * sizeof(t_part_gpu);

    cudaMalloc((void**)&d_parts, size);
    cudaMemcpy(d_parts, h_parts, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (np + threadsPerBlock - 1) / threadsPerBlock;

    spec_advance_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_parts, np, tem, dt_dx);

    cudaDeviceSynchronize();
    cudaMemcpy(h_parts, d_parts, size, cudaMemcpyDeviceToHost);

    cudaFree(d_parts);
}

// Limpeza da função antiga (podes comentar ou remover se o main.c já não a usar)
extern "C" void showIterNumber(int i) {
    // printf("Iteração: %d\n", i);
}

