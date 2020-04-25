/*
 * 
 * Programa de Introducción a los conceptos de CUDA
 * 
 *
 * 
 * 
 */

#include <stdio.h>
#include <stdlib.h>

/* Declaración de métodos/


/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

/* Kernel para sumar dos vectores en un sólo bloque de hilos */
__global__ void vect_add(int *d_a, int *d_b, int *d_out)
{
    /* 
     * Part 2B: Implementación del kernel para realizar la suma de los vectores en el GPU
     * Revisado por Victor
     */
    int idx = threadIdx.x;
    int numA = d_a[idx];
    int numB = d_b[idx];
	d_out[idx] = numA + numB;
}

/* Versión de múltiples bloques de la suma de vectores */
__global__ void vect_add_multiblock(int *d_a, int *d_b, int *d_out)
{
    /* Part 2C: Implementación del kernel pero esta vez permitiendo múltiples bloques de hilos. */
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int numA = d_a[idx];
    int numB = d_b[idx];
	d_out[idx] = numA + numB;
}

/* Numero de elementos en el vector */
#define ARRAY_SIZE 256

/*
 * Número de bloques e hilos
 * Su producto siempre debe ser el tamaño del vector (arreglo).
 */
#define NUM_BLOCKS  1
#define THREADS_PER_BLOCK 256

/* Main routine */
int main(int argc, char *argv[])
{
    int *a, *b, *c; /* Arreglos del CPU */
    int *d_a, *d_b, *d_c;/* Arreglos del GPU */

    int i;
    size_t sz = ARRAY_SIZE * sizeof(int);

    /*
     * Reservar memoria en el cpu
     */
    a = (int *) malloc(sz);
    b = (int *) malloc(sz);
    c = (int *) malloc(sz);

    /*
     * Parte 1A:Reservar memoria en el GPU
     * Revisado por Victor
     */
    cudaMalloc((void**) &d_a, sz);
    cudaMalloc((void**) &d_b, sz);
    cudaMalloc((void**) &d_c, sz);

    /* inicialización */
    for (i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i;
        b[i] = ARRAY_SIZE - i;
        c[i] = 0;
    }

    /* 
     * Parte 1B: Copiar los vectores del CPU al GPU
     * Revisado por Victor
     */
    cudaMemcpy(d_a, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sz, cudaMemcpyHostToDevice);

    /* run the kernel on the GPU */
    /*
     * Parte 2A: Configurar y llamar los kernels
     * Revisado por Victor
     */
    dim3 dimGrid(NUM_BLOCKS, ARRAY_SIZE);
    dim3 dimBlock(THREADS_PER_BLOCK);
    printf("\nSUMA en bloques\n");
    vect_add_multiblock<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
    // printf("\nSUMA simple\n");
    // vect_add<<<1, ARRAY_SIZE>>>(d_a, d_b, d_c);

    /* Esperar a que todos los threads acaben y checar por errores */
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    /* 
     * Part 1C: copiar el resultado de nuevo al CPU
     * Revisado por Victor
     */
    cudaMemcpy(c, d_c, sz, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    /* print out the result */
    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", c[i]);
    }
    printf("\n\n");

    /* 
     * Parte 1D: Liberar los arreglos
     * Revisado por Victor
     */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}


/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
