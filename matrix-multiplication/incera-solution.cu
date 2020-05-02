/*
 * Plantilla para la multiplicación de matrices
 * con memoria compartida
 * Jose Incera. Adaptado del código
 * de Robert Hochberg
 * Abril 2016
 *
 * Based nearly entirely on the code from the CUDA C Programming Guide
 */


#include <stdio.h>

// Estructura Matriz.
typedef struct{
    int nRen;
    int nCol;
    float *elementos;
    int salto; // stride para recorrer columnas
} Matriz;

// dimensión de un bloque
// El tamaño es TAM_BLOQUE * TAM_BLOQUE
#define TAM_BLOQUE 16

// Prototipo de función
__global__ void MatMultKernel(const Matriz, const Matriz, Matriz);

// Por facilidad, las dimensiones de la matriz son múltiplos de TAM_BLOQUE
void MatMult(const Matriz A, const Matriz B, Matriz C) {
    
    // Carga A y B en memoria GPU
    Matriz d_A;
    d_A.nRen = d_A.salto = A.nRen;
    d_A.nCol = A.nCol;
    size_t tam= A.nRen * A.nCol * sizeof(float);
    
    cudaError_t err = cudaMalloc();  //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
    cudaMemcpy(); //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
    
    Matriz d_B;
    d_B.nRen = d_B.salto = B.nRen;
    d_B.nCol = B.nCol;
    tam= B.nRen * B.nCol * sizeof(float);
    
    cudaMalloc( ); //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
    cudaMemcpy( );  //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
    
    // Asigna espacio para C en GPU
    Matriz d_C;
    d_C.nRen = d_C.salto = C.nRen;
    d_C.nCol = C.nCol;
    tam = C.nRen * C.nCol * sizeof(float);
    cudaMalloc( );  //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
    
    // Llama al kernel
    dim3 dimBlock(TAM_BLOQUE, TAM_BLOQUE);
    dim3 dimGrid(B.nRen / dimBlock.x, A.nCol / dimBlock.y);
    
    //  Descomenta y AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
    // MatMultKernel<<<  ,   >>>();
    
    // Espera a que todos terminen
    cudaThreadSynchronize();
    
    // Lee C from del GPU
    cudaMemcpy( );//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
    
    // Libera memoria GPU
    cudaFree();//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
    cudaFree();//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
    cudaFree();//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
}

// Toma un elemento de la matriz
__device__ float GetElement(const Matriz A, int ren, int col) {
    return A.elementos[ren* A.salto + col];
}

// Pon un elemento en la matriz
__device__ void SetElement(Matriz A, int ren, int col, float value) {
    A.elementos[ren* A.salto + col] = value;
}

// Toma una submatriz de A de tamaño TAM_BLOQUExTAM_BLOQUE
// localizada col sub-matrices a la derecha y ren sub-matrices abajo
// desde la esquina superior izquierda
__device__ Matriz LeeSubMatriz(Matriz A, int ren, int col) {
    Matriz Asub;
    Asub.nRen = TAM_BLOQUE;
    Asub.nCol = TAM_BLOQUE;
    Asub.salto = A.salto;
    Asub.elementos = &A.elementos[A.salto * TAM_BLOQUE * ren+ TAM_BLOQUE * col];
    return Asub;
}


// Kernel multiplicación de Matriz
__global__ void MatMultKernel(Matriz A, Matriz B, Matriz C) {
    
    // Renglon y columna del bloque
    int blockRen = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Cada bloque calcula una submatriz Csub de C
        Matriz Csub = LeeSubMatriz(C, blockRen, blockCol);
    
    // Cada thread calcula un elemento de Csub
    // acumulando elementos en valorC
    float valorC= 0.0;
    
    // Thread ren y col dentro de Csub
    int ren = threadIdx.y;
    int col = threadIdx.x;
    
    // Loop sobre todas las sub-matrices de A y B necesarias
    // para calcular Csub
    // Multiplica cada par de sub-matrices y acumula resultados
    for (int m = 0; m < (A.nRen / TAM_BLOQUE); ++m) {

        // Toma sub-Matriz Asub de A
        Matriz Asub = LeeSubMatriz( );//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
        
        // Toma sub-Matriz Bsub de B
        Matriz Bsub = LeeSubMatriz( );//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
        
        // La memoria compartida donde se almacenan Asub y Bsub
        __shared__ float As[TAM_BLOQUE][TAM_BLOQUE];
        __shared__ float Bs[TAM_BLOQUE][TAM_BLOQUE];
        
        // Transfiere Asub y Bsub de memoria global a shared
        // Cada thread carga un elemento de cada submatriz
        As[ren][col] = GetElement(Asub, ren, col);
        Bs[ren][col] = GetElement(Bsub, ren, col);
        
        // Punto de sincronización: Espera a que todas las
        // sub-matrices se hayan cargado antes de continuar
        __syncthreads();
        
        // Multiplica Asub y Bsub
        for (int e = 0; e < TAM_BLOQUE; ++e)
            
          // Descomenta y agrega la operación apropiada
         //   valorC += ;
        
        // Punto de sincronización antes de iniciar otra iteración
        __syncthreads();
    }
    
    // Escribe Csub a memoria global
    // Cada thread escribe un elemento
    SetElement(Csub, ren, col, valorC);
}


int main(int argc, char* argv[]){
    
    clock_t begin=clock();  // Para medir cuánto tarda

    Matriz A, B, C;
    int a1, a2, b1, b2;
    a1 = atoi(argv[1]);			/* nCol de A */
    a2 = atoi(argv[2]);			/* nRen  de A */
    b1 = a2;		         	/* nCol de B */
    b2 = atoi(argv[3]);			/* nRen  de B */
    
    A.nCol = a1;
    A.nRen = a2;
    A.elementos = (float*)malloc(A.nRen * A.nCol * sizeof(float));
    
    B.nCol = b1;
    B.nRen = b2;
    B.elementos = (float*)malloc(B.nRen * B.nCol * sizeof(float));
    
    C.nCol = A.nCol;
    C.nRen = B.nRen;
    C.elementos = (float*)malloc(C.nRen * C.nCol * sizeof(float));
    
    // Llena las matrices con valores aleatorios
    for(int i = 0; i < A.nCol; i++)
        for(int j = 0; j < A.nRen; j++)
            A.elementos[i*A.nRen + j] = (rand() % 3);
    
    for(int i = 0; i < B.nCol; i++)
        for(int j = 0; j < B.nRen; j++)
            B.elementos[i*B.nRen + j] = (rand() % 2);
    
    MatMult(A, B, C);
    
    clock_t end=clock();  // Checa el tiempo inmediatamente después de terminar
    
    double diffticks=end-begin;
    double diffms=(diffticks*10)/CLOCKS_PER_SEC;
    
    printf("Tiempo usado: %f\n\n", diffms);

    // Imprime hasta porciones de 10x10 de las tres matrices
  /*
   for(int i = 0; i < min(10, A.nCol); i++){
        for(int j = 0; j < min(10, A.nRen); j++)
            printf("%f ", A.elementos[i*A.nRen + j]);
        printf("\n");
    }
    printf("\n");
    for(int i = 0; i < min(10, B.nCol); i++){
        for(int j = 0; j < min(10, B.nRen); j++)
            printf("%f ", B.elementos[i*B.nRen + j]);
        printf("\n");
    }
    printf("\n");
    for(int i = 0; i < min(10, C.nCol); i++){
        for(int j = 0; j < min(10, C.nRen); j++)
            printf("%f ", C.elementos[i*C.nRen + j]);
        printf("\n");
    }
    printf("\n");
    */
}