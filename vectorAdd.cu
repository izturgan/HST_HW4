#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA Kernel для вычисления среднего значения строки
__global__ void calculateRowAverages(const float* matrix, float* averages, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            sum += matrix[row * N + col];
        }
        averages[row] = sum / N;

        // Отладочный вывод
        printf("Row %d: Sum = %f, Average = %f\n", row, sum, averages[row]);
    }
}

int main() {
    int N = 4; // Размер матрицы NxN
    size_t matrixSize = N * N * sizeof(float);
    size_t resultSize = N * sizeof(float);

    // Хостовая память для матрицы и результатов
    float* h_matrix = (float*)malloc(matrixSize);
    float* h_averages = (float*)malloc(resultSize);

    if (!h_matrix || !h_averages) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return EXIT_FAILURE;
    }

    // Чтение матрицы из файла
    FILE* inputFile = fopen("input_matrix.txt", "r");
    if (!inputFile) {
        fprintf(stderr, "Failed to open input file!\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N * N; ++i) {
        if (fscanf(inputFile, "%f", &h_matrix[i]) != 1) {
            fprintf(stderr, "Error reading matrix element at index %d\n", i);
            fclose(inputFile);
            return EXIT_FAILURE;
        }
    }
    fclose(inputFile);

    printf("Matrix on host:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", h_matrix[i * N + j]);
        }
        printf("\n");
    }

    // Выделение памяти на устройстве
    float* d_matrix = nullptr;
    float* d_averages = nullptr;
    cudaMalloc((void**)&d_matrix, matrixSize);
    cudaMalloc((void**)&d_averages, resultSize);

    // Копирование матрицы на устройство
    cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);

    // Конфигурация ядра
    int threadsPerBlock = N; // 4 потока для 4 строк
    int blocksPerGrid = 1;

    // Таймер CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Запуск CUDA ядра
    calculateRowAverages<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_averages, N);

    // Проверка ошибок выполнения ядра
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Копирование результатов с устройства
    cudaMemcpy(h_averages, d_averages, resultSize, cudaMemcpyDeviceToHost);

    // Вывод результатов
    printf("Results from device:\n");
    for (int i = 0; i < N; ++i) {
        printf("Row %d Average: %f\n", i, h_averages[i]);
    }

    // Запись результатов в файл
    FILE* outputFile = fopen("result.txt", "w");
    if (!outputFile) {
        fprintf(stderr, "Failed to open result file for writing!\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        fprintf(outputFile, "Row %d Average: %f\n", i, h_averages[i]);
    }
    fprintf(outputFile, "Execution time: %f ms\n", milliseconds);
    fclose(outputFile);

    // Очистка памяти
    free(h_matrix);
    free(h_averages);
    cudaFree(d_matrix);
    cudaFree(d_averages);

    printf("Done\n");
    return 0;
}
