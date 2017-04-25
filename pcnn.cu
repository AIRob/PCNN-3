#include <iostream>
#include <fstream>
#include <string>

#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "pcnn.h"

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void pcnn_on_gpu(
  float* dev_stimu,
  float* dev_weights,
  float* dev_F,
  float* dev_L,
  float* dev_U,
  float* dev_T,
  float* dev_Y,
  float* dev_tmpY,
  float  beta,
  float  vF,
  float  vL,
  float  vT,
  float  expL,
  float  expT,
  int    width,
  int    height,
  int    kernel_size
){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  // weighted input from Y values
  float W = 0.0;
  for(int i = 0; i < kernel_size; i++){
    for(int j = 0; j < kernel_size; j++){
      if(y + (i - (kernel_size/2)) < 0
         || y + (i - (kernel_size/2)) >= height
         || x + (j - (kernel_size/2)) < 0
         || x + (j - (kernel_size/2)) >= width
         ){
        continue;
      }
      W += dev_weights[(i * kernel_size) + j] * dev_Y[((y + (i - (kernel_size/2))) * width) + (x + (j - (kernel_size/2)))];
    }
  }

  //printf("x: %d, y: %d, W: %lf\n", x, y, W);

  dev_F[y * width + x] = dev_stimu[y * width + x];
  dev_L[y * width + x] = expL * dev_L[y * width + x] + (vL * W);
  dev_U[y * width + x] = dev_F[y * width + x] * (1.0 + beta * dev_L[y * width + x]);
        
  if(dev_U[y * width + x] > dev_T[y * width + x]){
    dev_tmpY[y * width + x] = 1.0;
  } else {
    dev_tmpY[y * width + x] = 0.0;
  }

  dev_T[y * width + x] = (expT * dev_T[y * width + x]) + (vT * dev_Y[y * width + x]);
  __syncthreads();
}

int pcnn_gpu(
  float* stimu,
  pcnn_params_t* parameter,
  int output_icon_to_file_flag,
  char* icon_to_file,
  int output_images_to_file_flag,
  char* images_to_file
){
  // Initialize
  double iStart;
  double iElaps;

  std::ofstream writing_file;
  if(output_icon_to_file_flag != 0){
    char file_name[100];
    sprintf(file_name, "gpu_%s", icon_to_file);
    writing_file.open(file_name, std::ios::out);

    writing_file << "t, sum_of_fires, elapsed\n";
  }

  char directory_name[100];
  if(output_images_to_file_flag != 0){
    sprintf(directory_name, "gpu_%s", images_to_file);
    struct stat st;
    if(stat(directory_name, &st) != 0){
      mkdir(directory_name, 0755);
    }
  }

  // Get kernel
  float* weights;
  weights = (float*)malloc(sizeof(float) * parameter->kernel_size * parameter->kernel_size);
  for(int y = 0; y < parameter->kernel_size; y++){
    for(int x = 0; x < parameter->kernel_size; x++){
      weights[(y * parameter->kernel_size) + x] = 0.01 * (1.0 / (1.0 + sqrt(y^2 + x^2)));
    }
  }

  // Init status
  float *F, *L, *U, *T, *Y, *tmpY;
  float expL = exp(parameter->tauL);
  float expT = exp(parameter->tauT);

  F = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  L = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  U = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  T = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  Y = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  tmpY = (float*)malloc(sizeof(float) * parameter->width * parameter->height);

  for(int y = 0; y < parameter->height; y++){
    for(int x = 0; x < parameter->width; x++){
      F[(y * parameter->width) + x] = 0.0;
      L[(y * parameter->width) + x] = 0.0;
      U[(y * parameter->width) + x] = 0.0;
      T[(y * parameter->width) + x] = 0.0;
      Y[(y * parameter->width) + x] = 0.0;
      tmpY[(y * parameter->width) + x] = 0.0;
    }
  }

  // Init for CUDA
  float *dev_stimu;
  float *dev_weights;
  float *dev_F;
  float *dev_L;
  float *dev_U;
  float *dev_T;
  float *dev_Y;
  float *dev_tmpY;
  int   nBytes = sizeof(float) * parameter->width * parameter->height;
  int   nWeightBytes = sizeof(float) * parameter->kernel_size * parameter->kernel_size;

  cudaMalloc((float **)&dev_stimu, nBytes);
  cudaMalloc((float **)&dev_weights, nWeightBytes);
  cudaMalloc((float **)&dev_F, nBytes);
  cudaMalloc((float **)&dev_L, nBytes);
  cudaMalloc((float **)&dev_U, nBytes);
  cudaMalloc((float **)&dev_T, nBytes);
  cudaMalloc((float **)&dev_Y, nBytes);
  cudaMalloc((float **)&dev_tmpY, nBytes);

  // PCNN processing
  for(int t = 0; t < parameter->time_steps; t++){
    iStart = cpuSecond();

    // Data transfer from host to device
    cudaMemcpy(dev_stimu, stimu, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weights, weights, nWeightBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_F, F, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_L, L, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_U, U, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_T, T, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Y, Y, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_tmpY, tmpY, nBytes, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    dim3 block(32, 32);
    dim3 grid((parameter->width + block.x - 1) / block.x,
              (parameter->height + block.y - 1) / block.y);

    pcnn_on_gpu<<<grid, block>>>(
                              dev_stimu,
                              dev_weights,
                              dev_F,
                              dev_L,
                              dev_U,
                              dev_T,
                              dev_Y,
                              dev_tmpY,
                              parameter->beta,
                              parameter->vF,
                              parameter->vL,
                              parameter->vT,
                              expL,
                              expT,
                              parameter->width,
                              parameter->height,
                              parameter->kernel_size
                              );
    cudaError_t error = cudaDeviceSynchronize();
    if ( error != cudaSuccess){
      std::cout << "Error: " << __FILE__ << __LINE__;
      std::cout << "code: " << error << "reason: "<< cudaGetErrorString(error) << std::endl;
    }

    // Data transfer from device to host
    //cudaMemcpy(stimu, dev_stimu, nBytes, cudaMemcpyDeviceToHost);
    //cudaMemcpy(weights, dev_weights, nWeightBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(F, dev_F, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(L, dev_L, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(U, dev_U, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(T, dev_T, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Y, dev_Y, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpY, dev_tmpY, nBytes, cudaMemcpyDeviceToHost);

    // Get PCNN-Icon
    int icon = 0;
    std::cout << "t: " << t << "\t";
    //float2gray_image(tmpY, parameter->width, parameter->height, 1);
    for(int y = 0; y < parameter->height; y++){
      for(int x = 0; x < parameter->width; x++){
        Y[y * parameter->width + x] = tmpY[y * parameter->width + x];
        if(tmpY[y * parameter->width + x] > 0.0){
          icon++;
        }
      }
    }
    std::cout << "sum_of_fires: " << icon;
    iElaps = cpuSecond() - iStart;
    std::cout << " Elapsed: " << iElaps << std::endl;

    if(output_icon_to_file_flag != 0){
      writing_file << t << ", " << icon << ", " << iElaps << "\n";
    }

    if(output_images_to_file_flag != 0){
      char path[256];
      sprintf(path, "%s/%s_%3d.png", directory_name, directory_name, t);
      save_float_gray_image(tmpY, parameter->width, parameter->height, path);
    }
  }

  cudaFree(dev_stimu);
  cudaFree(dev_weights);
  cudaFree(dev_F);
  cudaFree(dev_L);
  cudaFree(dev_U);
  cudaFree(dev_T);
  cudaFree(dev_Y);
  cudaFree(dev_tmpY);

  cudaDeviceReset();

  free(weights);
  free(F);
  free(L);
  free(U);
  free(T);
  free(Y);
  free(tmpY);

  return 0;
}

int pcnn(
  float* stimu,
  pcnn_params_t* parameter,
  int output_icon_to_file_flag,
  char* icon_to_file,
  int output_images_to_file_flag,
  char* images_to_file
){
  // Initialize
  double iStart;
  double iElaps;

  std::ofstream writing_file;
  if(output_icon_to_file_flag != 0){
    char file_name[100];
    sprintf(file_name, "cpu_%s", icon_to_file);
    writing_file.open(file_name, std::ios::out);
    writing_file << "t, sum_of_fires, elapsed\n";
  }

  char directory_name[256];
  if(output_images_to_file_flag != 0){
    sprintf(directory_name, "cpu_%s", images_to_file);
    struct stat st;
    if(stat(directory_name, &st) != 0){
      mkdir(directory_name, 0755);
    }
  }
  // Get kernel
  float* weights;
  weights = (float*)malloc(sizeof(float) * parameter->kernel_size * parameter->kernel_size);
  for(int y = 0; y < parameter->kernel_size; y++){
    for(int x = 0; x < parameter->kernel_size; x++){
      weights[(y * parameter->kernel_size) + x] = 0.01 * (1.0 / (1.0 + sqrt(y^2 + x^2)));
    }
  }

  // Init status
  float *F, *L, *U, *T, *Y, *tmpY;
  float expL = exp(parameter->tauL);
  float expT = exp(parameter->tauT);

  F = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  L = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  U = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  T = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  Y = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  tmpY = (float*)malloc(sizeof(float) * parameter->width * parameter->height);

  for(int y = 0; y < parameter->height; y++){
    for(int x = 0; x < parameter->width; x++){
      F[(y * parameter->width) + x] = 0.0;
      L[(y * parameter->width) + x] = 0.0;
      U[(y * parameter->width) + x] = 0.0;
      T[(y * parameter->width) + x] = 0.0;
      Y[(y * parameter->width) + x] = 0.0;
      tmpY[(y * parameter->width) + x] = 0.0;
    }
  }

  // PCNN processing
  for(int t = 0; t < parameter->time_steps; t++){
    iStart = cpuSecond();
    for(int y = 0; y < parameter->height; y++){
      for(int x = 0; x < parameter->width; x++){

        // weighted input from Y values
        float W = 0.0;
        for(int i = 0; i < parameter->kernel_size; i++){
          for(int j = 0; j < parameter->kernel_size; j++){
            if(y + (i - (parameter->kernel_size/2)) < 0
               || y + (i - (parameter->kernel_size/2)) >= parameter->height
               || x + (j - (parameter->kernel_size/2)) < 0
               || x + (j - (parameter->kernel_size/2)) >= parameter->width
               ){
              continue;
            }

            W += weights[(i * parameter->kernel_size) + j] * Y[((y + (i - (parameter->kernel_size/2))) * parameter->width) + (x + (j - (parameter->kernel_size/2)))];
          }
        }

        F[y * parameter->width + x] = stimu[y * parameter->width + x];
        L[y * parameter->width + x] = expL * L[y * parameter->width + x] + (parameter->vL * W);
        U[y * parameter->width + x] = F[y * parameter->width + x] * (1.0 + parameter->beta * L[y * parameter->width + x]);
        
        if(U[y * parameter->width + x] > T[y * parameter->width + x]){
          tmpY[y * parameter->width + x] = 1.0;
        } else {
          tmpY[y * parameter->width + x] = 0.0;
        }

        T[y * parameter->width + x] = (expT * T[y * parameter->width + x]) + (parameter->vT * Y[y * parameter->width + x]);
      }
    }

    // Get PCNN-Icon
    int icon = 0;
    std::cout << "t: " << t << "\t";
    //float2gray_image(tmpY, parameter->width, parameter->height, 1);
    for(int y = 0; y < parameter->height; y++){
      for(int x = 0; x < parameter->width; x++){
        Y[y * parameter->width + x] = tmpY[y * parameter->width + x];
        if(tmpY[y * parameter->width + x] > 0.0){
          icon++;
        }
      }
    }
    std::cout << "sum_of_fires: " << icon;
    iElaps = cpuSecond() - iStart;
    std::cout << " Elapsed: " << iElaps << std::endl;

    if(output_icon_to_file_flag != 0){
      writing_file << t << ", " << icon << ", " << iElaps << "\n";
    }

    if(output_images_to_file_flag != 0){
      char path[256];
      sprintf(path, "%s/%s_%3d.png", directory_name, directory_name, t);
      save_float_gray_image(tmpY, parameter->width, parameter->height, path);
    }
  }

  free(weights);
  free(F);
  free(L);
  free(U);
  free(T);
  free(Y);
  free(tmpY);

  return 0;
}
