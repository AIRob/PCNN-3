#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "pcnn.h"

double cpuSecond(void){
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void getWeightMatrix(float *weights, int size, float hh){
  for(int y = 0; y < size; y++){
    for(int x = 0; x < size; x++){
      if((x - (size / 2)) == 0 && (y - (size / 2)) == 0){
        /* self connection is 0.0 */
        weights[(y * size) + x] = 0.0;
      }else if((x - (size / 2)) * (x - (size / 2)) + (y - (size / 2)) * (y - (size / 2)) <= (size / 2) * (size / 2)){
        weights[(y * size) + x] = (1.0 / (hh
          * sqrt((y - (size / 2)) * (y - (size / 2))
          + (x - (size / 2)) * (x - (size / 2)))));
      } else {
        weights[(y * size) + x] = 0.0;
      }
    }
  }
}

__device__ void fire_sum_on_gpu(
  float* dev_tmpY,
  unsigned int *fire,
  unsigned int width,
  unsigned int height
){
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx >= width * height){
    return;
  }

  float* impulse = dev_tmpY + blockIdx.x * blockDim.x;

  for(int stride = 1; stride < blockDim.x; stride *= 2){
    int index = 2 * stride * threadIdx.x;
    if(index < width){
      impulse[index] += impulse[index + stride];
    }

    __syncthreads();
  }

  if(tid == 0){
    fire[blockIdx.x] = (int)impulse[0];
  }
}

__global__ void fire_renew_on_gpu(
  float *dev_Y,
  float *dev_tmpY,
  unsigned int *fire,
  unsigned int width,
  unsigned int height
){
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx >= width * height){
    return;
  }

  dev_Y[idx] = dev_tmpY[idx];
  __syncthreads();

  fire_sum_on_gpu(dev_tmpY, fire, width, height);
}

__device__ float weights_work_calc(
  float        *dev_weights,
  float        *dev_Y,
  unsigned int  kernel_size,
  unsigned int  width,
  unsigned int  height
){
  unsigned int x = threadIdx.x;
  unsigned int y = blockIdx.x;

  float W = 0.0;
  for(int i = 0; i < kernel_size; i++){
    for(int j = 0; j < kernel_size; j++){
      if((int)y + (i - ((int)kernel_size/2)) >= 0
         && (int)y + (i - ((int)kernel_size/2)) < (int)height
         && (int)x + (j - ((int)kernel_size/2)) >= 0
         && (int)x + (j - ((int)kernel_size/2)) < (int)width
       ){
        W += dev_weights[(i * kernel_size) + j] * dev_Y[((y + (i - (kernel_size/2))) * width) + (x + (j - (kernel_size/2)))];
      }
    }
  }

  return W;
}

__global__ void pcnn_on_gpu(
  float        *dev_stimu,
  float        *dev_weights,
  float        *dev_F,
  float        *dev_L,
  float        *dev_U,
  float        *dev_T,
  float        *dev_Y,
  float        *dev_tmpY,
  float         beta,
  float         vF,
  float         vL,
  float         vT,
  float         expL,
  float         expT,
  unsigned int  width,
  unsigned int  height,
  unsigned int  kernel_size
){
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx >= width * height){
    return;
  }

  // weighted input from Y values
  float W = 0.0;
  W = weights_work_calc(dev_weights, dev_Y, kernel_size, width, height);
  //printf("x: %d, y: %d, W: %lf\n", threadIdx.x, blockIdx.x, W);

  // Processing threads correspond to image pixels.
  dev_F[idx] = dev_stimu[idx];
  dev_L[idx] = (expL * dev_L[idx]) + (vL * W);
  dev_U[idx] = dev_F[idx] * (1.0 + beta * dev_L[idx]);
  dev_T[idx] = (expT * dev_T[idx]) + (vT * dev_Y[idx]);

  dev_tmpY[idx] = 0.0;
  if(dev_U[idx] > dev_T[idx]){
    dev_tmpY[idx] = 1.0;
  }
}

int pcnn_gpu(
  float *stimu,
  pcnn_params_t *parameter,
  int output_icon_to_file_flag,
  char *icon_to_file,
  int output_images_to_file_flag,
  char *images_to_file
){
  // Initialize
  double iStart;
  double iElaps;

  std::ofstream writing_file;
  if(output_icon_to_file_flag != 0){
    char file_name[100];
    sprintf(file_name, "%s.gpu", icon_to_file);
    writing_file.open(file_name, std::ios::out);

    //writing_file << "t, sum_of_fires, elapsed\n";
    writing_file << "t, sum_of_fires\n";
  }

  char directory_name[100];
  if(output_images_to_file_flag != 0){
    sprintf(directory_name, "%s_gpu", images_to_file);
    struct stat st;
    if(stat(directory_name, &st) != 0){
      mkdir(directory_name, 0755);
    }
  }

  // Init status
  float *F, *L, *U, *T, *Y, *tmpY;
  float expL = exp(-1.0 / parameter->tauL);
  float expT = exp(-1.0 / parameter->tauT);

  F = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  L = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  U = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  T = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  Y = (float*)malloc(sizeof(float) * parameter->width * parameter->height);
  tmpY = (float*)malloc(sizeof(float) * parameter->width * parameter->height);

  // Get kernel
  float *weights;
  weights = (float*)malloc(sizeof(float) * parameter->kernel_size * parameter->kernel_size);
  getWeightMatrix(weights, parameter->kernel_size, parameter->hh);
  
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
  float *dev_stimu, *dev_weights, *dev_F, *dev_L, *dev_U, *dev_T, *dev_Y, *dev_tmpY;
  unsigned int nBytes = sizeof(float) * parameter->width * parameter->height;
  unsigned int nWeightBytes = sizeof(float) * parameter->kernel_size * parameter->kernel_size;

  CHECK(cudaMalloc((float **)&dev_stimu, nBytes));
  CHECK(cudaMalloc((float **)&dev_weights, nWeightBytes));
  CHECK(cudaMalloc((float **)&dev_F, nBytes));
  CHECK(cudaMalloc((float **)&dev_L, nBytes));
  CHECK(cudaMalloc((float **)&dev_U, nBytes));
  CHECK(cudaMalloc((float **)&dev_T, nBytes));
  CHECK(cudaMalloc((float **)&dev_Y, nBytes));
  CHECK(cudaMalloc((float **)&dev_tmpY, nBytes));

  // For parallel fire adding
  unsigned int* fire;
  unsigned int* dev_fire;
  unsigned int fireBytes = sizeof(unsigned int) * parameter->height;
  fire = (unsigned int*)malloc(fireBytes);
  CHECK(cudaMalloc((unsigned int**)&dev_fire, fireBytes));

  // Data transfer from host to device
  CHECK(cudaMemcpy(dev_stimu, stimu, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_weights, weights, nWeightBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_F, F, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_L, L, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_U, U, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_T, T, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_Y, Y, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_tmpY, tmpY, nBytes, cudaMemcpyHostToDevice));

  CHECK(cudaDeviceSynchronize());

  // There are width number of threads in a block
  dim3 pixels_in_line(parameter->width, 1);
  // Number of all data element
  dim3 lines_grid(((parameter->width * parameter->height) + pixels_in_line.x - 1) / pixels_in_line.x, 1);

  // PCNN processing
  for(int t = 0; t < parameter->time_steps; t++){
    iStart = cpuSecond();

    pcnn_on_gpu<<<lines_grid, pixels_in_line>>>(
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
    CHECK(cudaDeviceSynchronize());

    // Data transfer from device to host
    //CHECK(cudaMemcpy(stimu, dev_stimu, nBytes, cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(weights, dev_weights, nWeightBytes, cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(F, dev_F, nBytes, cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(L, dev_L, nBytes, cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(U, dev_U, nBytes, cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(T, dev_T, nBytes, cudaMemcpyDeviceToHost));

    // Get PCNN-Icon
    for(int y = 0; y < parameter->height; y++){
      fire[y] = 0;
    }
    CHECK(cudaMemcpy(dev_fire, fire, fireBytes, cudaMemcpyHostToDevice));
    fire_renew_on_gpu<<<lines_grid, pixels_in_line>>>(
                                       dev_Y,
                                       dev_tmpY,
                                       dev_fire,
                                       parameter->width,
                                       parameter->height
                                       );
    CHECK(cudaDeviceSynchronize());

    //CHECK(cudaMemcpy(Y, dev_Y, nBytes, cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(tmpY, dev_tmpY, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(fire, dev_fire, fireBytes, cudaMemcpyDeviceToHost));

    unsigned int icon = 0;
    std::cout << "t: " << t << "\t";
    for(int y = 0; y < parameter->height; y++){
      icon += fire[y];
    }
    //float2gray_image(tmpY, parameter->width, parameter->height, 1);

    std::cout << "sum_of_fires: " << std::setw(10) << icon << "\t";
    iElaps = cpuSecond() - iStart;
    std::cout << "Elapsed: " << iElaps << std::endl;

    if(output_icon_to_file_flag != 0){
      //writing_file << t << ", " << icon << ", " << iElaps << "\n";
      writing_file << t << ", " << icon << "\n";
    }

    if(output_images_to_file_flag != 0){
      char path[256];
      sprintf(path, "%s/%s_gpu_%03d.png", directory_name, directory_name, t);
      save_float_gray_image(tmpY, parameter->width, parameter->height, path);
    }
  }

  CHECK(cudaFree(dev_stimu));
  CHECK(cudaFree(dev_weights));
  CHECK(cudaFree(dev_F));
  CHECK(cudaFree(dev_L));
  CHECK(cudaFree(dev_U));
  CHECK(cudaFree(dev_T));
  CHECK(cudaFree(dev_Y));
  CHECK(cudaFree(dev_tmpY));
  CHECK(cudaFree(dev_fire));

  CHECK(cudaDeviceReset());

  free(weights);
  free(F);
  free(L);
  free(U);
  free(T);
  free(Y);
  free(tmpY);
  free(fire);

  return 0;
}

int pcnn(
  float *stimu,
  pcnn_params_t* parameter,
  int output_icon_to_file_flag,
  char *icon_to_file,
  int output_images_to_file_flag,
  char *images_to_file
){
  // Initialize
  double iStart;
  double iElaps;

  std::ofstream writing_file;
  if(output_icon_to_file_flag != 0){
    char file_name[100];
    sprintf(file_name, "%s.cpu", icon_to_file);
    writing_file.open(file_name, std::ios::out);
    //writing_file << "t, sum_of_fires, elapsed\n";
    writing_file << "t, sum_of_fires\n";
  }

  char directory_name[256];
  if(output_images_to_file_flag != 0){
    sprintf(directory_name, "%s_cpu", images_to_file);
    struct stat st;
    if(stat(directory_name, &st) != 0){
      mkdir(directory_name, 0755);
    }
  }
  // Get kernel
  float *weights;
  weights = (float*)malloc(sizeof(float) * parameter->kernel_size * parameter->kernel_size);
  getWeightMatrix(weights, parameter->kernel_size, parameter->hh);

  // Init status
  float *F, *L, *U, *T, *Y, *tmpY;
  float expL = exp(-1.0 / parameter->tauL);
  float expT = exp(-1.0 / parameter->tauT);

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
        for(int i = 0; i < (int)parameter->kernel_size; i++){
          for(int j = 0; j < (int)parameter->kernel_size; j++){
            if(y + (i - ((int)parameter->kernel_size/2)) < 0
               || y + (i - ((int)parameter->kernel_size/2)) >= (int)parameter->height
               || x + (j - ((int)parameter->kernel_size/2)) < 0
               || x + (j - ((int)parameter->kernel_size/2)) >= (int)parameter->width
               ){
              continue;
            }

            W += weights[(i * parameter->kernel_size) + j] * Y[((y + (i - (parameter->kernel_size/2))) * parameter->width) + (x + (j - (parameter->kernel_size/2)))];
          }
        }

        F[y * parameter->width + x] = stimu[y * parameter->width + x];
        L[y * parameter->width + x] = (expL * L[y * parameter->width + x]) + (parameter->vL * W);
        U[y * parameter->width + x] = F[y * parameter->width + x] * (1.0 + parameter->beta * L[y * parameter->width + x]);
        T[y * parameter->width + x] = (expT * T[y * parameter->width + x]) + (parameter->vT * Y[y * parameter->width + x]);

        if(U[y * parameter->width + x] > T[y * parameter->width + x]){
          tmpY[y * parameter->width + x] = 1.0;
        } else {
          tmpY[y * parameter->width + x] = 0.0;
        }
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
    std::cout << "sum_of_fires: " << std::setw(10) << icon << "\t";
    iElaps = cpuSecond() - iStart;
    std::cout << "Elapsed: " << iElaps << std::endl;

    if(output_icon_to_file_flag != 0){
      //writing_file << t << ", " << icon << ", " << iElaps << "\n";
      writing_file << t << ", " << icon << "\n";
    }

    if(output_images_to_file_flag != 0){
      char path[256];
      sprintf(path, "%s/%s_cpu_%03d.png", directory_name, directory_name, t);
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
