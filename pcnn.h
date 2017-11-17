#ifndef __PCNN_H__
#define __PCNN_H__
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#define CHECK(call) {                                                   \
  const cudaError_t error = call;                                       \
  if(error != cudaSuccess){                                             \
    std::cout << "Error: " << __FILE__ << " " << __LINE__ << std::endl; \
    std::cout << "code: " << error << std::endl;                        \
    std::cout << "reason: " << cudaGetErrorString(error) << std::endl;  \
    exit(1);                                                            \
  }                                                                     \
}                                                                       \

typedef struct {
  float beta;
  float vF;
  float vL;
  float vT;
  float tauL;
  float tauT;
  float hh;

  unsigned int time_steps;
  unsigned int kernel_size;
  unsigned int width;
  unsigned int height;
} pcnn_params_t;

float* image2stimuF(char* filename, pcnn_params_t* parameter);
cv::Mat* float2gray_image(float* data, int row, int col);
int save_float_gray_image(float* data, int row, int col, char* filepath);
int pcnn(
  float* stimu,
  pcnn_params_t* parameter,
  int output_icon_to_file_flag,
  char* icon_to_file,
  int output_images_to_file_flag,
  char* images_to_file
);
int pcnn_gpu(
  float* stimu,
  pcnn_params_t* parameter,
  int output_icon_to_file_flag,
  char* icon_to_file,
  int output_images_to_file_flag,
  char* images_to_file
);
#endif
