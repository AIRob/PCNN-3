#ifndef __PCNN_H__
#define __PCNN_H__
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

typedef struct {
  float beta;
  float vF;
  float vL;
  float vT;
  float tauL;
  float tauT;

  int time_steps;
  int kernel_size;
  int width;
  int height;
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
