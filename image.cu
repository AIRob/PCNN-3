#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include "pcnn.h"

int save(cv::Mat* image, char* filepath){
  cv::imwrite(filepath, *image);

  return 0;
}

int view(cv::Mat* image){
  cv::namedWindow("view", cv::WINDOW_AUTOSIZE);

  cv::imshow("view", *image);

  cv::waitKey(33);

  //cv:: destroyAllWindows();

  return 0;
}

cv::Mat* float2gray_image(float* data, int row, int col, int view_flag){
  // Check for is image reading correct?
  static cv::Mat image(cv::Size(col, row), cv::IMREAD_GRAYSCALE);
  for(int y = 0; y < row; y++){
    for(int x = 0; x < col; x++){
      image.data[y * image.step + x * image.elemSize()] = (char)(data[(y * col) + x] * 255.0);
    }
  }
  if(view_flag != 0){
    view(&image);
  }

  return &image;
}

int save_float_gray_image(float* data, int col, int row, char* filepath){
  cv::Mat *image = float2gray_image(data, row, col, 0);

  save(image, filepath);

  return 0;
}

float* image2stimuF(char* filename, pcnn_params_t* parameter){
  float* stimu = NULL;
  cv::Mat input_image = imread(filename, cv::IMREAD_GRAYSCALE);

  if(!input_image.data){
    std::cout << "Inputting " << filename << " is failed." << std::endl;
    return NULL;
  }
  view(&input_image);

  // Get memory for image size float array.
  stimu = (float*)malloc(sizeof(float) * (input_image.rows * input_image.cols));
  // Set PCNN size
  parameter->width = input_image.cols;
  parameter->height = input_image.rows;

  for(int y = 0; y < input_image.rows; y++){
    for(int x = 0; x < input_image.cols; x++){
      for(int c = 0; c < input_image.channels(); c++){
        stimu[(y * input_image.cols) + x] = input_image.data[(y * input_image.step) + (x * input_image.elemSize()) + c ] / 255.0;
      }
    }
  }

  //float2gray_image(stimu, input_image.cols, input_image.rows);
  return stimu;
}
