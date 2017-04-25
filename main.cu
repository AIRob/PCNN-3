#include <iostream>
#include <unistd.h>

#include "pcnn.h"

int usage(char* program_name){
  std::cout << "Usage: ";
  std::cout << program_name << " [option] image_file_path" << std::endl;
  
  std::cout << "[Options]:" << std::endl;
  std::cout << "-h ";
  std::cout << "; Print usage." << std::endl;

  std::cout << "-o output_pcnn-icon_file_path ";
  std::cout << "; Output pcnn-icon to file." << std::endl;

  std::cout << "-O output_image_directory_path ";
  std::cout << "; Output pcnn processed image files." << std::endl;

  return 0;
}

int main(int argc, char* argv[]){
  // If input image file wasn't set, print usage.
  if(argc < 2){
    usage(argv[0]);
    return 1;
  };

  pcnn_params_t parameter;
  int bad_option_flag = 0;
  int output_icon_to_file_flag = 0;
  char *icon_to_file;
  int output_images_to_file_flag = 0;
  char *images_to_file; 

  parameter.beta = 0.03;
  parameter.vF = 0.01;
  parameter.vL = 1.0;
  parameter.vT = 10.0;
  parameter.tauL = -0.1;
  parameter.tauT = -0.4;

  parameter.time_steps = 100;
  parameter.kernel_size = 10;
  
  int option;
  while((option = getopt(argc, argv, "ho:O:")) != -1){
    switch(option){
      case 'h':
        usage(argv[0]);
        return 0;

      case 'o':
        if(optarg == NULL || output_icon_to_file_flag == 1){
          bad_option_flag = 1;
        } else {
          output_icon_to_file_flag = 1;
          icon_to_file = optarg;
          std::cout << "Output PCNN-Icon to: " << optarg << std::endl;
        }
        break;

      case 'O':
        if(optarg == NULL || output_images_to_file_flag == 1){
          bad_option_flag = 1;
        } else {
          output_images_to_file_flag = 1;
          images_to_file = optarg;
          std::cout << "Output PCNN output images to: " << optarg << std::endl;
        }
        break;

      default:
        // Unknown option.
        bad_option_flag = 1;
    }
  }
  if(bad_option_flag != 0){
    usage(argv[0]);
    return 1;
  }

  if(argc - optind > 1){
    usage(argv[0]);
    return 1;
  }

  char* input_filename = argv[optind];

  std::cout << "Input image: " << input_filename << std::endl;

  float* stimu;
  stimu = image2stimuF(input_filename, &parameter);

  if(stimu == NULL){
    std::cout << "ERROR" << std::endl;
    return 1;
  };

  int ret;
  ret = pcnn(stimu, &parameter, output_icon_to_file_flag, icon_to_file, output_images_to_file_flag, images_to_file);

  if(ret != 0){
    std::cout << "ERROR" << std::endl;
    return 1;
  };

  ret = pcnn_gpu(stimu, &parameter, output_icon_to_file_flag, icon_to_file, output_images_to_file_flag, images_to_file);

  if(ret != 0){
    std::cout << "ERROR" << std::endl;
    return 1;
  };

  free(stimu);
  return 0;
}
