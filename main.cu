#include <iostream>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>

#include "pcnn.h"

int usage(char* program_name){
  std::cout << "Usage: ";
  std::cout << program_name << " [Options] image_file_path" << std::endl;
  
  std::cout << "-- Options --" << std::endl;
  std::cout << "Print usage:" << std::endl;
  std::cout << "[-h|--help]\n" << std::endl;

  std::cout << "Set output pcnn-icon to file:" << std::endl;
  std::cout << "[-o|--output_file] output_pcnn-icon_file_path\n" << std::endl;

  std::cout << "Set output directory for pcnn processed image files:" << std::endl;
  std::cout << "[-O|--output_images] output_image_directory_path\n" << std::endl;

  std::cout << "Set the number of time steps for processing the PCNN:" << std::endl;
  std::cout << "[-s|--step] time_steps\n" << std::endl;

  std::cout << "Set the size of the weights kernel:" << std::endl;
  std::cout << "[-k|--kernel] kernel_size\n" << std::endl;

  std::cout << "Parameters setting options: " << std::endl;
  std::cout << "[--beta] beta" << std::endl;
  std::cout << "[--vF] vF" << std::endl;
  std::cout << "[--vL] vL" << std::endl;
  std::cout << "[--vT] vT" << std::endl;
  std::cout << "[--tauL] tauL" << std::endl;
  std::cout << "[--tauT] tauT" << std::endl;

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

  while(1){
    int option_index = 0;
    static struct option long_options[] = {
      {"help",           no_argument,       0, 'h' },
      {"output_file",    no_argument,       0, 'o' },
      {"output_images",  no_argument,       0, 'O' },
      {"beta",           required_argument, 0,  0  },
      {"vF",             required_argument, 0,  0  },
      {"vL",             required_argument, 0,  0  },
      {"vT",             required_argument, 0,  0  },
      {"tauL",           required_argument, 0,  0  },
      {"tauT",           required_argument, 0,  0  },
      {"step",           required_argument, 0, 's' },
      {"kernel",         required_argument, 0, 'k' }
    };
    int option = getopt_long(argc, argv, "ho:O:s:k:",
                             long_options, &option_index);
    if(option == -1){
      break;
    }
    switch(option){
      case 0:
        if(!strcmp(long_options[option_index].name, "beta")){
          parameter.beta = atof(optarg);
          break;
        }
        if(!strcmp(long_options[option_index].name, "vF")){
          parameter.vF = atof(optarg);
          break;
        }
        if(!strcmp(long_options[option_index].name, "vL")){
          parameter.vL = atof(optarg);
          break;
        }
        if(!strcmp(long_options[option_index].name, "vT")){
          parameter.vT = atof(optarg);
          break;
        }
        if(!strcmp(long_options[option_index].name, "tauL")){
          parameter.tauL = atof(optarg);
          break;
        }
        if(!strcmp(long_options[option_index].name, "tauT")){
          parameter.tauT = atof(optarg);
          break;
        }
        break;

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

      case 's':
        parameter.time_steps = atoi(optarg);
        break;

      case 'k':
        parameter.kernel_size = atoi(optarg);
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

  std::cout << "PCNN parameters: " << std::endl;
  std::cout << "beta = " << parameter.beta << std::endl;
  std::cout << "vF = " << parameter.vF << std::endl;
  std::cout << "vL = " << parameter.vL << std::endl;
  std::cout << "vT = " << parameter.vT << std::endl;
  std::cout << "tauL = " << parameter.tauL << std::endl;
  std::cout << "tauT = " << parameter.tauT << std::endl;
  std::cout << "time_steps = " << parameter.time_steps << std::endl;
  std::cout << "kernel_size = " << parameter.kernel_size << std::endl;

  int ret;

  std::cout << "PCNN on CPU start ..." << std::endl;
  ret = pcnn(stimu, &parameter, output_icon_to_file_flag, icon_to_file, output_images_to_file_flag, images_to_file);

  if(ret != 0){
    std::cout << "ERROR" << std::endl;
    return 1;
  };
  std::cout << "PCNN on CPU end.\n\n" << std::endl;

  std::cout << "PCNN on GPU start ..." << std::endl;
  ret = pcnn_gpu(stimu, &parameter, output_icon_to_file_flag, icon_to_file, output_images_to_file_flag, images_to_file);

  if(ret != 0){
    std::cout << "ERROR" << std::endl;
    return 1;
  };
  std::cout << "PCNN on GPU end.\n\n" << std::endl;

  free(stimu);
  return 0;
}
