project(pcnn)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch sm_20 -ccbin g++-5)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
cuda_add_executable(pcnn pcnn.cu image.cu main.cu pcnn.h)
target_link_libraries(pcnn ${OpenCV_LIBS})
