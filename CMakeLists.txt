project(pcnn)
if(CXX)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch sm_20 -ccbin ${CXX})
else()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch sm_20 -ccbin g++)
endif()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
cuda_add_executable(pcnn pcnn.cu image.cu main.cu pcnn.h)
target_link_libraries(pcnn ${OpenCV_LIBS})

INSTALL(TARGETS pcnn DESTINATION bin)
