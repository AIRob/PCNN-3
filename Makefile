TARGET = pcnn
OBJS = pcnn.o image.o main.o

NVCC = nvcc
CFLAGS = -arch=sm_20 -ccbin g++ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -I/usr/include/opencv

.SUFFIXES: .cu .o


$(TARGET): $(OBJS)
	$(NVCC) -o $(TARGET) $(CFLAGS) $^

.cu.o:
	$(NVCC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS)

main.o: pcnn.h

image.o: pcnn.h

pcnn.o: pcnn.h
