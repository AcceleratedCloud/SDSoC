#define chunkSizeMax 5000
#define numClasses 10
#define numFeatures 784

#pragma SDS data access_pattern(data1:SEQUENTIAL, data2:SEQUENTIAL, weights:SEQUENTIAL, gradients:SEQUENTIAL)
#pragma SDS data copy(data1[0:chunkSize * (numClasses + (1 + numFeatures)) / 2], data2[0:chunkSize * (numClasses + (1 + numFeatures)) / 2])
#pragma SDS data data_mover(data1:AXIDMA_SIMPLE, data2:AXIDMA_SIMPLE, weights:AXIDMA_SIMPLE, gradients:AXIDMA_SIMPLE)
#pragma SDS data mem_attribute (data1:PHYSICAL_CONTIGUOUS, data2:PHYSICAL_CONTIGUOUS, weights:PHYSICAL_CONTIGUOUS, gradients:PHYSICAL_CONTIGUOUS)
#pragma SDS data sys_port (data1:ACP, data2:ACP, weights:ACP, gradients:ACP)
void LR_gradients_kernel_accel(float data1[(int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2)], float data2[(int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2)], float weights[numClasses * (1 + numFeatures)], int chunkSize, float gradients[numClasses * (1 + numFeatures)]);

void LR_gradients_kernel(float data1[(int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2)], float data2[(int)(chunkSizeMax * (numClasses + (1 + numFeatures)) / 2)], float weights[numClasses * (1 + numFeatures)], int chunkSize, float gradients[numClasses * (1 + numFeatures)]);
