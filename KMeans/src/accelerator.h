#define chunkSizeMax 5000
#define numClusters 14
#define numFeatures 784

#pragma SDS data access_pattern (data1:SEQUENTIAL, data2:SEQUENTIAL, centroids:SEQUENTIAL, counts_sums:SEQUENTIAL)
#pragma SDS data copy (data1[0:chunkSize * numFeatures / 2], data2[0:chunkSize * numFeatures / 2])
#pragma SDS data data_mover (data1:AXIDMA_SIMPLE, data2:AXIDMA_SIMPLE, centroids:AXIDMA_SIMPLE, counts_sums:AXIDMA_SIMPLE)
#pragma SDS data mem_attribute (data1:PHYSICAL_CONTIGUOUS, data2:PHYSICAL_CONTIGUOUS, centroids:PHYSICAL_CONTIGUOUS, counts_sums:PHYSICAL_CONTIGUOUS)
#pragma SDS data sys_port (data1:ACP, data2:ACP, centroids:ACP, counts_sums:ACP)
void KM_centroids_kernel_accel(float data1[(int)(chunkSizeMax * numFeatures / 2)], float data2[(int)(chunkSizeMax * numFeatures / 2)], float centroids[numClusters * numFeatures], int chunkSize, float counts_sums[numClusters * (1 + numFeatures)]);

void KM_centroids_kernel(float data1[(int)(chunkSizeMax * numFeatures / 2)], float data2[(int)(chunkSizeMax * numFeatures / 2)], float centroids[numClusters * numFeatures], int chunkSize, float counts_sums[numClusters * (1 + numFeatures)]);
