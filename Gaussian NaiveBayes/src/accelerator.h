#ifndef ACCELERATOR_H_
#define ACCELERATOR_H_

#define N_feat 784
#define N_class 10


#pragma SDS data copy(data[0:N_feat*DataPack])

#pragma SDS data access_pattern  ( data: SEQUENTIAL, means :SEQUENTIAL, variances :SEQUENTIAL, priors:SEQUENTIAL )

#pragma SDS data mem_attribute ( data : CACHEABLE , means:CACHEABLE , variances :CACHEABLE ,  priors : CACHEABLE ,  class_sum : CACHEABLE  )

#pragma SDS data data_mover ( data :  AXIDMA_SIMPLE, priors : AXIDMA_SIMPLE , variances :AXIDMA_SIMPLE, means:AXIDMA_SIMPLE )

#pragma SDS data sys_port ( data : ACP, means:ACP , variances :ACP , priors:ACP  )

void NBtraining_accel(	int DataPack,
                        int *class_sum,
                        float *data,
                        float *priors,
                        float *means,
                        float *variances);

#pragma SDS data copy(data[0:N_feat])

#pragma SDS data access_pattern  ( data :SEQUENTIAL ,means :SEQUENTIAL, variances :SEQUENTIAL, priors:SEQUENTIAL)

#pragma SDS data mem_attribute ( data : CACHEABLE , means:CACHEABLE , variances :CACHEABLE ,  priors : CACHEABLE )

#pragma SDS data data_mover ( data :  AXIDMA_SIMPLE, priors : AXIDMA_SIMPLE , variances :AXIDMA_SIMPLE, means:AXIDMA_SIMPLE  )

#pragma SDS data sys_port ( data : ACP, means:ACP , variances :ACP , priors:ACP )

int NBprediction_accel( float *data,
                        float *means,
                        float *variances,
                        float *priors );

#endif /* ACCELERATOR_H_ */
