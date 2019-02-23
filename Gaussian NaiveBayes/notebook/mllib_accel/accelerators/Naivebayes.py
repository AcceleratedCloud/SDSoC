import numpy as np
import os
import platform
import re
import cffi
import random

from itertools import tee
from math import ceil
from pynq import Overlay

DataPackSizeMax = 2000
numClassesMax = 10
numFeaturesMax = 784

BS_SEARCH_PATH = os.path.dirname(os.path.realpath(__file__)) + "/overlays/"

ffi = cffi.FFI()

ffi.cdef("""
void overlay_download(char *bit_file);
void *sds_alloc(unsigned int size);
void sds_free(void *memptr);
void NBtraining_kernel(int DataPack,int *n_per_class, int data,float *priors,float *means,float *variances);
int NBprediction_kernel(int data,int means,int variances, int priors );
""")

LIB_SEARCH_PATH = os.path.dirname(os.path.realpath(__file__)) + "/drivers/"
if platform.machine() == "x86_64":
    # load 64bit ELF
    hw_tr = ffi.dlopen(LIB_SEARCH_PATH + "NBtraining_kernel64.so")
    hw_pr = ffi.dlopen(LIB_SEARCH_PATH + "NBprediction_kernel64.so")
elif platform.machine() == "i386":
    # load 32bit ELF
    hw_tr = ffi.dlopen(LIB_SEARCH_PATH + "NBtraining_kernel32.so")
    hw_pr = ffi.dlopen(LIB_SEARCH_PATH + "NBprediction_kernel32.so")
elif platform.machine() == "armv7l":
    # load 32bit ELF compiled for ARM
    hw_tr = ffi.dlopen(LIB_SEARCH_PATH + "NBtraining_kernel.so")
    hw_pr = ffi.dlopen(LIB_SEARCH_PATH + "NBprediction_kernel.so")
else:
    print("Machine type not supported. Exiting!")
    exit(1)


def cma_train(LabeledPoints):
    # -------------------------
    #   Download Overlay.
    # -------------------------    
    
    hw_tr.overlay_download((BS_SEARCH_PATH + "NBtraining.bit").encode('ascii'))

    elements = []
    flatLabelpoints = []

    numLabeledPoints = len(LabeledPoints)

    numDataPacks = int(ceil(numLabeledPoints / DataPackSizeMax))
    DataPackSize = int(ceil(numLabeledPoints / numDataPacks))
    if bool(DataPackSize & 1):
        DataPackSize += 1
    paddingSize = (numDataPacks * DataPackSize) - numLabeledPoints

    overall_class = []

    chunklist = [LabeledPoints[i:i + DataPackSize] for i in range(0, numLabeledPoints, DataPackSize)]
    for i in range(numDataPacks):
        chunklist[i] = sorted(chunklist[i] , key = lambda x: x.label)
        flatLabelpoints.append(chunklist[i])
        trainRDDcount = list(map(lambda x: x.label, chunklist[i]))

        num_class = []
        for i in range(numClassesMax):
            num_class.append(trainRDDcount.count(float(i)))    
        overall_class.append(num_class)

    c = 1
    # -------------------------
    #   Allocate physically contiguous memory buffers.
    # -------------------------   
    data = ffi.cast("float *", hw_tr.sds_alloc(DataPackSize * numFeaturesMax * 4))
    LabeledPoints = [i for sublist in flatLabelpoints for i in sublist]

    buffers = []
    buffers.append(int(re.split("0x|>", str(data))[1], 16))
    buffers.append(DataPackSize * numFeaturesMax)
    buffers.append((DataPackSize - paddingSize) if c == numDataPacks else DataPackSize)
    buffers.append(overall_class[c-1])
    elements.append(buffers)

    i = 0
    for LabeledPoint in LabeledPoints:
        if i < int(DataPackSize):
            f = ffi.from_buffer(LabeledPoint.features.astype(np.float32))
            features = ffi.cast("float *", f) 
            offset_point = i * numFeaturesMax
            data[offset_point:offset_point + len(LabeledPoint.features)] = features[0:len(LabeledPoint.features)] 
        i += 1
        if i == int(DataPackSize):
            c += 1
            if c <= numDataPacks:  
                # -------------------------
                #   Allocate physically contiguous memory buffers.
                # -------------------------
                data = ffi.cast("float *", hw_tr.sds_alloc(DataPackSize * numFeaturesMax * 4))
                buffers = []
                buffers.append(int(re.split("0x|>", str(data))[1], 16))
                buffers.append(DataPackSize * numFeaturesMax)
                buffers.append((DataPackSize - paddingSize) if c == numDataPacks else DataPackSize)
                buffers.append(overall_class[c-1])
                elements.append(buffers)
                i = 0
    return elements

def trainingNB_kernel_accel(buffers):
    
    # -------------------------
    #   Accelerator callsite.
    # ------------------------

    DatapackSize = int(buffers[1] / (numFeaturesMax))

    numClasses = 10
    numFeatures = 784

    offset_buf = ffi.cast("int *", hw_tr.sds_alloc(numClassesMax * 4))
    offset_buf[0:numClassesMax] = buffers[3][0:numClassesMax]

    m_buf = ffi.new("float[]", numClassesMax * numFeaturesMax)
    v_buf = ffi.new("float[]", numClassesMax * numFeaturesMax)
    p_buf = ffi.new("float[]", numClassesMax)
    
    hw_tr.NBtraining_kernel(DatapackSize, offset_buf, buffers[0], p_buf, m_buf, v_buf)
    
    v = np.frombuffer(ffi.buffer(v_buf, 4 * numClassesMax * numFeaturesMax), dtype = np.float32)
    m = np.frombuffer(ffi.buffer(m_buf, 4 * numClassesMax * numFeaturesMax), dtype = np.float32)
    p = np.frombuffer(ffi.buffer(p_buf, 4 *  numClassesMax), dtype = np.float32)


    variances = np.copy(np.reshape(v, (numClasses,numFeatures))[ : , :numFeatures])
    means= np.copy(np.reshape(m, (numClasses,numFeatures))[ : , :numFeatures])
    priors = np.copy(np.reshape(p, (numClasses))[:])

    trainPack = [(means), (variances), (priors)]

    return trainPack

def cmf_train(buffers):
    
    # -------------------------
    #   Free previously allocated buffers.
    # -------------------------

    hw_tr.sds_free(ffi.cast("void *", buffers[0]))

    return 0


def cma_predict(trainPack):

    # -------------------------
    #   Download Overlay.
    # -------------------------    
    hw_pr.overlay_download((BS_SEARCH_PATH + "NBprediction.bit").encode('ascii'))


    means = ffi.cast("float *", hw_pr.sds_alloc(numClassesMax * numFeaturesMax * 4))
    variances = ffi.cast("float *", hw_pr.sds_alloc(numClassesMax * numFeaturesMax * 4))
    priors = ffi.cast("float *", hw_pr.sds_alloc(numClassesMax * 4))

    address = []
    address.append(int(re.split("0x|>", str(means))[1], 16))
    address.append(int(re.split("0x|>", str(variances))[1], 16))
    address.append(int(re.split("0x|>", str(priors))[1], 16))
    
    m = np.reshape(list(trainPack[0]), numClassesMax * numFeaturesMax)
    v = np.reshape(trainPack[1], numClassesMax * numFeaturesMax) 
    p = np.reshape(trainPack[2], numClassesMax )

    means[0:numClassesMax * numFeaturesMax] = m[0:numClassesMax * numFeaturesMax]
    variances[0:numClassesMax * numFeaturesMax] = v[0:numClassesMax * numFeaturesMax]
    priors[0:numClassesMax] = p[0:numClassesMax]

    return address

def predictionNB_kernel_accel(buffers, line):
    
    # -------------------------
    #   Accelerator callsite.
    # ------------------------

    numFeatures = 784

    data = ffi.cast("float *", hw_tr.sds_alloc(numFeaturesMax * 4))
    addr = int(re.split("0x|>", str(data))[1], 16)
    data[0:numFeatures] = line[0:numFeatures]
    
    prediction = hw_pr.NBprediction_kernel(addr, buffers[0], buffers[1], buffers[2])
    
    hw_pr.sds_free(ffi.cast("void *", addr))

    return prediction

def cmf_predict(buffers):
    
    # -------------------------
    #   Free previously allocated buffers.
    # -------------------------

    hw_pr.sds_free(ffi.cast("void *", buffers[0]))
    hw_pr.sds_free(ffi.cast("void *", buffers[1]))
    hw_pr.sds_free(ffi.cast("void *", buffers[2]))

    return 0