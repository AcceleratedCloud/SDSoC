from pynq import MMIO, Overlay, PL
from pynq.mllib_accel import DMA

DMA_TO_DEV = 0    # DMA sends data to PL.
DMA_FROM_DEV = 1  # DMA receives data from PL.

class LR_Accel:
    """
    Python class for the LR Accelerator.
    """
    
    def __init__(self, chunkSize, numClasses, numFeatures):
        self.numClasses = numClasses
        self.numFeatures = numFeatures
           
        # -------------------------
        #   Download Overlay.
        # -------------------------    

        ol = Overlay("LogisticRegression.bit")
        ol.download()  
        
        # -------------------------
        #   Physical address of the Accelerator Adapter IP.
        # -------------------------

        ADDR_Accelerator_Adapter_BASE = int(PL.ip_dict["SEG_LR_gradients_kernel_accel_0_if_Reg"][0], 16)
        ADDR_Accelerator_Adapter_RANGE = int(PL.ip_dict["SEG_LR_gradients_kernel_accel_0_if_Reg"][1], 16)

        # -------------------------
        #    Initialize new MMIO object. 
        # -------------------------

        self.bus = MMIO(ADDR_Accelerator_Adapter_BASE, ADDR_Accelerator_Adapter_RANGE)

        # -------------------------
        #   Physical addresses of the DMA IPs.
        # -------------------------

        ADDR_DMA0_BASE = int(PL.ip_dict["SEG_dm_0_Reg"][0], 16)
        ADDR_DMA1_BASE = int(PL.ip_dict["SEG_dm_1_Reg"][0], 16)
        ADDR_DMA2_BASE = int(PL.ip_dict["SEG_dm_2_Reg"][0], 16)
        ADDR_DMA3_BASE = int(PL.ip_dict["SEG_dm_3_Reg"][0], 16)

        # -------------------------
        #    Initialize new DMA objects. 
        # -------------------------

        self.dma0 = DMA(ADDR_DMA0_BASE, direction = DMA_TO_DEV)    # data1 DMA.
        self.dma1 = DMA(ADDR_DMA1_BASE, direction = DMA_TO_DEV)    # data2 DMA.
        self.dma2 = DMA(ADDR_DMA2_BASE, direction = DMA_TO_DEV)    # weights DMA.
        self.dma3 = DMA(ADDR_DMA3_BASE, direction = DMA_FROM_DEV)  # gradients DMA.
        
        # -------------------------
        #    Allocate physically contiguous memory buffers.
        # -------------------------

        self.dma0.create_buf(int(chunkSize / 2) * (self.numClasses + (1 + self.numFeatures)) * 4, 1)
        self.dma1.create_buf(int(chunkSize / 2) * (self.numClasses + (1 + self.numFeatures)) * 4, 1)
        self.dma2.create_buf((self.numClasses * (1 + self.numFeatures)) * 4, 1)
        self.dma3.create_buf((self.numClasses * (1 + self.numFeatures)) * 4, 1)

        # -------------------------
        #    Get CFFI pointers to objects' internal buffers.
        # -------------------------

        self.data1_buf = self.dma0.get_buf(32, data_type = "float")
        self.data2_buf = self.dma1.get_buf(32, data_type = "float")
        self.weights_buf = self.dma2.get_buf(32, data_type = "float")
        self.gradients_buf = self.dma3.get_buf(32, data_type = "float")

    def gradients_kernel(self, data, weights):
        chunkSize = int(len(data) / (self.numClasses + (1 + self.numFeatures)))
        
        for i in range (0, int(len(data) / 2)):
            self.data1_buf[i] = float(data[i])
            self.data2_buf[i] = float(data[int(len(data) / 2) + i])
        for kj in range (0, self.numClasses * (1 + self.numFeatures)):
            self.weights_buf[kj] = float(weights[kj])

        # -------------------------
        #   Write data to MMIO.
        # -------------------------

        CMD = 0x0028            # Command.
        ISCALAR0_DATA = 0x0080  # Input Scalar-0 Write Data FIFO.

        self.bus.write(ISCALAR0_DATA, int(chunkSize))
        self.bus.write(CMD, 0x00010001)
        self.bus.write(CMD, 0x00020000)
        self.bus.write(CMD, 0x00000107)

        # -------------------------
        #   Transfer data using DMAs (Non-blocking).
        #   Block while DMAs are busy.
        # -------------------------

        self.dma0.transfer(int(len(data) / 2) * 4, direction = DMA_TO_DEV)
        self.dma1.transfer(int(len(data) / 2) * 4, direction = DMA_TO_DEV)
        self.dma2.transfer((self.numClasses * (1 + self.numFeatures)) * 4, direction = DMA_TO_DEV)

        self.dma0.wait()
        self.dma1.wait()
        self.dma2.wait()

        self.dma3.transfer((self.numClasses * (1 + self.numFeatures)) * 4, direction = DMA_FROM_DEV)

        self.dma3.wait()

        gradients = []
        for kj in range (0, self.numClasses * (1 + self.numFeatures)):
            gradients.append(float(self.gradients_buf[kj]))

        return gradients
    
    def __del__(self):

        # -------------------------
        #   Destructors for DMA objects.
        # -------------------------

        self.dma0.__del__()
        self.dma1.__del__()
        self.dma2.__del__()
        self.dma3.__del__()
