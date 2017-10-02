# KMeans

The KM\_centroids\_kernel\_accel IP is used for the training of KMeans applications. It is produced with SDSoC 2016.2 version.

The code provided in this folder is a C-based implementation of 'Partial Centroids Calculation (counts, sums)' for KMeans Training, optimized for ZC702 board, and is a case study of FPGA-Accelerated Machine Learning in Cloud Computing.

### Testing KMeans kernel in Software (Intel)
1. **Compiling**
	1. Run `make`. (-O3 optimizations)
1. **Execution**
	1. Edit `conf`. (optional)
		1. 'chunkSize' is the size of the data chunk (max: 5000).
	1. Run `./KMeans`.

### Testing KMeans kernel in Hardware
The C++ source files are provided here without project files, but they contain HLS/SDS directives specific to Xilinx SDSoC. 

`!The code of the hardware function is not fully annotated and contains only interface directives.!`

If you want to create a SDSoC project using these sources you may find the following instructions helpful:

1.  Launch SDSoC and create a new empty project. Choose `zc702` as target platform.
1.  Add the C++ sources in `src/` and set `KM_centroids_kernel_accel` as hardware function. Set clock frequency at `142.86 MHz`.
1.  All design parameters are set in the file `src/accelerator.h`.
1.  Select `Generate Bitstream` and `Generate SD Card Image`.
1.  Run `SDRelease`.
	
#### Performance (`KM_centroids_kernel`, `5000 chunkSize`)
Speedup (vs Intel)	|	4.05
:----------|----------:
SW-only `ARM Cortex-A9 @ 666.67MHz` (Measured time)	|	1757 msec
SW-only `Intel Core i5-5200U @ 2.20GHz` (Measured time)	|	231 msec
HW accelerated (Estimated time)	|	58 msec
HW accelerated (Measured time)	|	57 msec

#### Resource utilization estimates for hardware accelerator
Resource	|	Used	|	Total	|	% Utilization
:----------:|----------:|----------:|:----------
DSP	|	142	|	220	|	64.55
BRAM	|	86	|	140	|	61.43
LUT	|	44024	|	53200	|	82.75
FF	|	44966	|	106400	|	42.26
	
### Contacts
For any question or discussion, please contact the authors:

* Christoforos Kachris: kachris@microlab.ntua.gr
* Elias Koromilas: elias.koromilas@gmail.com
