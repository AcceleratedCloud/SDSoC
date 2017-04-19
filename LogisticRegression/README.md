# Logistic Regression

The LR\_gradients\_kernel\_accel IP is used for the training of Logistic Regression applications. It is produced with SDSoC 2016.2 version.

The code provided in this folder is a C-based implementation of 'Batch Gradient Descent Algorithm' for Logistic Regression Training, optimised for ZED board, and is a case study of FPGA-Accelerated Machine Learning in Cloud Computing.

### Testing Logistic Regression example in Software (Intel)
1. **Compiling**
	1. Run `make`. (also extracts datasets)
1. **Execution**
	1. Edit `conf`. (optional)
		1. 'chunkSize' is the size of each data chunk (max: 5000).
		1. 'alpha' is the learning rate.
		1. 'iteration' is the number of iterations.
	1. Run `./LogisticRegression`.

### Testing Logistic Regression example in Hardware
The C++ source files are provided here without project files, but they contain HLS/SDS directives specific to Xilinx SDSoC. 

`!The code of the hardware function is not fully annotated and contains only interface directives.!`

If you want to create a SDSoC project using these sources you may find the following instructions helpful:

1.  Launch SDSoC and create a new empty project. Choose `zed` as target platform.
1.  Add the C++ sources in `src/` and set `LR_gradients_kernel_accel` as hardware function. Set clock frequency at `142.86 MHz`.
1.  All design parameters are set in the file `src/accelerator.h`.
1.  Select `Generate Bitstream` and `Generate SD Card Image`.
1.  Run `SDRelease`.
	
#### Performance (`LR_gradients_kernel`, `5000 chunkSize`)
Speedup (vs Intel)	|	5.89
:----------|----------:
SW-only `ARM Cortex-A9 @ 666.67MHz` (Measured time)	|	1003 msec
SW-only `Intel Core i5-5200U @ 2.20GHz` (Measured time)	|	277 msec
HW accelerated (Estimated time)	|	48 msec
HW accelerated (Measured time)	|	47 msec

#### Resource utilization estimates for hardware accelerator
Resource	|	Used	|	Total	|	% Utilization
:----------:|----------:|----------:|:----------
DSP	|	160	|	220	|	72.73
BRAM	|	42	|	140	|	30
LUT	|	44491	|	53200	|	83.63
FF	|	48292	|	106400	|	45.39
	
### Contacts
For any question or discussion, please contact the authors:

* Christoforos Kachris: kachris@microlab.ntua.gr
* Elias Koromilas: el11059@central.ntua.gr
