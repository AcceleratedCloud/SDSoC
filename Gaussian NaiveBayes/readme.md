# Gaussian Naive Bayes

It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.

Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

The Naive Bayes IP is used forNaive Bayes applications. It is produced with SDSoC 2016.3 version.

The code provided in this folder is a C-based implementation for NaiveBayes Training an d Prediction, optimised for ZED board, and is a case study of FPGA-Accelerated Machine Learning.

# Publication

If you use any of the tools in your work, we would love to hear about it and would very much appreciate a citation:

- Hardware Acceleration on Gaussian Naive Bayes Machine Learning Alogorithm , Georgios Tzanos, Christoforos Kachris, and Dimitrios Soudris. International Conference on Modern Circuits and Systems Technologies (MOCAST), 2019.

```
@misc{
  author =       "Georgios Tzanos, Christoforos Kachris, and Dimitrios Soudris",
  title =        "Hardware Acceleration on Gaussian Naive Bayes Machine Learning Alogorithm",
  conference =   "(MOCAST)",
  year =         "2019",
  month =        "May",
}
```



### Testing Gaussian Naive Bayes example in Hardware
The C source files are provided here without project files, but they contain HLS/SDS directives specific to Xilinx SDSoC. 

`!The code of the hardware function is not fully annotated and contains only interface directives.!`

If you want to create a SDSoC project using these sources you may find the following instructions helpful:

1.  Launch SDSoC and create a new empty project. Choose `zed` as target platform.
1.  Add the C sources in `src/` and set `NBtraining_accel` or `NBprediction_accel` as hardware function. Set clock frequency at `100.00 MHz`.
1.  All design parameters are set in the file `src/accelerator.h`.
1.  Select `Generate Bitstream` and `Generate SD Card Image`.
1.  Run `SDRelease`.
	
#### Performance (`NBtraining_accel`, `2000 DataPack`)
Speedup (vs ARM-A9)	|	16.8
:----------|----------:
SW-only `ARM Cortex-A9 @ 666.67MHz` (Measured time)	|	273 msec
HW accelerated (Measured time)		|	16 msec


#### Resource utilization estimates for hardware accelerator
Resource	|	Used	|	Total	|	% Utilization
:----------:|----------:|----------:|:----------
DSP	|	197	|	220	|	89.55
BRAM	|	56	|	140	|	40
LUT	|	37929	|	53200	|	71.3
FF	|	29271	|	106400	|	27.51

#### Performance (`NBprediction_accel`, `2000 Examples`)
Speedup (vs ARM-A9)	|	14
:----------|----------:
SW-only `ARM Cortex-A9 @ 666.67MHz` (Measured time)	|	7.83 sec
HW accelerated (Measured time)		|	0.5 sec


#### Resource utilization estimates for hardware accelerator
Resource	|	Used	|	Total	|	% Utilization
:----------:|----------:|----------:|:----------
DSP	|	96	|	220	|	43.64
BRAM	|	56	|	140	|	40
LUT	|	28524	|	53200	|	53.62
FF	|	25251	|	106400	|	23.73
	
### Contacts
For any question or discussion, please contact the authors:

* Christoforos Kachris: kachris@microlab.ntua.gr
* Giorgos Tzanos: grg.tzan@gmail.com
