# Alternating Least Squares (ALS) Recomendation Engine

The Als_kernel is used for the training of Recomendation engines , It is produced with SDSoC 2016.2 version.
The code provided in this folder is a C-based implementation of 'Alternating Least Squares Training Algorithm' for Recomendation, optimised for ZED board, and is a case study of FPGA-Accelerated Machine Learning. The algorith in its current form supports only explicit feedback .

## Testing Als source in Software
1. compiling
    + run `make` (also extracts datasets)  
2. running
    + (optional) Edit run.sh
    + run using `./run.sh`

## Testing Als in Hardware
The C source files are provided here without project files, but they contain HLS/SDS directives specific to Xilinx SDSoC. 

If you want to create a SDSoC project using these sources you may find the following instructions helpful:

1.  Launch SDSoC and create a new empty project. Choose `zed` as target platform.
1.  Add the C sources in `src/` and set `topLevelHw` as hardware function. Set clock frequency at `142.86 MHz`.
1.  All design parameters are set in the file `src/headers.h`.
1.  Select `Generate Bitstream` and `Generate SD Card Image`.
1.  Run `SDRelease`.

### Performance using 80 Features
|Dataset|Software only exececution (ZedBoard) | Software-Hardware Execution (ZedBoard)|Speed-Up|
|-------|:---------------------------------:|:----------------:|:------:|
|movielens100k|470.82 s | 128.09s | x3.68
|movielens1m  |81533.82s| 7304.24s| x11.16

### Resource Utilization
|Resource|Used|Total|Total %|
|:-----:|:------:|:-----:|:----:|
|DSP    |204     |220    |92|
|BRAM   |58      |280    |20|
|LUT    |35866   |53200  |67|
|FF     |29030   |106400 |27|

### Contact

For any question or discussion, please contact the authors:

  + Christoforos Kachris: kachris@microlab.ntua.gr
  + Konstantinos Katsantonis: el11189@central.ntua.gr


