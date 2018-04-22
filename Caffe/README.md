<p align="center">
  <a href="http://caffe.berkeleyvision.org/">
    <img src="https://dashbouquet.com/assets/img/blog/caffe-banner.png" alt="" width=400 height=195>
  </a>

  <h2 align="center">Hardware acceleration of Deep Neural Networks</h2>

  <p align="center">
    Image recognition with <a href="http://caffe.berkeleyvision.org/"><strong>Caffe</strong></a> framework using FPGAs.
    <br>
    <br>
    <a href="https://github.com/AcceleratedCloud/SDSoC/issues/new?labels=bug">Report a bug</a>
    ·
    <a href="https://github.com/AcceleratedCloud/SDSoC/issues/new?labels=question">Ask a question</a>
    ·
    <a href="https://github.com/AcceleratedCloud/SDSoC/issues/new?labels=enhancement">Request feature</a>
    </p>
</p>

<br>

## Overview

Caffe is an intuitive and powerful framework that implements Deep Neural Networks (DNNs) geared towards image recognition, object detection etc.

This project demonstrates the deployment of DNNs with Caffe in the embedded system of Zynq 7000 SoC and the acceleration through the FPGA. The code provided is designed and optimized with Xilinx SDSoC and runs on the Zedboard but can be easily modified for other Zynq based SoCs.
This work includes:
- The evaluation of the FPGA hardware accelerator used in the project: `GEMM (General Matrix Multiply)`
- The compiled Caffe framework and all its libraries in order to run in the embbeded SoC: `ARM CPU`
- The final CPU-FPGA system that supports Caffe and utilizes the hardware accelerator: `SD boot card provided`
- The SW and HW system evaluation of DNNs using Caffe : `models supported -> SqueezeNet, GoogleNet, etc.`


## Quick start
 
 to be filled...
