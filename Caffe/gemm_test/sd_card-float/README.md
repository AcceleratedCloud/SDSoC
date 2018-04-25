#### SD card boot image `Platform: zed`

Steps:
1. Copy all the contents of this directory to an SD card
2. Set boot mode to SD
3. Insert SD card and turn board on
4. Run:
```
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/lib/
$ ./mnt/zed_gemm_float.elf
```
