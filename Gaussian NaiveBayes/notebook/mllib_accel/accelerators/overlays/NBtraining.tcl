
################################################################
# This is a generated script based on design: zed
#
# Though there are limitations about the generated script,
# the main purpose of this utility is to make learning
# IP Integrator Tcl commands easier.
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   set script_folder [file dirname $script_path]
   return $script_folder
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

################################################################
# Check if script is running in correct Vivado version.
################################################################
set scripts_vivado_version 2017.4
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   catch {common::send_msg_id "BD_TCL-109" "ERROR" "This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Please run the script in Vivado <$scripts_vivado_version> then open the design in Vivado <$current_vivado_version>. Upgrade the design by running \"Tools => Report => Report IP Status...\", then run write_bd_tcl to create an updated script."}

   return 1
}

################################################################
# START
################################################################

# To test this script, run the following commands from Vivado Tcl console:
# source zed_script.tcl

# If there is no project opened, this script will create a
# project, but make sure you do not have an existing project
# <./myproj/project_1.xpr> in the current working folder.

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   create_project project_1 myproj -part xc7z020clg400-1
}


# CHANGE DESIGN NAME HERE
variable design_name
set design_name zed

# If you do not already have an existing IP Integrator design open,
# you can create a design using the following command:
#    create_bd_design $design_name

# Creating design if needed
set errMsg ""
set nRet 0

set cur_design [current_bd_design -quiet]
set list_cells [get_bd_cells -quiet]

if { ${design_name} eq "" } {
   # USE CASES:
   #    1) Design_name not set

   set errMsg "Please set the variable <design_name> to a non-empty value."
   set nRet 1

} elseif { ${cur_design} ne "" && ${list_cells} eq "" } {
   # USE CASES:
   #    2): Current design opened AND is empty AND names same.
   #    3): Current design opened AND is empty AND names diff; design_name NOT in project.
   #    4): Current design opened AND is empty AND names diff; design_name exists in project.

   if { $cur_design ne $design_name } {
      common::send_msg_id "BD_TCL-001" "INFO" "Changing value of <design_name> from <$design_name> to <$cur_design> since current design is empty."
      set design_name [get_property NAME $cur_design]
   }
   common::send_msg_id "BD_TCL-002" "INFO" "Constructing design in IPI design <$cur_design>..."

} elseif { ${cur_design} ne "" && $list_cells ne "" && $cur_design eq $design_name } {
   # USE CASES:
   #    5) Current design opened AND has components AND same names.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 1
} elseif { [get_files -quiet ${design_name}.bd] ne "" } {
   # USE CASES: 
   #    6) Current opened design, has components, but diff names, design_name exists in project.
   #    7) No opened design, design_name exists in project.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 2

} else {
   # USE CASES:
   #    8) No opened design, design_name not in project.
   #    9) Current opened design, has components, but diff names, design_name not in project.

   common::send_msg_id "BD_TCL-003" "INFO" "Currently there is no design <$design_name> in project, so creating one..."

   create_bd_design -bdsource SDSoC $design_name

   common::send_msg_id "BD_TCL-004" "INFO" "Making design <$design_name> as current_bd_design."
   current_bd_design $design_name

}

common::send_msg_id "BD_TCL-005" "INFO" "Currently the variable <design_name> is equal to \"$design_name\"."

if { $nRet != 0 } {
   catch {common::send_msg_id "BD_TCL-114" "ERROR" $errMsg}
   return $nRet
}

set bCheckIPsPassed 1
##################################################################
# CHECK IPs
##################################################################
set bCheckIPs 1
if { $bCheckIPs == 1 } {
   set list_check_ips "\ 
xilinx.com:hls:NBtraining_accel:1.0\
xilinx.com:ip:adapter_v3_0:1.0\
xilinx.com:ip:xlconstant:1.1\
xilinx.com:ip:clk_wiz:5.4\
xilinx.com:ip:axi_dma:7.1\
xilinx.com:ip:proc_sys_reset:5.0\
xilinx.com:ip:processing_system7:5.5\
xilinx.com:ip:xlconcat:2.1\
"

   set list_ips_missing ""
   common::send_msg_id "BD_TCL-006" "INFO" "Checking if the following IPs exist in the project's IP catalog: $list_check_ips ."

   foreach ip_vlnv $list_check_ips {
      set ip_obj [get_ipdefs -all $ip_vlnv]
      if { $ip_obj eq "" } {
         lappend list_ips_missing $ip_vlnv
      }
   }

   if { $list_ips_missing ne "" } {
      catch {common::send_msg_id "BD_TCL-115" "ERROR" "The following IPs are not found in the IP Catalog:\n  $list_ips_missing\n\nResolution: Please add the repository containing the IP(s) to the project." }
      set bCheckIPsPassed 0
   }

}

if { $bCheckIPsPassed != 1 } {
  common::send_msg_id "BD_TCL-1003" "WARNING" "Will not continue with creation of design due to the error(s) above."
  return 3
}

##################################################################
# DESIGN PROCs
##################################################################



# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell } {

  variable script_folder
  variable design_name

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_msg_id "BD_TCL-100" "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_msg_id "BD_TCL-101" "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj


  # Create interface ports
  set DDR [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR ]
  set FIXED_IO [ create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO ]

  # Create ports

  # Create instance: NBtraining_accel_1, and set properties
  set NBtraining_accel_1 [ create_bd_cell -type ip -vlnv xilinx.com:hls:NBtraining_accel:1.0 NBtraining_accel_1 ]

  # Create instance: NBtraining_accel_1_if, and set properties
  set NBtraining_accel_1_if [ create_bd_cell -type ip -vlnv xilinx.com:ip:adapter_v3_0:1.0 NBtraining_accel_1_if ]
  set_property -dict [ list \
   CONFIG.C_INPUT_SCALAR_0_WIDTH {32} \
   CONFIG.C_NUM_INPUT_BRAMs {1} \
   CONFIG.C_NUM_INPUT_FIFOs {1} \
   CONFIG.C_NUM_OUTPUT_FIFOs {3} \
   CONFIG.C_N_INPUT_SCALARS {1} \
   CONFIG.M_AXIS_FIFO_0_BYTE_WIDTH {32} \
   CONFIG.M_AXIS_FIFO_0_DEPTH {1024} \
   CONFIG.M_AXIS_FIFO_0_DMWIDTH {64} \
   CONFIG.M_AXIS_FIFO_0_WIDTH {32} \
   CONFIG.M_AXIS_FIFO_1_BYTE_WIDTH {32} \
   CONFIG.M_AXIS_FIFO_1_DEPTH {1024} \
   CONFIG.M_AXIS_FIFO_1_DMWIDTH {64} \
   CONFIG.M_AXIS_FIFO_1_WIDTH {32} \
   CONFIG.M_AXIS_FIFO_2_BYTE_WIDTH {32} \
   CONFIG.M_AXIS_FIFO_2_DEPTH {1024} \
   CONFIG.M_AXIS_FIFO_2_DMWIDTH {64} \
   CONFIG.M_AXIS_FIFO_2_WIDTH {32} \
   CONFIG.S_AXIS_BRAM_0_ADDR_WIDTH {6} \
   CONFIG.S_AXIS_BRAM_0_DEPTH {16} \
   CONFIG.S_AXIS_BRAM_0_DMWIDTH {64} \
   CONFIG.S_AXIS_BRAM_0_MB_DEPTH {1} \
   CONFIG.S_AXIS_BRAM_0_WIDTH {32} \
   CONFIG.S_AXIS_FIFO_0_BYTE_WIDTH {32} \
   CONFIG.S_AXIS_FIFO_0_DEPTH {1024} \
   CONFIG.S_AXIS_FIFO_0_DMWIDTH {64} \
   CONFIG.S_AXIS_FIFO_0_WIDTH {32} \
 ] $NBtraining_accel_1_if

  # Create instance: axcache_0xE, and set properties
  set axcache_0xE [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 axcache_0xE ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {14} \
   CONFIG.CONST_WIDTH {4} \
 ] $axcache_0xE

  # Create instance: axi_ic_ps7_M_AXI_GP0, and set properties
  set axi_ic_ps7_M_AXI_GP0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ic_ps7_M_AXI_GP0 ]
  set_property -dict [ list \
   CONFIG.M00_HAS_REGSLICE {1} \
   CONFIG.M01_HAS_REGSLICE {1} \
   CONFIG.M02_HAS_REGSLICE {1} \
   CONFIG.M03_HAS_REGSLICE {1} \
   CONFIG.M04_HAS_REGSLICE {1} \
   CONFIG.M05_HAS_REGSLICE {1} \
   CONFIG.NUM_MI {6} \
   CONFIG.NUM_SI {1} \
   CONFIG.S00_HAS_DATA_FIFO {2} \
   CONFIG.S00_HAS_REGSLICE {1} \
   CONFIG.STRATEGY {2} \
 ] $axi_ic_ps7_M_AXI_GP0

  # Create instance: axi_ic_ps7_S_AXI_ACP, and set properties
  set axi_ic_ps7_S_AXI_ACP [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ic_ps7_S_AXI_ACP ]
  set_property -dict [ list \
   CONFIG.M00_HAS_DATA_FIFO {2} \
   CONFIG.M00_HAS_REGSLICE {1} \
   CONFIG.NUM_MI {1} \
   CONFIG.NUM_SI {5} \
   CONFIG.S00_HAS_DATA_FIFO {2} \
   CONFIG.S00_HAS_REGSLICE {1} \
   CONFIG.S01_HAS_DATA_FIFO {2} \
   CONFIG.S01_HAS_REGSLICE {1} \
   CONFIG.S02_HAS_DATA_FIFO {2} \
   CONFIG.S02_HAS_REGSLICE {1} \
   CONFIG.S03_HAS_DATA_FIFO {2} \
   CONFIG.S03_HAS_REGSLICE {1} \
   CONFIG.S04_HAS_DATA_FIFO {2} \
   CONFIG.S04_HAS_REGSLICE {1} \
   CONFIG.STRATEGY {2} \
 ] $axi_ic_ps7_S_AXI_ACP

  # Create instance: clk_wiz_0, and set properties
  set clk_wiz_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:5.4 clk_wiz_0 ]
  set_property -dict [ list \
   CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {100} \
   CONFIG.CLKOUT2_JITTER {122.190} \
   CONFIG.CLKOUT2_PHASE_ERROR {98.575} \
   CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {142} \
   CONFIG.CLKOUT2_USED {true} \
   CONFIG.CLKOUT3_JITTER {118.758} \
   CONFIG.CLKOUT3_PHASE_ERROR {98.575} \
   CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {166} \
   CONFIG.CLKOUT3_USED {true} \
   CONFIG.CLKOUT4_JITTER {114.829} \
   CONFIG.CLKOUT4_PHASE_ERROR {98.575} \
   CONFIG.CLKOUT4_REQUESTED_OUT_FREQ {200} \
   CONFIG.CLKOUT4_USED {true} \
   CONFIG.MMCM_CLKOUT1_DIVIDE {7} \
   CONFIG.MMCM_CLKOUT2_DIVIDE {6} \
   CONFIG.MMCM_CLKOUT3_DIVIDE {5} \
   CONFIG.MMCM_DIVCLK_DIVIDE {1} \
   CONFIG.NUM_OUT_CLKS {4} \
   CONFIG.RESET_PORT {resetn} \
   CONFIG.RESET_TYPE {ACTIVE_LOW} \
 ] $clk_wiz_0

  # Create instance: dm_0, and set properties
  set dm_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dm_0 ]
  set_property -dict [ list \
   CONFIG.c_dlytmr_resolution {1250} \
   CONFIG.c_include_mm2s {1} \
   CONFIG.c_include_mm2s_dre {1} \
   CONFIG.c_include_mm2s_sf {1} \
   CONFIG.c_include_s2mm {0} \
   CONFIG.c_include_sg {0} \
   CONFIG.c_m_axi_mm2s_data_width {64} \
   CONFIG.c_m_axis_mm2s_tdata_width {64} \
   CONFIG.c_mm2s_burst_size {64} \
   CONFIG.c_sg_include_stscntrl_strm {0} \
   CONFIG.c_sg_length_width {23} \
 ] $dm_0

  # Create instance: dm_1, and set properties
  set dm_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dm_1 ]
  set_property -dict [ list \
   CONFIG.c_dlytmr_resolution {1250} \
   CONFIG.c_include_mm2s {1} \
   CONFIG.c_include_mm2s_dre {1} \
   CONFIG.c_include_mm2s_sf {1} \
   CONFIG.c_include_s2mm {0} \
   CONFIG.c_include_sg {0} \
   CONFIG.c_m_axi_mm2s_data_width {64} \
   CONFIG.c_m_axis_mm2s_tdata_width {64} \
   CONFIG.c_mm2s_burst_size {64} \
   CONFIG.c_sg_include_stscntrl_strm {0} \
   CONFIG.c_sg_length_width {23} \
 ] $dm_1

  # Create instance: dm_2, and set properties
  set dm_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dm_2 ]
  set_property -dict [ list \
   CONFIG.c_dlytmr_resolution {1250} \
   CONFIG.c_include_mm2s {0} \
   CONFIG.c_include_s2mm {1} \
   CONFIG.c_include_s2mm_dre {1} \
   CONFIG.c_include_s2mm_sf {1} \
   CONFIG.c_include_sg {0} \
   CONFIG.c_m_axi_s2mm_data_width {64} \
   CONFIG.c_s2mm_burst_size {64} \
   CONFIG.c_sg_include_stscntrl_strm {0} \
   CONFIG.c_sg_length_width {23} \
 ] $dm_2

  # Create instance: dm_3, and set properties
  set dm_3 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dm_3 ]
  set_property -dict [ list \
   CONFIG.c_dlytmr_resolution {1250} \
   CONFIG.c_include_mm2s {0} \
   CONFIG.c_include_s2mm {1} \
   CONFIG.c_include_s2mm_dre {1} \
   CONFIG.c_include_s2mm_sf {1} \
   CONFIG.c_include_sg {0} \
   CONFIG.c_m_axi_s2mm_data_width {64} \
   CONFIG.c_s2mm_burst_size {64} \
   CONFIG.c_sg_include_stscntrl_strm {0} \
   CONFIG.c_sg_length_width {23} \
 ] $dm_3

  # Create instance: dm_4, and set properties
  set dm_4 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dm_4 ]
  set_property -dict [ list \
   CONFIG.c_dlytmr_resolution {1250} \
   CONFIG.c_include_mm2s {0} \
   CONFIG.c_include_s2mm {1} \
   CONFIG.c_include_s2mm_dre {1} \
   CONFIG.c_include_s2mm_sf {1} \
   CONFIG.c_include_sg {0} \
   CONFIG.c_m_axi_s2mm_data_width {64} \
   CONFIG.c_s2mm_burst_size {64} \
   CONFIG.c_sg_include_stscntrl_strm {0} \
   CONFIG.c_sg_length_width {23} \
 ] $dm_4

  # Create instance: proc_sys_reset_0, and set properties
  set proc_sys_reset_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0 ]

  # Create instance: proc_sys_reset_1, and set properties
  set proc_sys_reset_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_1 ]

  # Create instance: proc_sys_reset_2, and set properties
  set proc_sys_reset_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_2 ]

  # Create instance: proc_sys_reset_3, and set properties
  set proc_sys_reset_3 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_3 ]

  # Create instance: ps7, and set properties
  set ps7 [ create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7 ]
  set_property -dict [ list \
   CONFIG.PCW_ACT_APU_PERIPHERAL_FREQMHZ {666.666687} \
   CONFIG.PCW_ACT_CAN0_PERIPHERAL_FREQMHZ {23.8095} \
   CONFIG.PCW_ACT_CAN1_PERIPHERAL_FREQMHZ {23.8095} \
   CONFIG.PCW_ACT_CAN_PERIPHERAL_FREQMHZ {10.000000} \
   CONFIG.PCW_ACT_DCI_PERIPHERAL_FREQMHZ {10.158730} \
   CONFIG.PCW_ACT_ENET0_PERIPHERAL_FREQMHZ {125.000000} \
   CONFIG.PCW_ACT_ENET1_PERIPHERAL_FREQMHZ {10.000000} \
   CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
   CONFIG.PCW_ACT_FPGA1_PERIPHERAL_FREQMHZ {10.000000} \
   CONFIG.PCW_ACT_FPGA2_PERIPHERAL_FREQMHZ {10.000000} \
   CONFIG.PCW_ACT_FPGA3_PERIPHERAL_FREQMHZ {10.000000} \
   CONFIG.PCW_ACT_I2C_PERIPHERAL_FREQMHZ {50} \
   CONFIG.PCW_ACT_PCAP_PERIPHERAL_FREQMHZ {200.000000} \
   CONFIG.PCW_ACT_QSPI_PERIPHERAL_FREQMHZ {200.000000} \
   CONFIG.PCW_ACT_SDIO_PERIPHERAL_FREQMHZ {50.000000} \
   CONFIG.PCW_ACT_SMC_PERIPHERAL_FREQMHZ {10.000000} \
   CONFIG.PCW_ACT_SPI_PERIPHERAL_FREQMHZ {10.000000} \
   CONFIG.PCW_ACT_TPIU_PERIPHERAL_FREQMHZ {200.000000} \
   CONFIG.PCW_ACT_TTC0_CLK0_PERIPHERAL_FREQMHZ {111.111115} \
   CONFIG.PCW_ACT_TTC0_CLK1_PERIPHERAL_FREQMHZ {111.111115} \
   CONFIG.PCW_ACT_TTC0_CLK2_PERIPHERAL_FREQMHZ {111.111115} \
   CONFIG.PCW_ACT_TTC1_CLK0_PERIPHERAL_FREQMHZ {111.111115} \
   CONFIG.PCW_ACT_TTC1_CLK1_PERIPHERAL_FREQMHZ {111.111115} \
   CONFIG.PCW_ACT_TTC1_CLK2_PERIPHERAL_FREQMHZ {111.111115} \
   CONFIG.PCW_ACT_TTC_PERIPHERAL_FREQMHZ {50} \
   CONFIG.PCW_ACT_UART_PERIPHERAL_FREQMHZ {50.000000} \
   CONFIG.PCW_ACT_USB0_PERIPHERAL_FREQMHZ {60} \
   CONFIG.PCW_ACT_USB1_PERIPHERAL_FREQMHZ {60} \
   CONFIG.PCW_ACT_WDT_PERIPHERAL_FREQMHZ {111.111115} \
   CONFIG.PCW_APU_CLK_RATIO_ENABLE {6:2:1} \
   CONFIG.PCW_APU_PERIPHERAL_FREQMHZ {666.666667} \
   CONFIG.PCW_ARMPLL_CTRL_FBDIV {40} \
   CONFIG.PCW_CAN0_BASEADDR {0xE0008000} \
   CONFIG.PCW_CAN0_GRP_CLK_ENABLE {0} \
   CONFIG.PCW_CAN0_HIGHADDR {0xE0008FFF} \
   CONFIG.PCW_CAN0_PERIPHERAL_CLKSRC {External} \
   CONFIG.PCW_CAN0_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_CAN0_PERIPHERAL_FREQMHZ {-1} \
   CONFIG.PCW_CAN1_BASEADDR {0xE0009000} \
   CONFIG.PCW_CAN1_GRP_CLK_ENABLE {0} \
   CONFIG.PCW_CAN1_HIGHADDR {0xE0009FFF} \
   CONFIG.PCW_CAN1_PERIPHERAL_CLKSRC {External} \
   CONFIG.PCW_CAN1_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_CAN1_PERIPHERAL_FREQMHZ {-1} \
   CONFIG.PCW_CAN_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_CAN_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_CAN_PERIPHERAL_DIVISOR1 {1} \
   CONFIG.PCW_CAN_PERIPHERAL_FREQMHZ {100} \
   CONFIG.PCW_CAN_PERIPHERAL_VALID {0} \
   CONFIG.PCW_CLK0_FREQ {100000000} \
   CONFIG.PCW_CLK1_FREQ {10000000} \
   CONFIG.PCW_CLK2_FREQ {10000000} \
   CONFIG.PCW_CLK3_FREQ {10000000} \
   CONFIG.PCW_CORE0_FIQ_INTR {0} \
   CONFIG.PCW_CORE0_IRQ_INTR {0} \
   CONFIG.PCW_CORE1_FIQ_INTR {0} \
   CONFIG.PCW_CORE1_IRQ_INTR {0} \
   CONFIG.PCW_CPU_CPU_6X4X_MAX_RANGE {667} \
   CONFIG.PCW_CPU_CPU_PLL_FREQMHZ {1333.333} \
   CONFIG.PCW_CPU_PERIPHERAL_CLKSRC {ARM PLL} \
   CONFIG.PCW_CPU_PERIPHERAL_DIVISOR0 {2} \
   CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {33.333333} \
   CONFIG.PCW_DCI_PERIPHERAL_CLKSRC {DDR PLL} \
   CONFIG.PCW_DCI_PERIPHERAL_DIVISOR0 {15} \
   CONFIG.PCW_DCI_PERIPHERAL_DIVISOR1 {7} \
   CONFIG.PCW_DCI_PERIPHERAL_FREQMHZ {10.159} \
   CONFIG.PCW_DDRPLL_CTRL_FBDIV {32} \
   CONFIG.PCW_DDR_DDR_PLL_FREQMHZ {1066.667} \
   CONFIG.PCW_DDR_HPRLPR_QUEUE_PARTITION {HPR(0)/LPR(32)} \
   CONFIG.PCW_DDR_HPR_TO_CRITICAL_PRIORITY_LEVEL {15} \
   CONFIG.PCW_DDR_LPR_TO_CRITICAL_PRIORITY_LEVEL {2} \
   CONFIG.PCW_DDR_PERIPHERAL_CLKSRC {DDR PLL} \
   CONFIG.PCW_DDR_PERIPHERAL_DIVISOR0 {2} \
   CONFIG.PCW_DDR_PORT0_HPR_ENABLE {0} \
   CONFIG.PCW_DDR_PORT1_HPR_ENABLE {0} \
   CONFIG.PCW_DDR_PORT2_HPR_ENABLE {0} \
   CONFIG.PCW_DDR_PORT3_HPR_ENABLE {0} \
   CONFIG.PCW_DDR_RAM_BASEADDR {0x00100000} \
   CONFIG.PCW_DDR_RAM_HIGHADDR {0x1FFFFFFF} \
   CONFIG.PCW_DDR_WRITE_TO_CRITICAL_PRIORITY_LEVEL {2} \
   CONFIG.PCW_DM_WIDTH {4} \
   CONFIG.PCW_DQS_WIDTH {4} \
   CONFIG.PCW_DQ_WIDTH {32} \
   CONFIG.PCW_ENET0_BASEADDR {0xE000B000} \
   CONFIG.PCW_ENET0_ENET0_IO {MIO 16 .. 27} \
   CONFIG.PCW_ENET0_GRP_MDIO_ENABLE {1} \
   CONFIG.PCW_ENET0_GRP_MDIO_IO {MIO 52 .. 53} \
   CONFIG.PCW_ENET0_HIGHADDR {0xE000BFFF} \
   CONFIG.PCW_ENET0_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_ENET0_PERIPHERAL_DIVISOR0 {8} \
   CONFIG.PCW_ENET0_PERIPHERAL_DIVISOR1 {1} \
   CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {1} \
   CONFIG.PCW_ENET0_PERIPHERAL_FREQMHZ {1000 Mbps} \
   CONFIG.PCW_ENET0_RESET_ENABLE {0} \
   CONFIG.PCW_ENET1_BASEADDR {0xE000C000} \
   CONFIG.PCW_ENET1_GRP_MDIO_ENABLE {0} \
   CONFIG.PCW_ENET1_HIGHADDR {0xE000CFFF} \
   CONFIG.PCW_ENET1_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_ENET1_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_ENET1_PERIPHERAL_DIVISOR1 {1} \
   CONFIG.PCW_ENET1_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_ENET1_PERIPHERAL_FREQMHZ {1000 Mbps} \
   CONFIG.PCW_ENET1_RESET_ENABLE {0} \
   CONFIG.PCW_ENET_RESET_ENABLE {1} \
   CONFIG.PCW_ENET_RESET_POLARITY {Active Low} \
   CONFIG.PCW_ENET_RESET_SELECT {Share reset pin} \
   CONFIG.PCW_EN_4K_TIMER {0} \
   CONFIG.PCW_EN_CAN0 {0} \
   CONFIG.PCW_EN_CAN1 {0} \
   CONFIG.PCW_EN_CLK0_PORT {1} \
   CONFIG.PCW_EN_CLK1_PORT {0} \
   CONFIG.PCW_EN_CLK2_PORT {0} \
   CONFIG.PCW_EN_CLK3_PORT {0} \
   CONFIG.PCW_EN_CLKTRIG0_PORT {0} \
   CONFIG.PCW_EN_CLKTRIG1_PORT {0} \
   CONFIG.PCW_EN_CLKTRIG2_PORT {0} \
   CONFIG.PCW_EN_CLKTRIG3_PORT {0} \
   CONFIG.PCW_EN_DDR {1} \
   CONFIG.PCW_EN_EMIO_CAN0 {0} \
   CONFIG.PCW_EN_EMIO_CAN1 {0} \
   CONFIG.PCW_EN_EMIO_CD_SDIO0 {0} \
   CONFIG.PCW_EN_EMIO_CD_SDIO1 {0} \
   CONFIG.PCW_EN_EMIO_ENET0 {0} \
   CONFIG.PCW_EN_EMIO_ENET1 {0} \
   CONFIG.PCW_EN_EMIO_GPIO {0} \
   CONFIG.PCW_EN_EMIO_I2C0 {0} \
   CONFIG.PCW_EN_EMIO_I2C1 {0} \
   CONFIG.PCW_EN_EMIO_MODEM_UART0 {0} \
   CONFIG.PCW_EN_EMIO_MODEM_UART1 {0} \
   CONFIG.PCW_EN_EMIO_PJTAG {0} \
   CONFIG.PCW_EN_EMIO_SDIO0 {0} \
   CONFIG.PCW_EN_EMIO_SDIO1 {0} \
   CONFIG.PCW_EN_EMIO_SPI0 {0} \
   CONFIG.PCW_EN_EMIO_SPI1 {0} \
   CONFIG.PCW_EN_EMIO_SRAM_INT {0} \
   CONFIG.PCW_EN_EMIO_TRACE {0} \
   CONFIG.PCW_EN_EMIO_TTC0 {1} \
   CONFIG.PCW_EN_EMIO_TTC1 {0} \
   CONFIG.PCW_EN_EMIO_UART0 {0} \
   CONFIG.PCW_EN_EMIO_UART1 {0} \
   CONFIG.PCW_EN_EMIO_WDT {0} \
   CONFIG.PCW_EN_EMIO_WP_SDIO0 {0} \
   CONFIG.PCW_EN_EMIO_WP_SDIO1 {0} \
   CONFIG.PCW_EN_ENET0 {1} \
   CONFIG.PCW_EN_ENET1 {0} \
   CONFIG.PCW_EN_GPIO {1} \
   CONFIG.PCW_EN_I2C0 {0} \
   CONFIG.PCW_EN_I2C1 {0} \
   CONFIG.PCW_EN_MODEM_UART0 {0} \
   CONFIG.PCW_EN_MODEM_UART1 {0} \
   CONFIG.PCW_EN_PJTAG {0} \
   CONFIG.PCW_EN_PTP_ENET0 {0} \
   CONFIG.PCW_EN_PTP_ENET1 {0} \
   CONFIG.PCW_EN_QSPI {1} \
   CONFIG.PCW_EN_RST0_PORT {1} \
   CONFIG.PCW_EN_RST1_PORT {0} \
   CONFIG.PCW_EN_RST2_PORT {0} \
   CONFIG.PCW_EN_RST3_PORT {0} \
   CONFIG.PCW_EN_SDIO0 {1} \
   CONFIG.PCW_EN_SDIO1 {0} \
   CONFIG.PCW_EN_SMC {0} \
   CONFIG.PCW_EN_SPI0 {0} \
   CONFIG.PCW_EN_SPI1 {0} \
   CONFIG.PCW_EN_TRACE {0} \
   CONFIG.PCW_EN_TTC0 {1} \
   CONFIG.PCW_EN_TTC1 {0} \
   CONFIG.PCW_EN_UART0 {0} \
   CONFIG.PCW_EN_UART1 {1} \
   CONFIG.PCW_EN_USB0 {1} \
   CONFIG.PCW_EN_USB1 {0} \
   CONFIG.PCW_EN_WDT {0} \
   CONFIG.PCW_FCLK0_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_FCLK0_PERIPHERAL_DIVISOR0 {5} \
   CONFIG.PCW_FCLK0_PERIPHERAL_DIVISOR1 {2} \
   CONFIG.PCW_FCLK1_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_FCLK1_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_FCLK1_PERIPHERAL_DIVISOR1 {1} \
   CONFIG.PCW_FCLK2_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_FCLK2_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_FCLK2_PERIPHERAL_DIVISOR1 {1} \
   CONFIG.PCW_FCLK3_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_FCLK3_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_FCLK3_PERIPHERAL_DIVISOR1 {1} \
   CONFIG.PCW_FCLK_CLK0_BUF {TRUE} \
   CONFIG.PCW_FCLK_CLK1_BUF {FALSE} \
   CONFIG.PCW_FCLK_CLK2_BUF {FALSE} \
   CONFIG.PCW_FCLK_CLK3_BUF {FALSE} \
   CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
   CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {150.000000} \
   CONFIG.PCW_FPGA2_PERIPHERAL_FREQMHZ {50} \
   CONFIG.PCW_FPGA3_PERIPHERAL_FREQMHZ {50} \
   CONFIG.PCW_FPGA_FCLK0_ENABLE {1} \
   CONFIG.PCW_FPGA_FCLK1_ENABLE {0} \
   CONFIG.PCW_FPGA_FCLK2_ENABLE {0} \
   CONFIG.PCW_FPGA_FCLK3_ENABLE {0} \
   CONFIG.PCW_GP0_EN_MODIFIABLE_TXN {1} \
   CONFIG.PCW_GP0_NUM_READ_THREADS {4} \
   CONFIG.PCW_GP0_NUM_WRITE_THREADS {4} \
   CONFIG.PCW_GP1_EN_MODIFIABLE_TXN {1} \
   CONFIG.PCW_GP1_NUM_READ_THREADS {4} \
   CONFIG.PCW_GP1_NUM_WRITE_THREADS {4} \
   CONFIG.PCW_GPIO_BASEADDR {0xE000A000} \
   CONFIG.PCW_GPIO_EMIO_GPIO_ENABLE {0} \
   CONFIG.PCW_GPIO_EMIO_GPIO_WIDTH {64} \
   CONFIG.PCW_GPIO_HIGHADDR {0xE000AFFF} \
   CONFIG.PCW_GPIO_MIO_GPIO_ENABLE {1} \
   CONFIG.PCW_GPIO_MIO_GPIO_IO {MIO} \
   CONFIG.PCW_GPIO_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_I2C0_BASEADDR {0xE0004000} \
   CONFIG.PCW_I2C0_GRP_INT_ENABLE {0} \
   CONFIG.PCW_I2C0_HIGHADDR {0xE0004FFF} \
   CONFIG.PCW_I2C0_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_I2C0_RESET_ENABLE {0} \
   CONFIG.PCW_I2C1_BASEADDR {0xE0005000} \
   CONFIG.PCW_I2C1_GRP_INT_ENABLE {0} \
   CONFIG.PCW_I2C1_HIGHADDR {0xE0005FFF} \
   CONFIG.PCW_I2C1_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_I2C1_RESET_ENABLE {0} \
   CONFIG.PCW_I2C_PERIPHERAL_FREQMHZ {25} \
   CONFIG.PCW_I2C_RESET_ENABLE {1} \
   CONFIG.PCW_I2C_RESET_POLARITY {Active Low} \
   CONFIG.PCW_IMPORT_BOARD_PRESET {None} \
   CONFIG.PCW_INCLUDE_ACP_TRANS_CHECK {0} \
   CONFIG.PCW_INCLUDE_TRACE_BUFFER {0} \
   CONFIG.PCW_IOPLL_CTRL_FBDIV {30} \
   CONFIG.PCW_IO_IO_PLL_FREQMHZ {1000.000} \
   CONFIG.PCW_IRQ_F2P_INTR {1} \
   CONFIG.PCW_IRQ_F2P_MODE {DIRECT} \
   CONFIG.PCW_MIO_0_DIRECTION {inout} \
   CONFIG.PCW_MIO_0_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_0_PULLUP {disabled} \
   CONFIG.PCW_MIO_0_SLEW {slow} \
   CONFIG.PCW_MIO_10_DIRECTION {inout} \
   CONFIG.PCW_MIO_10_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_10_PULLUP {disabled} \
   CONFIG.PCW_MIO_10_SLEW {slow} \
   CONFIG.PCW_MIO_11_DIRECTION {inout} \
   CONFIG.PCW_MIO_11_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_11_PULLUP {disabled} \
   CONFIG.PCW_MIO_11_SLEW {slow} \
   CONFIG.PCW_MIO_12_DIRECTION {inout} \
   CONFIG.PCW_MIO_12_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_12_PULLUP {disabled} \
   CONFIG.PCW_MIO_12_SLEW {slow} \
   CONFIG.PCW_MIO_13_DIRECTION {inout} \
   CONFIG.PCW_MIO_13_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_13_PULLUP {disabled} \
   CONFIG.PCW_MIO_13_SLEW {slow} \
   CONFIG.PCW_MIO_14_DIRECTION {inout} \
   CONFIG.PCW_MIO_14_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_14_PULLUP {disabled} \
   CONFIG.PCW_MIO_14_SLEW {slow} \
   CONFIG.PCW_MIO_15_DIRECTION {inout} \
   CONFIG.PCW_MIO_15_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_15_PULLUP {disabled} \
   CONFIG.PCW_MIO_15_SLEW {slow} \
   CONFIG.PCW_MIO_16_DIRECTION {out} \
   CONFIG.PCW_MIO_16_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_16_PULLUP {disabled} \
   CONFIG.PCW_MIO_16_SLEW {fast} \
   CONFIG.PCW_MIO_17_DIRECTION {out} \
   CONFIG.PCW_MIO_17_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_17_PULLUP {disabled} \
   CONFIG.PCW_MIO_17_SLEW {fast} \
   CONFIG.PCW_MIO_18_DIRECTION {out} \
   CONFIG.PCW_MIO_18_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_18_PULLUP {disabled} \
   CONFIG.PCW_MIO_18_SLEW {fast} \
   CONFIG.PCW_MIO_19_DIRECTION {out} \
   CONFIG.PCW_MIO_19_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_19_PULLUP {disabled} \
   CONFIG.PCW_MIO_19_SLEW {fast} \
   CONFIG.PCW_MIO_1_DIRECTION {out} \
   CONFIG.PCW_MIO_1_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_1_PULLUP {disabled} \
   CONFIG.PCW_MIO_1_SLEW {fast} \
   CONFIG.PCW_MIO_20_DIRECTION {out} \
   CONFIG.PCW_MIO_20_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_20_PULLUP {disabled} \
   CONFIG.PCW_MIO_20_SLEW {fast} \
   CONFIG.PCW_MIO_21_DIRECTION {out} \
   CONFIG.PCW_MIO_21_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_21_PULLUP {disabled} \
   CONFIG.PCW_MIO_21_SLEW {fast} \
   CONFIG.PCW_MIO_22_DIRECTION {in} \
   CONFIG.PCW_MIO_22_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_22_PULLUP {disabled} \
   CONFIG.PCW_MIO_22_SLEW {fast} \
   CONFIG.PCW_MIO_23_DIRECTION {in} \
   CONFIG.PCW_MIO_23_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_23_PULLUP {disabled} \
   CONFIG.PCW_MIO_23_SLEW {fast} \
   CONFIG.PCW_MIO_24_DIRECTION {in} \
   CONFIG.PCW_MIO_24_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_24_PULLUP {disabled} \
   CONFIG.PCW_MIO_24_SLEW {fast} \
   CONFIG.PCW_MIO_25_DIRECTION {in} \
   CONFIG.PCW_MIO_25_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_25_PULLUP {disabled} \
   CONFIG.PCW_MIO_25_SLEW {fast} \
   CONFIG.PCW_MIO_26_DIRECTION {in} \
   CONFIG.PCW_MIO_26_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_26_PULLUP {disabled} \
   CONFIG.PCW_MIO_26_SLEW {fast} \
   CONFIG.PCW_MIO_27_DIRECTION {in} \
   CONFIG.PCW_MIO_27_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_27_PULLUP {disabled} \
   CONFIG.PCW_MIO_27_SLEW {fast} \
   CONFIG.PCW_MIO_28_DIRECTION {inout} \
   CONFIG.PCW_MIO_28_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_28_PULLUP {disabled} \
   CONFIG.PCW_MIO_28_SLEW {fast} \
   CONFIG.PCW_MIO_29_DIRECTION {in} \
   CONFIG.PCW_MIO_29_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_29_PULLUP {disabled} \
   CONFIG.PCW_MIO_29_SLEW {fast} \
   CONFIG.PCW_MIO_2_DIRECTION {inout} \
   CONFIG.PCW_MIO_2_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_2_PULLUP {disabled} \
   CONFIG.PCW_MIO_2_SLEW {fast} \
   CONFIG.PCW_MIO_30_DIRECTION {out} \
   CONFIG.PCW_MIO_30_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_30_PULLUP {disabled} \
   CONFIG.PCW_MIO_30_SLEW {fast} \
   CONFIG.PCW_MIO_31_DIRECTION {in} \
   CONFIG.PCW_MIO_31_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_31_PULLUP {disabled} \
   CONFIG.PCW_MIO_31_SLEW {fast} \
   CONFIG.PCW_MIO_32_DIRECTION {inout} \
   CONFIG.PCW_MIO_32_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_32_PULLUP {disabled} \
   CONFIG.PCW_MIO_32_SLEW {fast} \
   CONFIG.PCW_MIO_33_DIRECTION {inout} \
   CONFIG.PCW_MIO_33_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_33_PULLUP {disabled} \
   CONFIG.PCW_MIO_33_SLEW {fast} \
   CONFIG.PCW_MIO_34_DIRECTION {inout} \
   CONFIG.PCW_MIO_34_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_34_PULLUP {disabled} \
   CONFIG.PCW_MIO_34_SLEW {fast} \
   CONFIG.PCW_MIO_35_DIRECTION {inout} \
   CONFIG.PCW_MIO_35_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_35_PULLUP {disabled} \
   CONFIG.PCW_MIO_35_SLEW {fast} \
   CONFIG.PCW_MIO_36_DIRECTION {in} \
   CONFIG.PCW_MIO_36_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_36_PULLUP {disabled} \
   CONFIG.PCW_MIO_36_SLEW {fast} \
   CONFIG.PCW_MIO_37_DIRECTION {inout} \
   CONFIG.PCW_MIO_37_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_37_PULLUP {disabled} \
   CONFIG.PCW_MIO_37_SLEW {fast} \
   CONFIG.PCW_MIO_38_DIRECTION {inout} \
   CONFIG.PCW_MIO_38_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_38_PULLUP {disabled} \
   CONFIG.PCW_MIO_38_SLEW {fast} \
   CONFIG.PCW_MIO_39_DIRECTION {inout} \
   CONFIG.PCW_MIO_39_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_39_PULLUP {disabled} \
   CONFIG.PCW_MIO_39_SLEW {fast} \
   CONFIG.PCW_MIO_3_DIRECTION {inout} \
   CONFIG.PCW_MIO_3_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_3_PULLUP {disabled} \
   CONFIG.PCW_MIO_3_SLEW {fast} \
   CONFIG.PCW_MIO_40_DIRECTION {inout} \
   CONFIG.PCW_MIO_40_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_40_PULLUP {disabled} \
   CONFIG.PCW_MIO_40_SLEW {fast} \
   CONFIG.PCW_MIO_41_DIRECTION {inout} \
   CONFIG.PCW_MIO_41_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_41_PULLUP {disabled} \
   CONFIG.PCW_MIO_41_SLEW {fast} \
   CONFIG.PCW_MIO_42_DIRECTION {inout} \
   CONFIG.PCW_MIO_42_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_42_PULLUP {disabled} \
   CONFIG.PCW_MIO_42_SLEW {fast} \
   CONFIG.PCW_MIO_43_DIRECTION {inout} \
   CONFIG.PCW_MIO_43_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_43_PULLUP {disabled} \
   CONFIG.PCW_MIO_43_SLEW {fast} \
   CONFIG.PCW_MIO_44_DIRECTION {inout} \
   CONFIG.PCW_MIO_44_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_44_PULLUP {disabled} \
   CONFIG.PCW_MIO_44_SLEW {fast} \
   CONFIG.PCW_MIO_45_DIRECTION {inout} \
   CONFIG.PCW_MIO_45_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_45_PULLUP {disabled} \
   CONFIG.PCW_MIO_45_SLEW {fast} \
   CONFIG.PCW_MIO_46_DIRECTION {in} \
   CONFIG.PCW_MIO_46_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_46_PULLUP {disabled} \
   CONFIG.PCW_MIO_46_SLEW {slow} \
   CONFIG.PCW_MIO_47_DIRECTION {in} \
   CONFIG.PCW_MIO_47_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_47_PULLUP {disabled} \
   CONFIG.PCW_MIO_47_SLEW {slow} \
   CONFIG.PCW_MIO_48_DIRECTION {out} \
   CONFIG.PCW_MIO_48_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_48_PULLUP {disabled} \
   CONFIG.PCW_MIO_48_SLEW {slow} \
   CONFIG.PCW_MIO_49_DIRECTION {in} \
   CONFIG.PCW_MIO_49_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_49_PULLUP {disabled} \
   CONFIG.PCW_MIO_49_SLEW {slow} \
   CONFIG.PCW_MIO_4_DIRECTION {inout} \
   CONFIG.PCW_MIO_4_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_4_PULLUP {disabled} \
   CONFIG.PCW_MIO_4_SLEW {fast} \
   CONFIG.PCW_MIO_50_DIRECTION {inout} \
   CONFIG.PCW_MIO_50_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_50_PULLUP {disabled} \
   CONFIG.PCW_MIO_50_SLEW {slow} \
   CONFIG.PCW_MIO_51_DIRECTION {inout} \
   CONFIG.PCW_MIO_51_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_51_PULLUP {disabled} \
   CONFIG.PCW_MIO_51_SLEW {slow} \
   CONFIG.PCW_MIO_52_DIRECTION {out} \
   CONFIG.PCW_MIO_52_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_52_PULLUP {disabled} \
   CONFIG.PCW_MIO_52_SLEW {slow} \
   CONFIG.PCW_MIO_53_DIRECTION {inout} \
   CONFIG.PCW_MIO_53_IOTYPE {LVCMOS 1.8V} \
   CONFIG.PCW_MIO_53_PULLUP {disabled} \
   CONFIG.PCW_MIO_53_SLEW {slow} \
   CONFIG.PCW_MIO_5_DIRECTION {inout} \
   CONFIG.PCW_MIO_5_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_5_PULLUP {disabled} \
   CONFIG.PCW_MIO_5_SLEW {fast} \
   CONFIG.PCW_MIO_6_DIRECTION {out} \
   CONFIG.PCW_MIO_6_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_6_PULLUP {disabled} \
   CONFIG.PCW_MIO_6_SLEW {fast} \
   CONFIG.PCW_MIO_7_DIRECTION {out} \
   CONFIG.PCW_MIO_7_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_7_PULLUP {disabled} \
   CONFIG.PCW_MIO_7_SLEW {slow} \
   CONFIG.PCW_MIO_8_DIRECTION {out} \
   CONFIG.PCW_MIO_8_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_8_PULLUP {disabled} \
   CONFIG.PCW_MIO_8_SLEW {fast} \
   CONFIG.PCW_MIO_9_DIRECTION {inout} \
   CONFIG.PCW_MIO_9_IOTYPE {LVCMOS 3.3V} \
   CONFIG.PCW_MIO_9_PULLUP {disabled} \
   CONFIG.PCW_MIO_9_SLEW {slow} \
   CONFIG.PCW_MIO_PRIMITIVE {54} \
   CONFIG.PCW_MIO_TREE_PERIPHERALS {GPIO#Quad SPI Flash#Quad SPI Flash#Quad SPI Flash#Quad SPI Flash#Quad SPI Flash#Quad SPI Flash#GPIO#GPIO#GPIO#GPIO#GPIO#GPIO#GPIO#GPIO#GPIO#Enet 0#Enet 0#Enet 0#Enet 0#Enet 0#Enet 0#Enet 0#Enet 0#Enet 0#Enet 0#Enet 0#Enet 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#SD 0#SD 0#SD 0#SD 0#SD 0#SD 0#SD 0#SD 0#UART 1#UART 1#GPIO#GPIO#Enet 0#Enet 0} \
   CONFIG.PCW_MIO_TREE_SIGNALS {gpio[0]#qspi0_ss_b#qspi0_io[0]#qspi0_io[1]#qspi0_io[2]#qspi0_io[3]/HOLD_B#qspi0_sclk#gpio[7]#gpio[8]#gpio[9]#gpio[10]#gpio[11]#gpio[12]#gpio[13]#gpio[14]#gpio[15]#tx_clk#txd[0]#txd[1]#txd[2]#txd[3]#tx_ctl#rx_clk#rxd[0]#rxd[1]#rxd[2]#rxd[3]#rx_ctl#data[4]#dir#stp#nxt#data[0]#data[1]#data[2]#data[3]#clk#data[5]#data[6]#data[7]#clk#cmd#data[0]#data[1]#data[2]#data[3]#wp#cd#tx#rx#gpio[50]#gpio[51]#mdc#mdio} \
   CONFIG.PCW_M_AXI_GP0_ENABLE_STATIC_REMAP {0} \
   CONFIG.PCW_M_AXI_GP0_ID_WIDTH {12} \
   CONFIG.PCW_M_AXI_GP0_SUPPORT_NARROW_BURST {0} \
   CONFIG.PCW_M_AXI_GP0_THREAD_ID_WIDTH {12} \
   CONFIG.PCW_M_AXI_GP1_ENABLE_STATIC_REMAP {0} \
   CONFIG.PCW_M_AXI_GP1_ID_WIDTH {12} \
   CONFIG.PCW_M_AXI_GP1_SUPPORT_NARROW_BURST {0} \
   CONFIG.PCW_M_AXI_GP1_THREAD_ID_WIDTH {12} \
   CONFIG.PCW_NAND_CYCLES_T_AR {1} \
   CONFIG.PCW_NAND_CYCLES_T_CLR {1} \
   CONFIG.PCW_NAND_CYCLES_T_RC {11} \
   CONFIG.PCW_NAND_CYCLES_T_REA {1} \
   CONFIG.PCW_NAND_CYCLES_T_RR {1} \
   CONFIG.PCW_NAND_CYCLES_T_WC {11} \
   CONFIG.PCW_NAND_CYCLES_T_WP {1} \
   CONFIG.PCW_NAND_GRP_D8_ENABLE {0} \
   CONFIG.PCW_NAND_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_NOR_CS0_T_CEOE {1} \
   CONFIG.PCW_NOR_CS0_T_PC {1} \
   CONFIG.PCW_NOR_CS0_T_RC {11} \
   CONFIG.PCW_NOR_CS0_T_TR {1} \
   CONFIG.PCW_NOR_CS0_T_WC {11} \
   CONFIG.PCW_NOR_CS0_T_WP {1} \
   CONFIG.PCW_NOR_CS0_WE_TIME {0} \
   CONFIG.PCW_NOR_CS1_T_CEOE {1} \
   CONFIG.PCW_NOR_CS1_T_PC {1} \
   CONFIG.PCW_NOR_CS1_T_RC {11} \
   CONFIG.PCW_NOR_CS1_T_TR {1} \
   CONFIG.PCW_NOR_CS1_T_WC {11} \
   CONFIG.PCW_NOR_CS1_T_WP {1} \
   CONFIG.PCW_NOR_CS1_WE_TIME {0} \
   CONFIG.PCW_NOR_GRP_A25_ENABLE {0} \
   CONFIG.PCW_NOR_GRP_CS0_ENABLE {0} \
   CONFIG.PCW_NOR_GRP_CS1_ENABLE {0} \
   CONFIG.PCW_NOR_GRP_SRAM_CS0_ENABLE {0} \
   CONFIG.PCW_NOR_GRP_SRAM_CS1_ENABLE {0} \
   CONFIG.PCW_NOR_GRP_SRAM_INT_ENABLE {0} \
   CONFIG.PCW_NOR_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_NOR_SRAM_CS0_T_CEOE {1} \
   CONFIG.PCW_NOR_SRAM_CS0_T_PC {1} \
   CONFIG.PCW_NOR_SRAM_CS0_T_RC {11} \
   CONFIG.PCW_NOR_SRAM_CS0_T_TR {1} \
   CONFIG.PCW_NOR_SRAM_CS0_T_WC {11} \
   CONFIG.PCW_NOR_SRAM_CS0_T_WP {1} \
   CONFIG.PCW_NOR_SRAM_CS0_WE_TIME {0} \
   CONFIG.PCW_NOR_SRAM_CS1_T_CEOE {1} \
   CONFIG.PCW_NOR_SRAM_CS1_T_PC {1} \
   CONFIG.PCW_NOR_SRAM_CS1_T_RC {11} \
   CONFIG.PCW_NOR_SRAM_CS1_T_TR {1} \
   CONFIG.PCW_NOR_SRAM_CS1_T_WC {11} \
   CONFIG.PCW_NOR_SRAM_CS1_T_WP {1} \
   CONFIG.PCW_NOR_SRAM_CS1_WE_TIME {0} \
   CONFIG.PCW_OVERRIDE_BASIC_CLOCK {0} \
   CONFIG.PCW_P2F_CAN0_INTR {0} \
   CONFIG.PCW_P2F_CAN1_INTR {0} \
   CONFIG.PCW_P2F_CTI_INTR {0} \
   CONFIG.PCW_P2F_DMAC0_INTR {0} \
   CONFIG.PCW_P2F_DMAC1_INTR {0} \
   CONFIG.PCW_P2F_DMAC2_INTR {0} \
   CONFIG.PCW_P2F_DMAC3_INTR {0} \
   CONFIG.PCW_P2F_DMAC4_INTR {0} \
   CONFIG.PCW_P2F_DMAC5_INTR {0} \
   CONFIG.PCW_P2F_DMAC6_INTR {0} \
   CONFIG.PCW_P2F_DMAC7_INTR {0} \
   CONFIG.PCW_P2F_DMAC_ABORT_INTR {0} \
   CONFIG.PCW_P2F_ENET0_INTR {0} \
   CONFIG.PCW_P2F_ENET1_INTR {0} \
   CONFIG.PCW_P2F_GPIO_INTR {0} \
   CONFIG.PCW_P2F_I2C0_INTR {0} \
   CONFIG.PCW_P2F_I2C1_INTR {0} \
   CONFIG.PCW_P2F_QSPI_INTR {0} \
   CONFIG.PCW_P2F_SDIO0_INTR {0} \
   CONFIG.PCW_P2F_SDIO1_INTR {0} \
   CONFIG.PCW_P2F_SMC_INTR {0} \
   CONFIG.PCW_P2F_SPI0_INTR {0} \
   CONFIG.PCW_P2F_SPI1_INTR {0} \
   CONFIG.PCW_P2F_UART0_INTR {0} \
   CONFIG.PCW_P2F_UART1_INTR {0} \
   CONFIG.PCW_P2F_USB0_INTR {0} \
   CONFIG.PCW_P2F_USB1_INTR {0} \
   CONFIG.PCW_PACKAGE_DDR_BOARD_DELAY0 {0.063} \
   CONFIG.PCW_PACKAGE_DDR_BOARD_DELAY1 {0.062} \
   CONFIG.PCW_PACKAGE_DDR_BOARD_DELAY2 {0.065} \
   CONFIG.PCW_PACKAGE_DDR_BOARD_DELAY3 {0.083} \
   CONFIG.PCW_PACKAGE_DDR_DQS_TO_CLK_DELAY_0 {-0.007} \
   CONFIG.PCW_PACKAGE_DDR_DQS_TO_CLK_DELAY_1 {-0.010} \
   CONFIG.PCW_PACKAGE_DDR_DQS_TO_CLK_DELAY_2 {-0.006} \
   CONFIG.PCW_PACKAGE_DDR_DQS_TO_CLK_DELAY_3 {-0.048} \
   CONFIG.PCW_PACKAGE_NAME {clg484} \
   CONFIG.PCW_PCAP_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_PCAP_PERIPHERAL_DIVISOR0 {5} \
   CONFIG.PCW_PCAP_PERIPHERAL_FREQMHZ {200} \
   CONFIG.PCW_PERIPHERAL_BOARD_PRESET {part0} \
   CONFIG.PCW_PJTAG_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_PLL_BYPASSMODE_ENABLE {0} \
   CONFIG.PCW_PRESET_BANK0_VOLTAGE {LVCMOS 3.3V} \
   CONFIG.PCW_PRESET_BANK1_VOLTAGE {LVCMOS 1.8V} \
   CONFIG.PCW_PS7_SI_REV {PRODUCTION} \
   CONFIG.PCW_QSPI_GRP_FBCLK_ENABLE {0} \
   CONFIG.PCW_QSPI_GRP_IO1_ENABLE {0} \
   CONFIG.PCW_QSPI_GRP_SINGLE_SS_ENABLE {1} \
   CONFIG.PCW_QSPI_GRP_SINGLE_SS_IO {MIO 1 .. 6} \
   CONFIG.PCW_QSPI_GRP_SS1_ENABLE {0} \
   CONFIG.PCW_QSPI_INTERNAL_HIGHADDRESS {0xFCFFFFFF} \
   CONFIG.PCW_QSPI_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_QSPI_PERIPHERAL_DIVISOR0 {5} \
   CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {1} \
   CONFIG.PCW_QSPI_PERIPHERAL_FREQMHZ {200} \
   CONFIG.PCW_QSPI_QSPI_IO {MIO 1 .. 6} \
   CONFIG.PCW_SD0_GRP_CD_ENABLE {1} \
   CONFIG.PCW_SD0_GRP_CD_IO {MIO 47} \
   CONFIG.PCW_SD0_GRP_POW_ENABLE {0} \
   CONFIG.PCW_SD0_GRP_WP_ENABLE {1} \
   CONFIG.PCW_SD0_GRP_WP_IO {MIO 46} \
   CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
   CONFIG.PCW_SD0_SD0_IO {MIO 40 .. 45} \
   CONFIG.PCW_SD1_GRP_CD_ENABLE {0} \
   CONFIG.PCW_SD1_GRP_POW_ENABLE {0} \
   CONFIG.PCW_SD1_GRP_WP_ENABLE {0} \
   CONFIG.PCW_SD1_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_SDIO0_BASEADDR {0xE0100000} \
   CONFIG.PCW_SDIO0_HIGHADDR {0xE0100FFF} \
   CONFIG.PCW_SDIO1_BASEADDR {0xE0101000} \
   CONFIG.PCW_SDIO1_HIGHADDR {0xE0101FFF} \
   CONFIG.PCW_SDIO_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_SDIO_PERIPHERAL_DIVISOR0 {20} \
   CONFIG.PCW_SDIO_PERIPHERAL_FREQMHZ {50} \
   CONFIG.PCW_SDIO_PERIPHERAL_VALID {1} \
   CONFIG.PCW_SINGLE_QSPI_DATA_MODE {x4} \
   CONFIG.PCW_SMC_CYCLE_T0 {NA} \
   CONFIG.PCW_SMC_CYCLE_T1 {NA} \
   CONFIG.PCW_SMC_CYCLE_T2 {NA} \
   CONFIG.PCW_SMC_CYCLE_T3 {NA} \
   CONFIG.PCW_SMC_CYCLE_T4 {NA} \
   CONFIG.PCW_SMC_CYCLE_T5 {NA} \
   CONFIG.PCW_SMC_CYCLE_T6 {NA} \
   CONFIG.PCW_SMC_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_SMC_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_SMC_PERIPHERAL_FREQMHZ {100} \
   CONFIG.PCW_SMC_PERIPHERAL_VALID {0} \
   CONFIG.PCW_SPI0_BASEADDR {0xE0006000} \
   CONFIG.PCW_SPI0_GRP_SS0_ENABLE {0} \
   CONFIG.PCW_SPI0_GRP_SS1_ENABLE {0} \
   CONFIG.PCW_SPI0_GRP_SS2_ENABLE {0} \
   CONFIG.PCW_SPI0_HIGHADDR {0xE0006FFF} \
   CONFIG.PCW_SPI0_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_SPI1_BASEADDR {0xE0007000} \
   CONFIG.PCW_SPI1_GRP_SS0_ENABLE {0} \
   CONFIG.PCW_SPI1_GRP_SS1_ENABLE {0} \
   CONFIG.PCW_SPI1_GRP_SS2_ENABLE {0} \
   CONFIG.PCW_SPI1_HIGHADDR {0xE0007FFF} \
   CONFIG.PCW_SPI1_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_SPI_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_SPI_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_SPI_PERIPHERAL_FREQMHZ {166.666666} \
   CONFIG.PCW_SPI_PERIPHERAL_VALID {0} \
   CONFIG.PCW_S_AXI_ACP_ARUSER_VAL {31} \
   CONFIG.PCW_S_AXI_ACP_AWUSER_VAL {31} \
   CONFIG.PCW_S_AXI_ACP_ID_WIDTH {3} \
   CONFIG.PCW_S_AXI_GP0_ID_WIDTH {6} \
   CONFIG.PCW_S_AXI_GP1_ID_WIDTH {6} \
   CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
   CONFIG.PCW_S_AXI_HP0_ID_WIDTH {6} \
   CONFIG.PCW_S_AXI_HP1_DATA_WIDTH {64} \
   CONFIG.PCW_S_AXI_HP1_ID_WIDTH {6} \
   CONFIG.PCW_S_AXI_HP2_DATA_WIDTH {64} \
   CONFIG.PCW_S_AXI_HP2_ID_WIDTH {6} \
   CONFIG.PCW_S_AXI_HP3_DATA_WIDTH {64} \
   CONFIG.PCW_S_AXI_HP3_ID_WIDTH {6} \
   CONFIG.PCW_TPIU_PERIPHERAL_CLKSRC {External} \
   CONFIG.PCW_TPIU_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_TPIU_PERIPHERAL_FREQMHZ {200} \
   CONFIG.PCW_TRACE_BUFFER_CLOCK_DELAY {12} \
   CONFIG.PCW_TRACE_BUFFER_FIFO_SIZE {128} \
   CONFIG.PCW_TRACE_GRP_16BIT_ENABLE {0} \
   CONFIG.PCW_TRACE_GRP_2BIT_ENABLE {0} \
   CONFIG.PCW_TRACE_GRP_32BIT_ENABLE {0} \
   CONFIG.PCW_TRACE_GRP_4BIT_ENABLE {0} \
   CONFIG.PCW_TRACE_GRP_8BIT_ENABLE {0} \
   CONFIG.PCW_TRACE_INTERNAL_WIDTH {2} \
   CONFIG.PCW_TRACE_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_TRACE_PIPELINE_WIDTH {8} \
   CONFIG.PCW_TTC0_BASEADDR {0xE0104000} \
   CONFIG.PCW_TTC0_CLK0_PERIPHERAL_CLKSRC {CPU_1X} \
   CONFIG.PCW_TTC0_CLK0_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_TTC0_CLK0_PERIPHERAL_FREQMHZ {133.333333} \
   CONFIG.PCW_TTC0_CLK1_PERIPHERAL_CLKSRC {CPU_1X} \
   CONFIG.PCW_TTC0_CLK1_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_TTC0_CLK1_PERIPHERAL_FREQMHZ {133.333333} \
   CONFIG.PCW_TTC0_CLK2_PERIPHERAL_CLKSRC {CPU_1X} \
   CONFIG.PCW_TTC0_CLK2_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_TTC0_CLK2_PERIPHERAL_FREQMHZ {133.333333} \
   CONFIG.PCW_TTC0_HIGHADDR {0xE0104fff} \
   CONFIG.PCW_TTC0_PERIPHERAL_ENABLE {1} \
   CONFIG.PCW_TTC0_TTC0_IO {EMIO} \
   CONFIG.PCW_TTC1_BASEADDR {0xE0105000} \
   CONFIG.PCW_TTC1_CLK0_PERIPHERAL_CLKSRC {CPU_1X} \
   CONFIG.PCW_TTC1_CLK0_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_TTC1_CLK0_PERIPHERAL_FREQMHZ {133.333333} \
   CONFIG.PCW_TTC1_CLK1_PERIPHERAL_CLKSRC {CPU_1X} \
   CONFIG.PCW_TTC1_CLK1_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_TTC1_CLK1_PERIPHERAL_FREQMHZ {133.333333} \
   CONFIG.PCW_TTC1_CLK2_PERIPHERAL_CLKSRC {CPU_1X} \
   CONFIG.PCW_TTC1_CLK2_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_TTC1_CLK2_PERIPHERAL_FREQMHZ {133.333333} \
   CONFIG.PCW_TTC1_HIGHADDR {0xE0105fff} \
   CONFIG.PCW_TTC1_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_TTC_PERIPHERAL_FREQMHZ {50} \
   CONFIG.PCW_UART0_BASEADDR {0xE0000000} \
   CONFIG.PCW_UART0_BAUD_RATE {115200} \
   CONFIG.PCW_UART0_GRP_FULL_ENABLE {0} \
   CONFIG.PCW_UART0_HIGHADDR {0xE0000FFF} \
   CONFIG.PCW_UART0_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_UART1_BASEADDR {0xE0001000} \
   CONFIG.PCW_UART1_BAUD_RATE {115200} \
   CONFIG.PCW_UART1_GRP_FULL_ENABLE {0} \
   CONFIG.PCW_UART1_HIGHADDR {0xE0001FFF} \
   CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
   CONFIG.PCW_UART1_UART1_IO {MIO 48 .. 49} \
   CONFIG.PCW_UART_PERIPHERAL_CLKSRC {IO PLL} \
   CONFIG.PCW_UART_PERIPHERAL_DIVISOR0 {20} \
   CONFIG.PCW_UART_PERIPHERAL_FREQMHZ {50} \
   CONFIG.PCW_UART_PERIPHERAL_VALID {1} \
   CONFIG.PCW_UIPARAM_ACT_DDR_FREQ_MHZ {533.333374} \
   CONFIG.PCW_UIPARAM_DDR_ADV_ENABLE {0} \
   CONFIG.PCW_UIPARAM_DDR_AL {0} \
   CONFIG.PCW_UIPARAM_DDR_BANK_ADDR_COUNT {3} \
   CONFIG.PCW_UIPARAM_DDR_BL {8} \
   CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY0 {0.41} \
   CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY1 {0.411} \
   CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY2 {0.341} \
   CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY3 {0.358} \
   CONFIG.PCW_UIPARAM_DDR_BUS_WIDTH {32 Bit} \
   CONFIG.PCW_UIPARAM_DDR_CL {7} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_0_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_0_PACKAGE_LENGTH {61.0905} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_0_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_1_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_1_PACKAGE_LENGTH {61.0905} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_1_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_2_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_2_PACKAGE_LENGTH {61.0905} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_2_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_3_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_3_PACKAGE_LENGTH {61.0905} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_3_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_CLOCK_STOP_EN {0} \
   CONFIG.PCW_UIPARAM_DDR_COL_ADDR_COUNT {10} \
   CONFIG.PCW_UIPARAM_DDR_CWL {6} \
   CONFIG.PCW_UIPARAM_DDR_DEVICE_CAPACITY {2048 MBits} \
   CONFIG.PCW_UIPARAM_DDR_DQS_0_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_DQS_0_PACKAGE_LENGTH {68.4725} \
   CONFIG.PCW_UIPARAM_DDR_DQS_0_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_DQS_1_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_DQS_1_PACKAGE_LENGTH {71.086} \
   CONFIG.PCW_UIPARAM_DDR_DQS_1_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_DQS_2_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_DQS_2_PACKAGE_LENGTH {66.794} \
   CONFIG.PCW_UIPARAM_DDR_DQS_2_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_DQS_3_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_DQS_3_PACKAGE_LENGTH {108.7385} \
   CONFIG.PCW_UIPARAM_DDR_DQS_3_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_0 {0.025} \
   CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_1 {0.028} \
   CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_2 {-0.009} \
   CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_3 {-0.061} \
   CONFIG.PCW_UIPARAM_DDR_DQ_0_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_DQ_0_PACKAGE_LENGTH {64.1705} \
   CONFIG.PCW_UIPARAM_DDR_DQ_0_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_DQ_1_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_DQ_1_PACKAGE_LENGTH {63.686} \
   CONFIG.PCW_UIPARAM_DDR_DQ_1_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_DQ_2_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_DQ_2_PACKAGE_LENGTH {68.46} \
   CONFIG.PCW_UIPARAM_DDR_DQ_2_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_DQ_3_LENGTH_MM {0} \
   CONFIG.PCW_UIPARAM_DDR_DQ_3_PACKAGE_LENGTH {105.4895} \
   CONFIG.PCW_UIPARAM_DDR_DQ_3_PROPOGATION_DELAY {160} \
   CONFIG.PCW_UIPARAM_DDR_DRAM_WIDTH {16 Bits} \
   CONFIG.PCW_UIPARAM_DDR_ECC {Disabled} \
   CONFIG.PCW_UIPARAM_DDR_ENABLE {1} \
   CONFIG.PCW_UIPARAM_DDR_FREQ_MHZ {533.333313} \
   CONFIG.PCW_UIPARAM_DDR_HIGH_TEMP {Normal (0-85)} \
   CONFIG.PCW_UIPARAM_DDR_MEMORY_TYPE {DDR 3} \
   CONFIG.PCW_UIPARAM_DDR_PARTNO {MT41J128M16 HA-15E} \
   CONFIG.PCW_UIPARAM_DDR_ROW_ADDR_COUNT {14} \
   CONFIG.PCW_UIPARAM_DDR_SPEED_BIN {DDR3_1066F} \
   CONFIG.PCW_UIPARAM_DDR_TRAIN_DATA_EYE {1} \
   CONFIG.PCW_UIPARAM_DDR_TRAIN_READ_GATE {1} \
   CONFIG.PCW_UIPARAM_DDR_TRAIN_WRITE_LEVEL {1} \
   CONFIG.PCW_UIPARAM_DDR_T_FAW {45.0} \
   CONFIG.PCW_UIPARAM_DDR_T_RAS_MIN {36.0} \
   CONFIG.PCW_UIPARAM_DDR_T_RC {49.5} \
   CONFIG.PCW_UIPARAM_DDR_T_RCD {7} \
   CONFIG.PCW_UIPARAM_DDR_T_RP {7} \
   CONFIG.PCW_UIPARAM_DDR_USE_INTERNAL_VREF {1} \
   CONFIG.PCW_UIPARAM_GENERATE_SUMMARY {NA} \
   CONFIG.PCW_USB0_BASEADDR {0xE0102000} \
   CONFIG.PCW_USB0_HIGHADDR {0xE0102fff} \
   CONFIG.PCW_USB0_PERIPHERAL_ENABLE {1} \
   CONFIG.PCW_USB0_PERIPHERAL_FREQMHZ {60} \
   CONFIG.PCW_USB0_RESET_ENABLE {0} \
   CONFIG.PCW_USB0_USB0_IO {MIO 28 .. 39} \
   CONFIG.PCW_USB1_BASEADDR {0xE0103000} \
   CONFIG.PCW_USB1_HIGHADDR {0xE0103fff} \
   CONFIG.PCW_USB1_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_USB1_PERIPHERAL_FREQMHZ {60} \
   CONFIG.PCW_USB1_RESET_ENABLE {0} \
   CONFIG.PCW_USB_RESET_ENABLE {1} \
   CONFIG.PCW_USB_RESET_POLARITY {Active Low} \
   CONFIG.PCW_USB_RESET_SELECT {Share reset pin} \
   CONFIG.PCW_USE_AXI_FABRIC_IDLE {0} \
   CONFIG.PCW_USE_AXI_NONSECURE {0} \
   CONFIG.PCW_USE_CORESIGHT {0} \
   CONFIG.PCW_USE_CROSS_TRIGGER {0} \
   CONFIG.PCW_USE_CR_FABRIC {1} \
   CONFIG.PCW_USE_DDR_BYPASS {0} \
   CONFIG.PCW_USE_DEBUG {0} \
   CONFIG.PCW_USE_DEFAULT_ACP_USER_VAL {1} \
   CONFIG.PCW_USE_DMA0 {0} \
   CONFIG.PCW_USE_DMA1 {0} \
   CONFIG.PCW_USE_DMA2 {0} \
   CONFIG.PCW_USE_DMA3 {0} \
   CONFIG.PCW_USE_EXPANDED_IOP {0} \
   CONFIG.PCW_USE_EXPANDED_PS_SLCR_REGISTERS {0} \
   CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
   CONFIG.PCW_USE_HIGH_OCM {0} \
   CONFIG.PCW_USE_M_AXI_GP0 {1} \
   CONFIG.PCW_USE_M_AXI_GP1 {0} \
   CONFIG.PCW_USE_PROC_EVENT_BUS {0} \
   CONFIG.PCW_USE_PS_SLCR_REGISTERS {0} \
   CONFIG.PCW_USE_S_AXI_ACP {1} \
   CONFIG.PCW_USE_S_AXI_GP0 {0} \
   CONFIG.PCW_USE_S_AXI_GP1 {0} \
   CONFIG.PCW_USE_S_AXI_HP0 {0} \
   CONFIG.PCW_USE_S_AXI_HP1 {0} \
   CONFIG.PCW_USE_S_AXI_HP2 {0} \
   CONFIG.PCW_USE_S_AXI_HP3 {0} \
   CONFIG.PCW_USE_TRACE {0} \
   CONFIG.PCW_USE_TRACE_DATA_EDGE_DETECTOR {0} \
   CONFIG.PCW_VALUE_SILVERSION {3} \
   CONFIG.PCW_WDT_PERIPHERAL_CLKSRC {CPU_1X} \
   CONFIG.PCW_WDT_PERIPHERAL_DIVISOR0 {1} \
   CONFIG.PCW_WDT_PERIPHERAL_ENABLE {0} \
   CONFIG.PCW_WDT_PERIPHERAL_FREQMHZ {133.333333} \
   CONFIG.preset {ZedBoard} \
 ] $ps7

  # Create instance: sds_irq_const, and set properties
  set sds_irq_const [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 sds_irq_const ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {0} \
   CONFIG.CONST_WIDTH {1} \
 ] $sds_irq_const

  # Create instance: xlconcat_0, and set properties
  set xlconcat_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0 ]
  set_property -dict [ list \
   CONFIG.NUM_PORTS {16} \
 ] $xlconcat_0

  # Create interface connections
  connect_bd_intf_net -intf_net NBtraining_accel_1_data [get_bd_intf_pins NBtraining_accel_1/data] [get_bd_intf_pins NBtraining_accel_1_if/AP_FIFO_IARG_0]
  connect_bd_intf_net -intf_net NBtraining_accel_1_if_M_AXIS_FIFO_0 [get_bd_intf_pins NBtraining_accel_1_if/M_AXIS_FIFO_0] [get_bd_intf_pins dm_4/S_AXIS_S2MM]
  connect_bd_intf_net -intf_net NBtraining_accel_1_if_M_AXIS_FIFO_1 [get_bd_intf_pins NBtraining_accel_1_if/M_AXIS_FIFO_1] [get_bd_intf_pins dm_3/S_AXIS_S2MM]
  connect_bd_intf_net -intf_net NBtraining_accel_1_if_M_AXIS_FIFO_2 [get_bd_intf_pins NBtraining_accel_1_if/M_AXIS_FIFO_2] [get_bd_intf_pins dm_2/S_AXIS_S2MM]
  connect_bd_intf_net -intf_net NBtraining_accel_1_if_ap_ctrl [get_bd_intf_pins NBtraining_accel_1/ap_ctrl] [get_bd_intf_pins NBtraining_accel_1_if/ap_ctrl]
  connect_bd_intf_net -intf_net NBtraining_accel_1_means [get_bd_intf_pins NBtraining_accel_1/means] [get_bd_intf_pins NBtraining_accel_1_if/AP_FIFO_OARG_1]
  connect_bd_intf_net -intf_net NBtraining_accel_1_n_per_class_PORTA [get_bd_intf_pins NBtraining_accel_1/n_per_class_PORTA] [get_bd_intf_pins NBtraining_accel_1_if/AP_BRAM_IARG_0_0]
  connect_bd_intf_net -intf_net NBtraining_accel_1_priors [get_bd_intf_pins NBtraining_accel_1/priors] [get_bd_intf_pins NBtraining_accel_1_if/AP_FIFO_OARG_0]
  connect_bd_intf_net -intf_net NBtraining_accel_1_variances [get_bd_intf_pins NBtraining_accel_1/variances] [get_bd_intf_pins NBtraining_accel_1_if/AP_FIFO_OARG_2]
  connect_bd_intf_net -intf_net axi_ic_ps7_M_AXI_GP0_M00_AXI [get_bd_intf_pins NBtraining_accel_1_if/S_AXI] [get_bd_intf_pins axi_ic_ps7_M_AXI_GP0/M00_AXI]
  connect_bd_intf_net -intf_net axi_ic_ps7_M_AXI_GP0_M01_AXI [get_bd_intf_pins axi_ic_ps7_M_AXI_GP0/M01_AXI] [get_bd_intf_pins dm_0/S_AXI_LITE]
  connect_bd_intf_net -intf_net axi_ic_ps7_M_AXI_GP0_M02_AXI [get_bd_intf_pins axi_ic_ps7_M_AXI_GP0/M02_AXI] [get_bd_intf_pins dm_1/S_AXI_LITE]
  connect_bd_intf_net -intf_net axi_ic_ps7_M_AXI_GP0_M03_AXI [get_bd_intf_pins axi_ic_ps7_M_AXI_GP0/M03_AXI] [get_bd_intf_pins dm_2/S_AXI_LITE]
  connect_bd_intf_net -intf_net axi_ic_ps7_M_AXI_GP0_M04_AXI [get_bd_intf_pins axi_ic_ps7_M_AXI_GP0/M04_AXI] [get_bd_intf_pins dm_3/S_AXI_LITE]
  connect_bd_intf_net -intf_net axi_ic_ps7_M_AXI_GP0_M05_AXI [get_bd_intf_pins axi_ic_ps7_M_AXI_GP0/M05_AXI] [get_bd_intf_pins dm_4/S_AXI_LITE]
  connect_bd_intf_net -intf_net axi_ic_ps7_S_AXI_ACP_M00_AXI [get_bd_intf_pins axi_ic_ps7_S_AXI_ACP/M00_AXI] [get_bd_intf_pins ps7/S_AXI_ACP]
  connect_bd_intf_net -intf_net dm_0_M_AXIS_MM2S [get_bd_intf_pins NBtraining_accel_1_if/S_AXIS_BRAM_0] [get_bd_intf_pins dm_0/M_AXIS_MM2S]
  connect_bd_intf_net -intf_net dm_0_M_AXI_MM2S [get_bd_intf_pins axi_ic_ps7_S_AXI_ACP/S00_AXI] [get_bd_intf_pins dm_0/M_AXI_MM2S]
  connect_bd_intf_net -intf_net dm_1_M_AXIS_MM2S [get_bd_intf_pins NBtraining_accel_1_if/S_AXIS_FIFO_0] [get_bd_intf_pins dm_1/M_AXIS_MM2S]
  connect_bd_intf_net -intf_net dm_1_M_AXI_MM2S [get_bd_intf_pins axi_ic_ps7_S_AXI_ACP/S01_AXI] [get_bd_intf_pins dm_1/M_AXI_MM2S]
  connect_bd_intf_net -intf_net dm_2_M_AXI_S2MM [get_bd_intf_pins axi_ic_ps7_S_AXI_ACP/S02_AXI] [get_bd_intf_pins dm_2/M_AXI_S2MM]
  connect_bd_intf_net -intf_net dm_3_M_AXI_S2MM [get_bd_intf_pins axi_ic_ps7_S_AXI_ACP/S03_AXI] [get_bd_intf_pins dm_3/M_AXI_S2MM]
  connect_bd_intf_net -intf_net dm_4_M_AXI_S2MM [get_bd_intf_pins axi_ic_ps7_S_AXI_ACP/S04_AXI] [get_bd_intf_pins dm_4/M_AXI_S2MM]
  connect_bd_intf_net -intf_net ps7_DDR [get_bd_intf_ports DDR] [get_bd_intf_pins ps7/DDR]
  connect_bd_intf_net -intf_net ps7_FIXED_IO [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins ps7/FIXED_IO]
  connect_bd_intf_net -intf_net ps7_M_AXI_GP0 [get_bd_intf_pins axi_ic_ps7_M_AXI_GP0/S00_AXI] [get_bd_intf_pins ps7/M_AXI_GP0]

  # Create port connections
  connect_bd_net -net NBtraining_accel_1_if_ap_clk [get_bd_pins NBtraining_accel_1/ap_clk] [get_bd_pins NBtraining_accel_1_if/ap_clk]
  connect_bd_net -net NBtraining_accel_1_if_ap_iscalar_0_dout [get_bd_pins NBtraining_accel_1/DataPack] [get_bd_pins NBtraining_accel_1_if/ap_iscalar_0_dout]
  connect_bd_net -net NBtraining_accel_1_if_ap_resetn [get_bd_pins NBtraining_accel_1/ap_rst_n] [get_bd_pins NBtraining_accel_1_if/ap_resetn]
  connect_bd_net -net axcache_0xE_dout [get_bd_pins axcache_0xE/dout] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S00_AXI_arcache] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S01_AXI_arcache] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S02_AXI_awcache] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S03_AXI_awcache] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S04_AXI_awcache]
  connect_bd_net -net clk_wiz_0_clk_out1 [get_bd_pins NBtraining_accel_1_if/acc_aclk] [get_bd_pins NBtraining_accel_1_if/m_axis_fifo_0_aclk] [get_bd_pins NBtraining_accel_1_if/m_axis_fifo_1_aclk] [get_bd_pins NBtraining_accel_1_if/m_axis_fifo_2_aclk] [get_bd_pins NBtraining_accel_1_if/s_axi_aclk] [get_bd_pins NBtraining_accel_1_if/s_axis_bram_0_aclk] [get_bd_pins NBtraining_accel_1_if/s_axis_fifo_0_aclk] [get_bd_pins axi_ic_ps7_M_AXI_GP0/ACLK] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M00_ACLK] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M01_ACLK] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M02_ACLK] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M03_ACLK] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M04_ACLK] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M05_ACLK] [get_bd_pins axi_ic_ps7_M_AXI_GP0/S00_ACLK] [get_bd_pins axi_ic_ps7_S_AXI_ACP/ACLK] [get_bd_pins axi_ic_ps7_S_AXI_ACP/M00_ACLK] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S00_ACLK] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S01_ACLK] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S02_ACLK] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S03_ACLK] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S04_ACLK] [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins dm_0/m_axi_mm2s_aclk] [get_bd_pins dm_0/s_axi_lite_aclk] [get_bd_pins dm_1/m_axi_mm2s_aclk] [get_bd_pins dm_1/s_axi_lite_aclk] [get_bd_pins dm_2/m_axi_s2mm_aclk] [get_bd_pins dm_2/s_axi_lite_aclk] [get_bd_pins dm_3/m_axi_s2mm_aclk] [get_bd_pins dm_3/s_axi_lite_aclk] [get_bd_pins dm_4/m_axi_s2mm_aclk] [get_bd_pins dm_4/s_axi_lite_aclk] [get_bd_pins proc_sys_reset_0/slowest_sync_clk] [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins ps7/S_AXI_ACP_ACLK]
  connect_bd_net -net clk_wiz_0_clk_out2 [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins proc_sys_reset_1/slowest_sync_clk]
  connect_bd_net -net clk_wiz_0_clk_out3 [get_bd_pins clk_wiz_0/clk_out3] [get_bd_pins proc_sys_reset_2/slowest_sync_clk]
  connect_bd_net -net clk_wiz_0_clk_out4 [get_bd_pins clk_wiz_0/clk_out4] [get_bd_pins proc_sys_reset_3/slowest_sync_clk]
  connect_bd_net -net clk_wiz_0_locked [get_bd_pins clk_wiz_0/locked] [get_bd_pins proc_sys_reset_0/dcm_locked] [get_bd_pins proc_sys_reset_1/dcm_locked] [get_bd_pins proc_sys_reset_2/dcm_locked] [get_bd_pins proc_sys_reset_3/dcm_locked]
  connect_bd_net -net proc_sys_reset_0_interconnect_aresetn [get_bd_pins axi_ic_ps7_M_AXI_GP0/ARESETN] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M00_ARESETN] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M01_ARESETN] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M02_ARESETN] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M03_ARESETN] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M04_ARESETN] [get_bd_pins axi_ic_ps7_M_AXI_GP0/M05_ARESETN] [get_bd_pins axi_ic_ps7_M_AXI_GP0/S00_ARESETN] [get_bd_pins axi_ic_ps7_S_AXI_ACP/ARESETN] [get_bd_pins axi_ic_ps7_S_AXI_ACP/M00_ARESETN] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S00_ARESETN] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S01_ARESETN] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S02_ARESETN] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S03_ARESETN] [get_bd_pins axi_ic_ps7_S_AXI_ACP/S04_ARESETN] [get_bd_pins proc_sys_reset_0/interconnect_aresetn]
  connect_bd_net -net proc_sys_reset_0_peripheral_aresetn [get_bd_pins NBtraining_accel_1_if/acc_aresetn] [get_bd_pins NBtraining_accel_1_if/m_axis_fifo_0_aresetn] [get_bd_pins NBtraining_accel_1_if/m_axis_fifo_1_aresetn] [get_bd_pins NBtraining_accel_1_if/m_axis_fifo_2_aresetn] [get_bd_pins NBtraining_accel_1_if/s_axi_aresetn] [get_bd_pins NBtraining_accel_1_if/s_axis_bram_0_aresetn] [get_bd_pins NBtraining_accel_1_if/s_axis_fifo_0_aresetn] [get_bd_pins dm_0/axi_resetn] [get_bd_pins dm_1/axi_resetn] [get_bd_pins dm_2/axi_resetn] [get_bd_pins dm_3/axi_resetn] [get_bd_pins dm_4/axi_resetn] [get_bd_pins proc_sys_reset_0/peripheral_aresetn]
  connect_bd_net -net ps7_FCLK_CLK0 [get_bd_pins clk_wiz_0/clk_in1] [get_bd_pins ps7/FCLK_CLK0]
  connect_bd_net -net ps7_FCLK_RESET0_N [get_bd_pins clk_wiz_0/resetn] [get_bd_pins proc_sys_reset_0/ext_reset_in] [get_bd_pins proc_sys_reset_1/ext_reset_in] [get_bd_pins proc_sys_reset_2/ext_reset_in] [get_bd_pins proc_sys_reset_3/ext_reset_in] [get_bd_pins ps7/FCLK_RESET0_N]
  connect_bd_net -net sds_irq_const_dout [get_bd_pins sds_irq_const/dout] [get_bd_pins xlconcat_0/In0] [get_bd_pins xlconcat_0/In1] [get_bd_pins xlconcat_0/In2] [get_bd_pins xlconcat_0/In3] [get_bd_pins xlconcat_0/In4] [get_bd_pins xlconcat_0/In5] [get_bd_pins xlconcat_0/In6] [get_bd_pins xlconcat_0/In7] [get_bd_pins xlconcat_0/In8] [get_bd_pins xlconcat_0/In9] [get_bd_pins xlconcat_0/In10] [get_bd_pins xlconcat_0/In11] [get_bd_pins xlconcat_0/In12] [get_bd_pins xlconcat_0/In13] [get_bd_pins xlconcat_0/In14] [get_bd_pins xlconcat_0/In15]
  connect_bd_net -net xlconcat_0_dout [get_bd_pins ps7/IRQ_F2P] [get_bd_pins xlconcat_0/dout]

  # Create address segments
  create_bd_addr_seg -range 0x20000000 -offset 0x00000000 [get_bd_addr_spaces dm_0/Data_MM2S] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_ps7_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x01000000 -offset 0xFC000000 [get_bd_addr_spaces dm_0/Data_MM2S] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_QSPI_LINEAR] SEG_ps7_ACP_QSPI_LINEAR
  create_bd_addr_seg -range 0x20000000 -offset 0x00000000 [get_bd_addr_spaces dm_1/Data_MM2S] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_ps7_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x01000000 -offset 0xFC000000 [get_bd_addr_spaces dm_1/Data_MM2S] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_QSPI_LINEAR] SEG_ps7_ACP_QSPI_LINEAR
  create_bd_addr_seg -range 0x20000000 -offset 0x00000000 [get_bd_addr_spaces dm_2/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_ps7_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x01000000 -offset 0xFC000000 [get_bd_addr_spaces dm_2/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_QSPI_LINEAR] SEG_ps7_ACP_QSPI_LINEAR
  create_bd_addr_seg -range 0x20000000 -offset 0x00000000 [get_bd_addr_spaces dm_3/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_ps7_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x01000000 -offset 0xFC000000 [get_bd_addr_spaces dm_3/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_QSPI_LINEAR] SEG_ps7_ACP_QSPI_LINEAR
  create_bd_addr_seg -range 0x20000000 -offset 0x00000000 [get_bd_addr_spaces dm_4/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_DDR_LOWOCM] SEG_ps7_ACP_DDR_LOWOCM
  create_bd_addr_seg -range 0x01000000 -offset 0xFC000000 [get_bd_addr_spaces dm_4/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_QSPI_LINEAR] SEG_ps7_ACP_QSPI_LINEAR
  create_bd_addr_seg -range 0x00010000 -offset 0x43C00000 [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs NBtraining_accel_1_if/S_AXI/reg0] SEG_NBtraining_accel_1_if_reg0
  create_bd_addr_seg -range 0x00010000 -offset 0x40400000 [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs dm_0/S_AXI_LITE/Reg] SEG_dm_0_Reg
  create_bd_addr_seg -range 0x00010000 -offset 0x40410000 [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs dm_1/S_AXI_LITE/Reg] SEG_dm_1_Reg
  create_bd_addr_seg -range 0x00010000 -offset 0x40420000 [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs dm_2/S_AXI_LITE/Reg] SEG_dm_2_Reg
  create_bd_addr_seg -range 0x00010000 -offset 0x40430000 [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs dm_3/S_AXI_LITE/Reg] SEG_dm_3_Reg
  create_bd_addr_seg -range 0x00010000 -offset 0x40440000 [get_bd_addr_spaces ps7/Data] [get_bd_addr_segs dm_4/S_AXI_LITE/Reg] SEG_dm_4_Reg

  # Exclude Address Segments
  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 [get_bd_addr_spaces dm_0/Data_MM2S] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_IOP] SEG_ps7_ACP_IOP
  exclude_bd_addr_seg [get_bd_addr_segs dm_0/Data_MM2S/SEG_ps7_ACP_IOP]

  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 [get_bd_addr_spaces dm_0/Data_MM2S] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_M_AXI_GP0] SEG_ps7_ACP_M_AXI_GP0
  exclude_bd_addr_seg [get_bd_addr_segs dm_0/Data_MM2S/SEG_ps7_ACP_M_AXI_GP0]

  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 [get_bd_addr_spaces dm_1/Data_MM2S] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_IOP] SEG_ps7_ACP_IOP
  exclude_bd_addr_seg [get_bd_addr_segs dm_1/Data_MM2S/SEG_ps7_ACP_IOP]

  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 [get_bd_addr_spaces dm_1/Data_MM2S] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_M_AXI_GP0] SEG_ps7_ACP_M_AXI_GP0
  exclude_bd_addr_seg [get_bd_addr_segs dm_1/Data_MM2S/SEG_ps7_ACP_M_AXI_GP0]

  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 [get_bd_addr_spaces dm_2/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_IOP] SEG_ps7_ACP_IOP
  exclude_bd_addr_seg [get_bd_addr_segs dm_2/Data_S2MM/SEG_ps7_ACP_IOP]

  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 [get_bd_addr_spaces dm_2/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_M_AXI_GP0] SEG_ps7_ACP_M_AXI_GP0
  exclude_bd_addr_seg [get_bd_addr_segs dm_2/Data_S2MM/SEG_ps7_ACP_M_AXI_GP0]

  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 [get_bd_addr_spaces dm_3/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_IOP] SEG_ps7_ACP_IOP
  exclude_bd_addr_seg [get_bd_addr_segs dm_3/Data_S2MM/SEG_ps7_ACP_IOP]

  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 [get_bd_addr_spaces dm_3/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_M_AXI_GP0] SEG_ps7_ACP_M_AXI_GP0
  exclude_bd_addr_seg [get_bd_addr_segs dm_3/Data_S2MM/SEG_ps7_ACP_M_AXI_GP0]

  create_bd_addr_seg -range 0x00400000 -offset 0xE0000000 [get_bd_addr_spaces dm_4/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_IOP] SEG_ps7_ACP_IOP
  exclude_bd_addr_seg [get_bd_addr_segs dm_4/Data_S2MM/SEG_ps7_ACP_IOP]

  create_bd_addr_seg -range 0x40000000 -offset 0x40000000 [get_bd_addr_spaces dm_4/Data_S2MM] [get_bd_addr_segs ps7/S_AXI_ACP/ACP_M_AXI_GP0] SEG_ps7_ACP_M_AXI_GP0
  exclude_bd_addr_seg [get_bd_addr_segs dm_4/Data_S2MM/SEG_ps7_ACP_M_AXI_GP0]



  # Restore current instance
  current_bd_instance $oldCurInst

  save_bd_design
}
# End of create_root_design()


##################################################################
# MAIN FLOW
##################################################################

create_root_design ""


