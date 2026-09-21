`timescale 1ns / 1ps
// =============================================================================
// Remap_Info_Memory_Page
// =============================================================================
// Remap-value storage for ONE current page.
//
// Inter:
//   32 PBG x 1 Shift Mode = 32 x 3-bit
//
// Intra:
//   32 PBG x 8 Bank x 8 RG x 1 CW = 2048 x 3-bit
//
// Global Remap Mode is held by the Control Unit; this module stores only values.
// =============================================================================

module Remap_Info_Memory_Page #(
    parameter integer PBG_LENGTH     = 5,
    parameter integer BANK_LENGTH    = 3,
    parameter integer RG_LENGTH      = 3,
    parameter integer CONTROL_WIDTH  = 3,
    parameter integer NUM_PBG        = (1 << PBG_LENGTH),
    parameter integer NUM_BANK       = (1 << BANK_LENGTH),
    parameter integer NUM_RG         = (1 << RG_LENGTH),
    parameter integer INTRA_DEPTH    = NUM_PBG * NUM_BANK * NUM_RG
)(
    input  wire                          clk,

    input  wire                          Inter_Write_Enable,
    input  wire [PBG_LENGTH-1:0]         Inter_Write_PBG,
    input  wire [CONTROL_WIDTH-1:0]      Inter_Shift_In,

    input  wire                          Intra_Write_Enable,
    input  wire [PBG_LENGTH-1:0]         Intra_Write_PBG,
    input  wire [BANK_LENGTH-1:0]        Intra_Write_Bank,
    input  wire [RG_LENGTH-1:0]          Intra_Write_RG,
    input  wire [CONTROL_WIDTH-1:0]      Intra_CW_In,

    input  wire [PBG_LENGTH-1:0]         Read_PBG,
    input  wire [BANK_LENGTH-1:0]        Read_Bank,
    input  wire [RG_LENGTH-1:0]          Read_RG,
    output wire [CONTROL_WIDTH-1:0]      Inter_Shift_Out,
    output wire [CONTROL_WIDTH-1:0]      Intra_CW_Out
);

reg [CONTROL_WIDTH-1:0] Inter_Shift_Mem [0:NUM_PBG-1];
reg [CONTROL_WIDTH-1:0] Intra_CW_Mem    [0:INTRA_DEPTH-1];

wire [PBG_LENGTH+BANK_LENGTH+RG_LENGTH-1:0] intra_read_index;
wire [PBG_LENGTH+BANK_LENGTH+RG_LENGTH-1:0] intra_write_index;

assign intra_read_index  = {Read_PBG, Read_Bank, Read_RG};
assign intra_write_index = {Intra_Write_PBG, Intra_Write_Bank, Intra_Write_RG};

assign Inter_Shift_Out = Inter_Shift_Mem[Read_PBG];
assign Intra_CW_Out    = Intra_CW_Mem[intra_read_index];

// No array-wide reset is required:
//   * Inter evaluation writes all 32 PBG locations before Data_Ready can assert.
//   * Intra evaluation writes all 2048 {PBG,Bank,RG} locations before Data_Ready.
//   * Base/Unrepairable modes never consume remap values.
// Avoiding a 2048-word reset network is substantially friendlier to synthesis.
always @(posedge clk) begin
    if (Inter_Write_Enable)
        Inter_Shift_Mem[Inter_Write_PBG] <= Inter_Shift_In;

    if (Intra_Write_Enable)
        Intra_CW_Mem[intra_write_index] <= Intra_CW_In;
end

endmodule
