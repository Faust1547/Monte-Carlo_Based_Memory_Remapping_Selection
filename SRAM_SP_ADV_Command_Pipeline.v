`timescale 1ns / 1ps
// Command timeline (10-ns example clock):
//
//   P0(posedge) : accept and latch command
//   N0(negedge) : launch stable macro pins
//   P1(posedge) : SRAM executes read/write
//   N1(negedge) : return macro to idle
//   P2(posedge) : capture read Q and pulse Read_Data_Valid

module SRAM_SP_ADV_Command_Pipeline #(
    parameter integer TAG_WIDTH = 3,
    parameter [2:0] EMA_NORMAL = 3'b000
)(
    input  wire                   clk,
    input  wire                   rst_n,

    input  wire                   Command_Valid,
    input  wire                   Command_Write,
    input  wire [13:0]            Command_Address,
    input  wire [7:0]             Command_Write_Data,
    input  wire [TAG_WIDTH-1:0]   Command_Tag,

    output wire                   Command_Ready,
    output wire                   Command_Accepted,
    output wire                   Busy,

    output reg                    Write_Done,
    output reg                    Read_Data_Valid,
    output reg  [7:0]             Read_Data,
    output reg  [TAG_WIDTH-1:0]   Completion_Tag,

    // Macro-pin visibility for waveform/TB inspection.
    output wire                   CEN_Debug,
    output wire                   WEN_Debug,
    output wire [13:0]            Address_Debug,
    output wire [7:0]             Data_In_Debug,
    output wire [7:0]             Data_Out_Debug,
    output wire [2:0]             EMA_Debug,
    output wire [TAG_WIDTH-1:0]   Macro_Tag_Debug,
    output wire [1:0]             State_Debug
);

localparam [1:0] ST_IDLE      = 2'd0;
localparam [1:0] ST_ARMED     = 2'd1;
localparam [1:0] ST_READ_WAIT = 2'd2;

reg [1:0] state;
reg       command_write_latched;
reg [13:0] command_address_latched;
reg [7:0] command_data_latched;
reg [TAG_WIDTH-1:0] command_tag_latched;

// These registers are the only signals connected to the timing model pins.
// They change exclusively on falling edges.
reg        macro_cen_reg;
reg        macro_wen_reg;
reg [13:0] macro_address_reg;
reg [7:0]  macro_data_reg;
reg [TAG_WIDTH-1:0] macro_tag_reg;
wire [7:0] macro_q;

assign Command_Ready    = rst_n && (state == ST_IDLE);
assign Command_Accepted = Command_Valid && Command_Ready;
assign Busy             = (state != ST_IDLE);
assign State_Debug      = state;

assign CEN_Debug       = macro_cen_reg;
assign WEN_Debug       = macro_wen_reg;
assign Address_Debug   = macro_address_reg;
assign Data_In_Debug   = macro_data_reg;
assign Data_Out_Debug  = macro_q;
assign EMA_Debug       = EMA_NORMAL;
assign Macro_Tag_Debug = macro_tag_reg;

// System-side state and completion pulses use the rising edge.
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                    <= ST_IDLE;
        command_write_latched    <= 1'b0;
        command_address_latched  <= 14'b0;
        command_data_latched     <= 8'b0;
        command_tag_latched      <= {TAG_WIDTH{1'b0}};
        Write_Done               <= 1'b0;
        Read_Data_Valid          <= 1'b0;
        Read_Data                <= 8'b0;
        Completion_Tag           <= {TAG_WIDTH{1'b0}};
    end
    else begin
        Write_Done      <= 1'b0;
        Read_Data_Valid <= 1'b0;

        case (state)
            ST_IDLE: begin
                if (Command_Accepted) begin
                    command_write_latched   <= Command_Write;
                    command_address_latched <= Command_Address;
                    command_data_latched    <= Command_Write_Data;
                    command_tag_latched     <= Command_Tag;
                    state                   <= ST_ARMED;
                end
            end

            // The macro operation occurs on this same rising edge because the
            // falling-edge block launched the command half a cycle earlier.
            ST_ARMED: begin
                if (command_write_latched) begin
                    Write_Done     <= 1'b1;
                    Completion_Tag <= command_tag_latched;
                    state          <= ST_IDLE;
                end
                else begin
                    state <= ST_READ_WAIT;
                end
            end

            // Q has been stable since approximately 1 ns after the preceding
            // active macro edge, so it is safe to sample here.
            ST_READ_WAIT: begin
                Read_Data       <= macro_q;
                Read_Data_Valid <= 1'b1;
                Completion_Tag  <= command_tag_latched;
                state           <= ST_IDLE;
            end

            default: state <= ST_IDLE;
        endcase
    end
end

// Macro pins change only on falling edges.  This gives 5 ns setup and 5 ns
// hold with the 10-ns test clock, comfortably beyond the macro's 1.0/0.5-ns
// setup/hold requirements.
always @(negedge clk) begin
    if (!rst_n) begin
        macro_cen_reg     <= 1'b1;
        macro_wen_reg     <= 1'b1;
        macro_address_reg <= 14'b0;
        macro_data_reg    <= 8'b0;
        macro_tag_reg     <= {TAG_WIDTH{1'b0}};
    end
    else if (state == ST_ARMED) begin
        macro_cen_reg     <= 1'b0;
        macro_wen_reg     <= ~command_write_latched;
        macro_address_reg <= command_address_latched;
        macro_data_reg    <= command_data_latched;
        macro_tag_reg     <= command_tag_latched;
    end
    else begin
        // Keep A/D/tag stable while disabling the macro.  CEN changes at the
        // falling edge, safely away from either active rising edge.
        macro_cen_reg <= 1'b1;
        macro_wen_reg <= 1'b1;
    end
end

SRAM_SP_ADV_rtl_top u_sram_macro (
    .Q   (macro_q),
    .CLK (clk),
    .CEN (macro_cen_reg),
    .WEN (macro_wen_reg),
    .A   (macro_address_reg),
    .D   (macro_data_reg),
    .EMA (EMA_NORMAL)
);

endmodule
