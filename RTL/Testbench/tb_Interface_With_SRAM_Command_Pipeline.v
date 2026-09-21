`timescale 1ns / 1ps

module tb_TOP;
// -----------------------------------------------------------------------------
// TOP parameters
// -----------------------------------------------------------------------------
localparam integer PPN_LENGTH         = 16;
localparam integer PBG_LENGTH         = 5;
localparam integer BANK_LENGTH        = 3;
localparam integer RG_LENGTH          = 3;
localparam integer FW_LENGTH          = 3;
localparam integer FP_LENGTH          = 3;
localparam integer CW_LENGTH          = 3;
localparam integer FICAM_ENTRIES      = 64;
localparam integer FI_PER_ENTRY       = 2;
localparam integer SRAM_ADDR_LENGTH   = 14;
localparam integer WORD_ADDR_LENGTH   = 3;
localparam integer ES_WIDTH           = 20;
localparam integer TOTAL_COUNT_WIDTH  = 24;
localparam integer DATA_WIDTH         = 8;
localparam integer FI_SLOT_WIDTH      = 1 + FW_LENGTH + FP_LENGTH;
localparam integer FICAM_LENGTH       = 1 + PPN_LENGTH + PBG_LENGTH
                                       + BANK_LENGTH + RG_LENGTH
                                       + FI_PER_ENTRY*FI_SLOT_WIDTH;
localparam integer DATA_PACKET_LENGTH = 1 + SRAM_ADDR_LENGTH + DATA_WIDTH;

localparam [1:0] MODE_BASE         = 2'b00;
localparam [1:0] MODE_INTER        = 2'b01;
localparam [1:0] MODE_INTRA        = 2'b10;
localparam [1:0] MODE_UNREPAIRABLE = 2'b11;

// These three values must match constraint.dat because current TOP uses params.
localparam [TOTAL_COUNT_WIDTH-1:0] DUT_BASE_MAX  = 24'h000001;
localparam [TOTAL_COUNT_WIDTH-1:0] DUT_INTER_MAX = 24'h00053F;
localparam [TOTAL_COUNT_WIDTH-1:0] DUT_INTRA_MAX = 24'h001A3B;

// -----------------------------------------------------------------------------
// DUT I/O
// -----------------------------------------------------------------------------
reg                               clk;
reg                               rst_n;
reg                               Group_Is_Exp;
reg  [FICAM_LENGTH-1:0]           FICAM_Entry;
reg                               Eval_Start;
reg  [DATA_PACKET_LENGTH-1:0]     Data_Write_Packet;
wire                              Data_Ready;
wire [1:0]                        Remap_Mode;

// -----------------------------------------------------------------------------
// Constraint file
//   line 0 : BASE_MAX
//   line 1 : INTER_MAX
//   line 2 : INTRA_MAX
// -----------------------------------------------------------------------------
reg [TOTAL_COUNT_WIDTH-1:0] constraint_mem [0:2];
reg [TOTAL_COUNT_WIDTH-1:0] base_max;
reg [TOTAL_COUNT_WIDTH-1:0] inter_max;
reg [TOTAL_COUNT_WIDTH-1:0] intra_max;

integer tb_fault_count;
integer i;
integer timeout;

// Hardware-counted fault total from the current control unit.
// Hierarchical access is testbench-only and does not change the RTL.
wire [TOTAL_COUNT_WIDTH-1:0] dut_fault_count =
    dut.u_mc_control_unit.total_fault_count;

TOP #(
    .PPN_LENGTH        (PPN_LENGTH),
    .PBG_LENGTH        (PBG_LENGTH),
    .BANK_LENGTH       (BANK_LENGTH),
    .RG_LENGTH         (RG_LENGTH),
    .FW_LENGTH         (FW_LENGTH),
    .FP_LENGTH         (FP_LENGTH),
    .CW_LENGTH         (CW_LENGTH),
    .FICAM_ENTRIES     (FICAM_ENTRIES),
    .FI_PER_ENTRY      (FI_PER_ENTRY),
    .SRAM_ADDR_LENGTH  (SRAM_ADDR_LENGTH),
    .WORD_ADDR_LENGTH  (WORD_ADDR_LENGTH),
    .ES_WIDTH          (ES_WIDTH),
    .TOTAL_COUNT_WIDTH (TOTAL_COUNT_WIDTH),
    .DATA_WIDTH        (DATA_WIDTH),

    .BASE_MAX_FAULTS   (DUT_BASE_MAX),
    .INTER_MAX_FAULTS  (DUT_INTER_MAX),
    .INTRA_MAX_FAULTS  (DUT_INTRA_MAX)
) dut (
    .clk               (clk),
    .rst_n             (rst_n),
    .Group_Is_Exp      (Group_Is_Exp),
    .FICAM_Entry       (FICAM_Entry),
    .Eval_Start        (Eval_Start),
    .Data_Write_Packet (Data_Write_Packet),
    .Data_Ready        (Data_Ready),
    .Remap_Mode        (Remap_Mode)
);

always #5 clk = ~clk;

// -----------------------------------------------------------------------------
// FICAM entry packer
// -----------------------------------------------------------------------------
// Verified against the current FICAM.v packed order:
//   {EntryValid, PPN, PBG, Bank, RG,
//    FI1_Valid, FI1_FW, FI1_FP,
//    FI0_Valid, FI0_FW, FI0_FP}
//
// For the default widths this is exactly 42 bits.
// FI0 occupies bits [6:0], FI1 occupies bits [13:7], then RG/Bank/PBG/PPN/Valid.
// -----------------------------------------------------------------------------
function [FICAM_LENGTH-1:0] pack_ficam_entry;
    input                          entry_valid;
    input [PPN_LENGTH-1:0]         ppn;
    input [PBG_LENGTH-1:0]         pbg;
    input [BANK_LENGTH-1:0]        bank;
    input [RG_LENGTH-1:0]          rg;
    input                          fi0_valid;
    input [FW_LENGTH-1:0]          fi0_fw;
    input [FP_LENGTH-1:0]          fi0_fp;
    input                          fi1_valid;
    input [FW_LENGTH-1:0]          fi1_fw;
    input [FP_LENGTH-1:0]          fi1_fp;
    begin
        pack_ficam_entry = {
            entry_valid,
            ppn,
            pbg,
            bank,
            rg,
            fi1_valid, fi1_fw, fi1_fp,
            fi0_valid, fi0_fw, fi0_fp
        };
    end
endfunction

// Send one already-packed FICAM entry for one clock.
task send_ficam_entry;
    input [FICAM_LENGTH-1:0] entry;
    input                    fi0_valid;
    input                    fi1_valid;
    begin
        @(negedge clk);
        FICAM_Entry = entry;

        if (fi0_valid)
            tb_fault_count = tb_fault_count + 1;
        if (fi1_valid)
            tb_fault_count = tb_fault_count + 1;

        // Hold through the next positive edge so FICAM can capture it.
        @(negedge clk);
        FICAM_Entry = {FICAM_LENGTH{1'b0}};
    end
endtask

// Example stimulus: 23 entries x 2 valid FIs = 46 faults.
// Locations are deliberately spread across different PBGs.
task load_46_faults_directly_into_ficam;
    begin
        for (i = 0; i < 23; i = i + 1) begin
            send_ficam_entry(
                pack_ficam_entry(
                    1'b1,                  // Entry valid
                    {PPN_LENGTH{1'b0}},     // PPN = 0
                    i[PBG_LENGTH-1:0],      // PBG = 0..22
                    3'd0,                   // Bank
                    3'd0,                   // RG
                    1'b1, 3'd0, i[2:0],     // FI0
                    1'b1, 3'd1, i[2:0]      // FI1
                ),
                1'b1,
                1'b1
            );
        end
    end
endtask

// -----------------------------------------------------------------------------
// Summary printer
// -----------------------------------------------------------------------------
task print_constraint_summary;
    begin
        $display("==============================================================");
        $display(" FINAL CONSTRAINT DECISION SUMMARY");
        $display("==============================================================");

        $display("[RANGE] BASE         : 0 <= fault_count <= %0d",
                 base_max);
        $display("[RANGE] INTER        : %0d <= fault_count <= %0d",
                 base_max + 1'b1, inter_max);
        $display("[RANGE] INTRA        : %0d <= fault_count <= %0d",
                 inter_max + 1'b1, intra_max);
        $display("[RANGE] UNREPAIRABLE : fault_count >= %0d",
                 intra_max + 1'b1);

        $display("");

        if (dut_fault_count <= base_max)
            $display("[DECISION] %0d <= %0d", dut_fault_count, base_max);
        else if (dut_fault_count <= inter_max)
            $display("[DECISION] %0d < %0d <= %0d",
                     base_max, dut_fault_count, inter_max);
        else if (dut_fault_count <= intra_max)
            $display("[DECISION] %0d < %0d <= %0d",
                     inter_max, dut_fault_count, intra_max);
        else
            $display("[DECISION] %0d > %0d",
                     dut_fault_count, intra_max);

        case (Remap_Mode)
            MODE_BASE:
                $display("[RESULT] Selected Global Remap Mode = BASE (2'b00)");
            MODE_INTER:
                $display("[RESULT] Selected Global Remap Mode = INTER (2'b01)");
            MODE_INTRA:
                $display("[RESULT] Selected Global Remap Mode = INTRA (2'b10)");
            MODE_UNREPAIRABLE:
                $display("[RESULT] Selected Global Remap Mode = UNREPAIRABLE (2'b11)");
            default:
                $display("[RESULT] Selected Global Remap Mode = UNKNOWN (%b)", Remap_Mode);
        endcase

        $display("==============================================================");
    end
endtask

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
initial begin
    clk               = 1'b0;
    rst_n             = 1'b0;
    Group_Is_Exp      = 1'b0;
    FICAM_Entry       = {FICAM_LENGTH{1'b0}};
    Eval_Start        = 1'b0;
    Data_Write_Packet = {DATA_PACKET_LENGTH{1'b0}};
    tb_fault_count    = 0;

    // Read the 3 threshold values from the .dat file.
    $readmemh("D:/vgg16_cifar10_r980_constraint.dat", constraint_mem);
    base_max  = constraint_mem[0];
    inter_max = constraint_mem[1];
    intra_max = constraint_mem[2];

    // Current TOP uses threshold PARAMETERS, so make sure the runtime file
    // still matches the values used to elaborate the DUT.
    if ((base_max  !== DUT_BASE_MAX)  ||
        (inter_max !== DUT_INTER_MAX) ||
        (intra_max !== DUT_INTRA_MAX)) begin
        $display("[ERROR] constraint.dat does not match TOP parameter overrides.");
        $display("        DAT : BASE=%0d INTER=%0d INTRA=%0d",
                 base_max, inter_max, intra_max);
        $display("        DUT : BASE=%0d INTER=%0d INTRA=%0d",
                 DUT_BASE_MAX, DUT_INTER_MAX, DUT_INTRA_MAX);
        $display("        Current TOP cannot receive $readmemh values at runtime");
        $display("        because these thresholds are parameters.");
        $finish;
    end

    // Basic threshold sanity check.
    if (!((base_max <= inter_max) && (inter_max <= intra_max))) begin
        $display("[ERROR] Invalid constraint order: BASE=%0d INTER=%0d INTRA=%0d",
                 base_max, inter_max, intra_max);
        $finish;
    end

    // Reset.
    repeat (4) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    // Direct FICAM-FI input. No bitmap conversion/statistics are used.
    load_46_faults_directly_into_ficam();

    // Start evaluation after FICAM loading is complete.
    @(negedge clk);
    Eval_Start = 1'b1;
    @(negedge clk);
    Eval_Start = 1'b0;

    // TOP does not expose Eval_Done as a port, but cu_eval_done exists inside
    // the current TOP, so the TB waits for it hierarchically.
    timeout = 0;
    while (!dut.cu_eval_done && (timeout < 2000000)) begin
        @(negedge clk);
        timeout = timeout + 1;
    end

    if (timeout >= 2000000) begin
        $display("[ERROR] Evaluation timeout.");
        $finish;
    end

    // Move away from the completion edge before printing Remap_Mode.
    @(negedge clk);

    // Sanity-check that the control unit counted exactly the valid FI slots
    // loaded by this testbench. This prints only on mismatch.
    if (dut_fault_count !== tb_fault_count[TOTAL_COUNT_WIDTH-1:0]) begin
        $display("[ERROR] Fault count mismatch: TB=%0d DUT=%0d",
                 tb_fault_count, dut_fault_count);
        $finish;
    end

    print_constraint_summary();

    #20;
    $finish;
end

endmodule
