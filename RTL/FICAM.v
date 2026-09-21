// =============================================================================
// FICAM
// =============================================================================
// Direct packed fault-entry memory for ONE current page evaluation.
// Default depth is 4096 entries (12-bit index, 13-bit entry count).
//
// Default packed entry = 42 bits:
//   { Entry_Valid,
//     PPN[15:0],
//     PBG_ID[4:0],
//     Bank_ID[2:0],
//     RG_ID[2:0],
//     FI1_Valid, FI1_FW[2:0], FI1_FP[2:0],
//     FI0_Valid, FI0_FW[2:0], FI0_FP[2:0] }
//
// PBG_ID is explicit because remap-value granularity is:
//   Inter : one Shift Mode per PBG
//   Intra : one CW per {PBG, Bank, RG}
//
// Repeated {PPN,PBG,Bank,RG} tags are legal.  This allows more than
// FI_PER_ENTRY faults in one RG by using multiple packed FICAM entries.
// =============================================================================

module FICAM #(
    parameter integer PPN_LENGTH       = 16,
    parameter integer PBG_LENGTH       = 5,
    parameter integer BANK_LENGTH      = 3,
    parameter integer RG_LENGTH        = 3,
    parameter integer FW_LENGTH        = 3,
    parameter integer FP_LENGTH        = 3,
    parameter integer FICAM_ENTRIES    = 4096,
    parameter integer FI_PER_ENTRY     = 2,

    parameter integer FI_SLOT_WIDTH    = 1 + FW_LENGTH + FP_LENGTH,
    parameter integer FICAM_LENGTH     = 1 + PPN_LENGTH + PBG_LENGTH
                                         + BANK_LENGTH + RG_LENGTH
                                         + FI_PER_ENTRY*FI_SLOT_WIDTH,
    parameter integer INDEX_WIDTH      = (FICAM_ENTRIES <= 2) ? 1 : $clog2(FICAM_ENTRIES),
    parameter integer COUNT_WIDTH      = (FICAM_ENTRIES <= 1) ? 1 : $clog2(FICAM_ENTRIES + 1)
)(
    input  wire                                  clk,
    input  wire                                  rst_n,

    input  wire                                  Group_Is_Exp,
    input  wire [FICAM_LENGTH-1:0]               FICAM_Entry,
    input  wire                                  Load_Lock,

    output wire                                  FICAM_Full,
    output wire [COUNT_WIDTH-1:0]                Entry_Count_Out,

    input  wire                                  Eval_Read_Enable,
    input  wire [INDEX_WIDTH-1:0]                Eval_Read_Index,
    output reg                                   Eval_Read_Valid,
    output reg  [PBG_LENGTH-1:0]                 Eval_Read_PBG,
    output reg  [BANK_LENGTH-1:0]                Eval_Read_Bank,
    output reg  [RG_LENGTH-1:0]                  Eval_Read_RG,
    output reg  [FI_PER_ENTRY-1:0]               Eval_Read_FI_Valid,
    output reg  [FI_PER_ENTRY*FP_LENGTH-1:0]     Eval_Read_FI_FP,
    output reg                                   Eval_Read_Is_Exp
);

localparam integer RG_LSB     = FI_PER_ENTRY * FI_SLOT_WIDTH;
localparam integer BANK_LSB   = RG_LSB + RG_LENGTH;
localparam integer PBG_LSB    = BANK_LSB + BANK_LENGTH;
localparam integer PPN_LSB    = PBG_LSB + PBG_LENGTH;
localparam integer VALID_BIT  = PPN_LSB + PPN_LENGTH;

localparam integer FP_OFFSET  = 0;
localparam integer FV_OFFSET  = FP_LENGTH + FW_LENGTH;

reg [FICAM_LENGTH-1:0] Entry_Mem [0:FICAM_ENTRIES-1];
reg                    Entry_Is_Exp_Mem [0:FICAM_ENTRIES-1];
reg [COUNT_WIDTH-1:0]  entry_count;

integer eval_fi;

assign Entry_Count_Out = entry_count;
assign FICAM_Full      = (entry_count >= FICAM_ENTRIES);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // entry_count is the validity boundary; physical contents do not need reset.
        entry_count <= {COUNT_WIDTH{1'b0}};
    end
    else if (!Load_Lock && FICAM_Entry[VALID_BIT] && !FICAM_Full) begin
        Entry_Mem[entry_count[INDEX_WIDTH-1:0]]        <= FICAM_Entry;
        Entry_Is_Exp_Mem[entry_count[INDEX_WIDTH-1:0]] <= Group_Is_Exp;
        entry_count <= entry_count + 1'b1;
    end
end

always @(*) begin
    Eval_Read_Valid    = 1'b0;
    Eval_Read_PBG      = {PBG_LENGTH{1'b0}};
    Eval_Read_Bank     = {BANK_LENGTH{1'b0}};
    Eval_Read_RG       = {RG_LENGTH{1'b0}};
    Eval_Read_FI_Valid = {FI_PER_ENTRY{1'b0}};
    Eval_Read_FI_FP    = {(FI_PER_ENTRY*FP_LENGTH){1'b0}};
    Eval_Read_Is_Exp   = 1'b0;

    if (Eval_Read_Enable &&
        ({{(COUNT_WIDTH-INDEX_WIDTH){1'b0}}, Eval_Read_Index} < entry_count)) begin

        Eval_Read_Valid  = Entry_Mem[Eval_Read_Index][VALID_BIT];
        Eval_Read_PBG    = Entry_Mem[Eval_Read_Index][PBG_LSB  +: PBG_LENGTH];
        Eval_Read_Bank   = Entry_Mem[Eval_Read_Index][BANK_LSB +: BANK_LENGTH];
        Eval_Read_RG     = Entry_Mem[Eval_Read_Index][RG_LSB   +: RG_LENGTH];
        Eval_Read_Is_Exp = Entry_Is_Exp_Mem[Eval_Read_Index];

        for (eval_fi = 0; eval_fi < FI_PER_ENTRY; eval_fi = eval_fi + 1) begin
            Eval_Read_FI_Valid[eval_fi] =
                Entry_Mem[Eval_Read_Index][eval_fi*FI_SLOT_WIDTH + FV_OFFSET];

            Eval_Read_FI_FP[eval_fi*FP_LENGTH +: FP_LENGTH] =
                Entry_Mem[Eval_Read_Index][eval_fi*FI_SLOT_WIDTH + FP_OFFSET +: FP_LENGTH];
        end
    end
end

endmodule
