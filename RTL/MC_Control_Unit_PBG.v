module MC_Control_Unit_PBG #(
    parameter integer PBG_LENGTH         = 5,
    parameter integer BANK_LENGTH        = 3,
    parameter integer RG_LENGTH          = 3,
    parameter integer FP_LENGTH          = 3,
    parameter integer CW_LENGTH          = 3,
    parameter integer ES_WIDTH           = 20,
    parameter integer TOTAL_COUNT_WIDTH  = 24,
    parameter integer FI_PER_ENTRY       = 2,
    parameter integer FICAM_ENTRIES      = 64,
    parameter integer CANDIDATES         = 8,
    parameter integer INDEX_WIDTH        = (FICAM_ENTRIES <= 2) ? 1 : $clog2(FICAM_ENTRIES),
    parameter integer COUNT_WIDTH        = (FICAM_ENTRIES <= 1) ? 1 : $clog2(FICAM_ENTRIES + 1),

    parameter [TOTAL_COUNT_WIDTH-1:0] BASE_MAX_FAULTS  = {TOTAL_COUNT_WIDTH{1'b0}},
    parameter [TOTAL_COUNT_WIDTH-1:0] INTER_MAX_FAULTS = {TOTAL_COUNT_WIDTH{1'b0}},
    parameter [TOTAL_COUNT_WIDTH-1:0] INTRA_MAX_FAULTS = {TOTAL_COUNT_WIDTH{1'b0}}
)(
    input  wire                                  clk,
    input  wire                                  rst_n,
    input  wire                                  Eval_Start,
    input  wire [COUNT_WIDTH-1:0]                Eval_Entry_Count,

    output wire                                  Eval_Read_Enable,
    output wire [INDEX_WIDTH-1:0]                Eval_Read_Index,
    input  wire                                  Eval_Read_Valid,
    input  wire [PBG_LENGTH-1:0]                 Eval_Read_PBG,
    input  wire [BANK_LENGTH-1:0]                Eval_Read_Bank,
    input  wire [RG_LENGTH-1:0]                  Eval_Read_RG,
    input  wire [FI_PER_ENTRY-1:0]               Eval_Read_FI_Valid,
    input  wire [FI_PER_ENTRY*FP_LENGTH-1:0]     Eval_Read_FI_FP,
    input  wire                                  Eval_Read_Is_Exp,

    output wire                                  Inter_Write_Enable,
    output wire [PBG_LENGTH-1:0]                 Inter_Write_PBG,
    output wire [CW_LENGTH-1:0]                  Inter_Shift_Out,

    output wire                                  Intra_Write_Enable,
    output wire [PBG_LENGTH-1:0]                 Intra_Write_PBG,
    output wire [BANK_LENGTH-1:0]                Intra_Write_Bank,
    output wire [RG_LENGTH-1:0]                  Intra_Write_RG,
    output wire [CW_LENGTH-1:0]                  Intra_CW_Out,

    output reg  [1:0]                            Remap_Mode,
    output wire                                  Busy,
    output wire                                  Eval_Done
);

localparam [1:0] MODE_BASE         = 2'b00;
localparam [1:0] MODE_INTER        = 2'b01;
localparam [1:0] MODE_INTRA        = 2'b10;
localparam [1:0] MODE_UNREPAIRABLE = 2'b11;

localparam [3:0] ST_IDLE          = 4'd0;
localparam [3:0] ST_COUNT         = 4'd1;
localparam [3:0] ST_MODE_SELECT   = 4'd2;
localparam [3:0] ST_INTER_SCAN    = 4'd3;
localparam [3:0] ST_INTER_CAND    = 4'd4;
localparam [3:0] ST_INTER_WRITE   = 4'd5;
localparam [3:0] ST_INTRA_SCAN    = 4'd6;
localparam [3:0] ST_INTRA_CAND    = 4'd7;
localparam [3:0] ST_INTRA_WRITE   = 4'd8;
localparam [3:0] ST_DONE          = 4'd9;

localparam [CW_LENGTH-1:0] LAST_CAND = CANDIDATES-1;
localparam [PBG_LENGTH-1:0] LAST_PBG = (1 << PBG_LENGTH)-1;
localparam [BANK_LENGTH-1:0] LAST_BANK = (1 << BANK_LENGTH)-1;
localparam [RG_LENGTH-1:0] LAST_RG = (1 << RG_LENGTH)-1;
localparam [ES_WIDTH-1:0] ES_MAX = {ES_WIDTH{1'b1}};

reg [3:0] state;
reg [COUNT_WIDTH-1:0] eval_count_latched;
reg [INDEX_WIDTH-1:0] scan_index;
reg [TOTAL_COUNT_WIDTH-1:0] total_fault_count;

reg [PBG_LENGTH-1:0] current_pbg;
reg [BANK_LENGTH-1:0] current_bank;
reg [RG_LENGTH-1:0] current_rg;
reg [CW_LENGTH-1:0] candidate_control;

reg [ES_WIDTH-1:0] candidate_es;
reg [ES_WIDTH-1:0] best_es;
reg [CW_LENGTH-1:0] best_control;
reg [CW_LENGTH-1:0] write_control;

assign Busy = (state != ST_IDLE) && (state != ST_DONE);
assign Eval_Done = (state == ST_DONE);

assign Eval_Read_Enable =
    ((state == ST_COUNT) ||
     (state == ST_INTER_SCAN) ||
     (state == ST_INTRA_SCAN)) &&
    ({{(COUNT_WIDTH-INDEX_WIDTH){1'b0}}, scan_index} < eval_count_latched);

assign Eval_Read_Index = scan_index;

assign Inter_Write_Enable = (state == ST_INTER_WRITE);
assign Inter_Write_PBG    = current_pbg;
assign Inter_Shift_Out    = write_control;

assign Intra_Write_Enable = (state == ST_INTRA_WRITE);
assign Intra_Write_PBG    = current_pbg;
assign Intra_Write_Bank   = current_bank;
assign Intra_Write_RG     = current_rg;
assign Intra_CW_Out       = write_control;

// Monte-Carlo ES profile retained.
function [ES_WIDTH-1:0] MC_Weight;
    input       Is_Exp;
    input [2:0] Logical_Index;
    begin
        if (Is_Exp) begin
            case (Logical_Index)
                3'd7: MC_Weight = 0;
                3'd6, 3'd5, 3'd4, 3'd3: MC_Weight = 7;
                3'd2, 3'd1: MC_Weight = 15;
                3'd0: MC_Weight = 3;
                default: MC_Weight = {ES_WIDTH{1'b0}};
            endcase
        end
        else begin
            case (Logical_Index)
                3'd7: MC_Weight = 31;
                3'd6: MC_Weight = 15;
                3'd5: MC_Weight = 7;
                3'd4: MC_Weight = 3;
                3'd3, 3'd2, 3'd1, 3'd0: MC_Weight = 1;
                default: MC_Weight = {ES_WIDTH{1'b0}};
            endcase
        end
    end
endfunction

reg [TOTAL_COUNT_WIDTH-1:0] current_entry_fault_count;
reg [ES_WIDTH-1:0] current_entry_es;
reg [2:0] logical_bank;
reg [2:0] logical_fp;
reg [FP_LENGTH-1:0] current_fp;
integer fi_count;
integer fi_es;

always @(*) begin
    current_entry_fault_count = {TOTAL_COUNT_WIDTH{1'b0}};
    for (fi_count = 0; fi_count < FI_PER_ENTRY; fi_count = fi_count + 1)
        if (Eval_Read_FI_Valid[fi_count])
            current_entry_fault_count = current_entry_fault_count + 1'b1;
end

always @(*) begin
    current_entry_es = {ES_WIDTH{1'b0}};
    logical_bank = Eval_Read_Bank + candidate_control;
    current_fp = {FP_LENGTH{1'b0}};
    logical_fp = 3'b000;

    if (Eval_Read_Valid) begin
        if ((state == ST_INTER_SCAN) &&
            (Eval_Read_PBG == current_pbg)) begin

            for (fi_es = 0; fi_es < FI_PER_ENTRY; fi_es = fi_es + 1)
                if (Eval_Read_FI_Valid[fi_es])
                    current_entry_es =
                        current_entry_es +
                        MC_Weight(Eval_Read_Is_Exp, logical_bank);
        end
        else if ((state == ST_INTRA_SCAN) &&
                 (Eval_Read_PBG  == current_pbg) &&
                 (Eval_Read_Bank == current_bank) &&
                 (Eval_Read_RG   == current_rg)) begin

            for (fi_es = 0; fi_es < FI_PER_ENTRY; fi_es = fi_es + 1) begin
                if (Eval_Read_FI_Valid[fi_es]) begin
                    current_fp =
                        Eval_Read_FI_FP[fi_es*FP_LENGTH +: FP_LENGTH];
                    logical_fp = current_fp ^ candidate_control;
                    current_entry_es =
                        current_entry_es +
                        MC_Weight(Eval_Read_Is_Exp, logical_fp);
                end
            end
        end
    end
end

wire [ES_WIDTH-1:0] candidate_es_after_entry;
assign candidate_es_after_entry = candidate_es + current_entry_es;

reg [1:0] selected_mode_comb;
always @(*) begin
    if ((BASE_MAX_FAULTS > INTER_MAX_FAULTS) ||
        (INTER_MAX_FAULTS > INTRA_MAX_FAULTS))
        selected_mode_comb = MODE_UNREPAIRABLE;
    else if (total_fault_count <= BASE_MAX_FAULTS)
        selected_mode_comb = MODE_BASE;
    else if (total_fault_count <= INTER_MAX_FAULTS)
        selected_mode_comb = MODE_INTER;
    else if (total_fault_count <= INTRA_MAX_FAULTS)
        selected_mode_comb = MODE_INTRA;
    else
        selected_mode_comb = MODE_UNREPAIRABLE;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state               <= ST_IDLE;
        eval_count_latched  <= {COUNT_WIDTH{1'b0}};
        scan_index          <= {INDEX_WIDTH{1'b0}};
        total_fault_count   <= {TOTAL_COUNT_WIDTH{1'b0}};
        current_pbg         <= {PBG_LENGTH{1'b0}};
        current_bank        <= {BANK_LENGTH{1'b0}};
        current_rg          <= {RG_LENGTH{1'b0}};
        candidate_control   <= {CW_LENGTH{1'b0}};
        candidate_es        <= {ES_WIDTH{1'b0}};
        best_es             <= ES_MAX;
        best_control        <= {CW_LENGTH{1'b0}};
        write_control       <= {CW_LENGTH{1'b0}};
        Remap_Mode          <= MODE_BASE;
    end
    else begin
        case (state)
            ST_IDLE: begin
                if (Eval_Start) begin
                    eval_count_latched <= Eval_Entry_Count;
                    scan_index         <= {INDEX_WIDTH{1'b0}};
                    total_fault_count  <= {TOTAL_COUNT_WIDTH{1'b0}};
                    Remap_Mode         <= MODE_BASE;

                    if (Eval_Entry_Count == {COUNT_WIDTH{1'b0}})
                        state <= ST_MODE_SELECT;
                    else
                        state <= ST_COUNT;
                end
            end

            // Count all FI slots once for global page mode selection.
            ST_COUNT: begin
                if (Eval_Read_Valid) begin
                    total_fault_count <=
                        total_fault_count + current_entry_fault_count;

                    if ({{(COUNT_WIDTH-INDEX_WIDTH){1'b0}}, scan_index} + 1'b1 >=
                        eval_count_latched) begin
                        scan_index <= {INDEX_WIDTH{1'b0}};
                        state <= ST_MODE_SELECT;
                    end
                    else begin
                        scan_index <= scan_index + 1'b1;
                    end
                end
            end

            ST_MODE_SELECT: begin
                Remap_Mode <= selected_mode_comb;

                current_pbg       <= {PBG_LENGTH{1'b0}};
                current_bank      <= {BANK_LENGTH{1'b0}};
                current_rg        <= {RG_LENGTH{1'b0}};
                candidate_control <= {CW_LENGTH{1'b0}};
                candidate_es      <= {ES_WIDTH{1'b0}};
                best_es           <= ES_MAX;
                best_control      <= {CW_LENGTH{1'b0}};
                scan_index        <= {INDEX_WIDTH{1'b0}};

                if (selected_mode_comb == MODE_INTER)
                    state <= ST_INTER_SCAN;
                else if (selected_mode_comb == MODE_INTRA)
                    state <= ST_INTRA_SCAN;
                else
                    state <= ST_DONE;
            end

            // -------------------------------------------------------------
            // INTER: one best shift for each PBG
            // -------------------------------------------------------------
            ST_INTER_SCAN: begin
                if (eval_count_latched == {COUNT_WIDTH{1'b0}}) begin
                    candidate_es <= {ES_WIDTH{1'b0}};
                    state <= ST_INTER_CAND;
                end
                else if (Eval_Read_Valid) begin
                    candidate_es <= candidate_es_after_entry;

                    if ({{(COUNT_WIDTH-INDEX_WIDTH){1'b0}}, scan_index} + 1'b1 >=
                        eval_count_latched) begin
                        scan_index <= {INDEX_WIDTH{1'b0}};
                        state <= ST_INTER_CAND;
                    end
                    else begin
                        scan_index <= scan_index + 1'b1;
                    end
                end
            end

            ST_INTER_CAND: begin
                if ((candidate_control == {CW_LENGTH{1'b0}}) ||
                    (candidate_es < best_es)) begin
                    best_es      <= candidate_es;
                    best_control <= candidate_control;
                end

                if (candidate_control == LAST_CAND) begin
                    if ((candidate_control == {CW_LENGTH{1'b0}}) ||
                        (candidate_es < best_es))
                        write_control <= candidate_control;
                    else
                        write_control <= best_control;

                    state <= ST_INTER_WRITE;
                end
                else begin
                    candidate_control <= candidate_control + 1'b1;
                    candidate_es      <= {ES_WIDTH{1'b0}};
                    scan_index        <= {INDEX_WIDTH{1'b0}};
                    state             <= ST_INTER_SCAN;
                end
            end

            ST_INTER_WRITE: begin
                if (current_pbg == LAST_PBG) begin
                    state <= ST_DONE;
                end
                else begin
                    current_pbg       <= current_pbg + 1'b1;
                    candidate_control <= {CW_LENGTH{1'b0}};
                    candidate_es      <= {ES_WIDTH{1'b0}};
                    best_es           <= ES_MAX;
                    best_control      <= {CW_LENGTH{1'b0}};
                    scan_index        <= {INDEX_WIDTH{1'b0}};
                    state             <= ST_INTER_SCAN;
                end
            end

            // -------------------------------------------------------------
            // INTRA: one best CW for each {PBG,Bank,RG}
            // -------------------------------------------------------------
            ST_INTRA_SCAN: begin
                if (eval_count_latched == {COUNT_WIDTH{1'b0}}) begin
                    candidate_es <= {ES_WIDTH{1'b0}};
                    state <= ST_INTRA_CAND;
                end
                else if (Eval_Read_Valid) begin
                    candidate_es <= candidate_es_after_entry;

                    if ({{(COUNT_WIDTH-INDEX_WIDTH){1'b0}}, scan_index} + 1'b1 >=
                        eval_count_latched) begin
                        scan_index <= {INDEX_WIDTH{1'b0}};
                        state <= ST_INTRA_CAND;
                    end
                    else begin
                        scan_index <= scan_index + 1'b1;
                    end
                end
            end

            ST_INTRA_CAND: begin
                if ((candidate_control == {CW_LENGTH{1'b0}}) ||
                    (candidate_es < best_es)) begin
                    best_es      <= candidate_es;
                    best_control <= candidate_control;
                end

                if (candidate_control == LAST_CAND) begin
                    if ((candidate_control == {CW_LENGTH{1'b0}}) ||
                        (candidate_es < best_es))
                        write_control <= candidate_control;
                    else
                        write_control <= best_control;

                    state <= ST_INTRA_WRITE;
                end
                else begin
                    candidate_control <= candidate_control + 1'b1;
                    candidate_es      <= {ES_WIDTH{1'b0}};
                    scan_index        <= {INDEX_WIDTH{1'b0}};
                    state             <= ST_INTRA_SCAN;
                end
            end

            ST_INTRA_WRITE: begin
                if ((current_pbg == LAST_PBG) &&
                    (current_bank == LAST_BANK) &&
                    (current_rg == LAST_RG)) begin
                    state <= ST_DONE;
                end
                else begin
                    if (current_rg == LAST_RG) begin
                        current_rg <= {RG_LENGTH{1'b0}};

                        if (current_bank == LAST_BANK) begin
                            current_bank <= {BANK_LENGTH{1'b0}};
                            current_pbg  <= current_pbg + 1'b1;
                        end
                        else begin
                            current_bank <= current_bank + 1'b1;
                        end
                    end
                    else begin
                        current_rg <= current_rg + 1'b1;
                    end

                    candidate_control <= {CW_LENGTH{1'b0}};
                    candidate_es      <= {ES_WIDTH{1'b0}};
                    best_es           <= ES_MAX;
                    best_control      <= {CW_LENGTH{1'b0}};
                    scan_index        <= {INDEX_WIDTH{1'b0}};
                    state             <= ST_INTRA_SCAN;
                end
            end

            ST_DONE: begin
                state <= ST_IDLE;
            end

            default: state <= ST_IDLE;
        endcase
    end
end

`ifndef SYNTHESIS
initial begin
    if (PBG_LENGTH != 5 || BANK_LENGTH != 3 ||
        RG_LENGTH != 3 || FP_LENGTH != 3 ||
        CW_LENGTH != 3 || CANDIDATES != 8)
        $error("MC_Control_Unit_PBG is specialized for 32 PBG / 8 Bank / 8 RG / 8 candidate architecture");
end
`endif

endmodule
