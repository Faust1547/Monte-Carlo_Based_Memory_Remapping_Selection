// =============================================================================
// Interface_With_SRAM_Command_Pipeline
// =============================================================================
// Correct page-level Monte-Carlo remapping hierarchy.
//
// ONE current page:
//   Global Remap Mode : one value for the whole page
//
// If Global Mode = INTER:
//   Remap Value       : one 3-bit Shift Mode per PBG
//                       32 values / page
//
// If Global Mode = INTRA:
//   Remap Value       : one 3-bit CW per {PBG, Bank, RG}
//                       32 x 8 x 8 = 2048 values / page
//
// Normal 14-bit SRAM address: {PBG[4:0], Bank[2:0], RG[2:0], Word[2:0]}
// Therefore normal data needs only: Data_Write_Packet = {Valid, Address[13:0], Data[7:0]}
// =============================================================================

module TOP #(
    parameter integer PPN_LENGTH         = 16,
    parameter integer PBG_LENGTH         = 5,
    parameter integer BANK_LENGTH        = 3,
    parameter integer RG_LENGTH          = 3,
    parameter integer FW_LENGTH          = 3,
    parameter integer FP_LENGTH          = 3,
    parameter integer CW_LENGTH          = 3,
    parameter integer FICAM_ENTRIES      = 64,
    parameter integer FI_PER_ENTRY       = 2,

    parameter integer SRAM_ADDR_LENGTH   = 14,
    parameter integer WORD_ADDR_LENGTH   = 3,
    parameter integer ES_WIDTH           = 20,
    parameter integer TOTAL_COUNT_WIDTH  = 24,
    parameter integer DATA_WIDTH         = 8,
    parameter integer WW_LENGTH          = DATA_WIDTH,
    parameter integer TWW_LENGTH         = DATA_WIDTH,

    parameter integer FI_SLOT_WIDTH      = 1 + FW_LENGTH + FP_LENGTH,
    parameter integer FICAM_LENGTH       = 1 + PPN_LENGTH + PBG_LENGTH
                                           + BANK_LENGTH + RG_LENGTH
                                           + FI_PER_ENTRY*FI_SLOT_WIDTH,
    parameter integer FICAM_INDEX_WIDTH  = (FICAM_ENTRIES <= 2) ? 1 : $clog2(FICAM_ENTRIES),
    parameter integer FICAM_COUNT_WIDTH  = (FICAM_ENTRIES <= 1) ? 1 : $clog2(FICAM_ENTRIES + 1),
    parameter integer DATA_PACKET_LENGTH = 1 + SRAM_ADDR_LENGTH + DATA_WIDTH,

    parameter [TOTAL_COUNT_WIDTH-1:0] BASE_MAX_FAULTS  = {TOTAL_COUNT_WIDTH{1'b0}},
    parameter [TOTAL_COUNT_WIDTH-1:0] INTER_MAX_FAULTS = {TOTAL_COUNT_WIDTH{1'b0}},
    parameter [TOTAL_COUNT_WIDTH-1:0] INTRA_MAX_FAULTS = {TOTAL_COUNT_WIDTH{1'b0}}
)(
    input  wire                              clk,
    input  wire                              rst_n,

    input  wire                              Group_Is_Exp,
    input  wire [FICAM_LENGTH-1:0]           FICAM_Entry,
    input  wire                              Eval_Start,

    input  wire [DATA_PACKET_LENGTH-1:0]      Data_Write_Packet,
    output wire                              Data_Ready,

    // 00=None/Base, 01=Inter, 10=Intra, 11=Unrepairable.
    output wire [1:0]                        Remap_Mode
);

localparam [1:0] REMAP_NONE         = 2'b00;
localparam [1:0] REMAP_INTER_BANK   = 2'b01;
localparam [1:0] REMAP_INTRA_BANK   = 2'b10;
localparam [1:0] REMAP_UNREPAIRABLE = 2'b11;

localparam integer INTRA_FIFO_DEPTH = 8;
localparam integer INTRA_FIFO_PTR_WIDTH =
    (INTRA_FIFO_DEPTH <= 2) ? 1 : $clog2(INTRA_FIFO_DEPTH);
localparam integer INTRA_FIFO_COUNT_WIDTH =
    $clog2(INTRA_FIFO_DEPTH + 1);

localparam [2:0] CMD_TAG_NONE         = 3'd0;
localparam [2:0] CMD_TAG_DIRECT_WRITE = 3'd1;
localparam [2:0] CMD_TAG_INTRA_WRITE  = 3'd2;

// -----------------------------------------------------------------------------
// Evaluation phase
// -----------------------------------------------------------------------------
reg  evaluation_phase;
wire eval_start_accept;
wire cu_busy;
wire cu_eval_done;

assign eval_start_accept = Eval_Start && !evaluation_phase && !cu_busy;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        evaluation_phase <= 1'b0;
    else if (eval_start_accept)
        evaluation_phase <= 1'b1;
end

// -----------------------------------------------------------------------------
// FICAM
// -----------------------------------------------------------------------------
wire                                  ficam_full;
wire [FICAM_COUNT_WIDTH-1:0]          ficam_entry_count;
wire                                  eval_read_enable;
wire [FICAM_INDEX_WIDTH-1:0]          eval_read_index;
wire                                  eval_read_valid;
wire [PBG_LENGTH-1:0]                 eval_read_pbg;
wire [BANK_LENGTH-1:0]                eval_read_bank;
wire [RG_LENGTH-1:0]                  eval_read_rg;
wire [FI_PER_ENTRY-1:0]               eval_read_fi_valid;
wire [FI_PER_ENTRY*FP_LENGTH-1:0]     eval_read_fi_fp;
wire                                  eval_read_is_exp;

FICAM #(
    .PPN_LENGTH     (PPN_LENGTH),
    .PBG_LENGTH     (PBG_LENGTH),
    .BANK_LENGTH    (BANK_LENGTH),
    .RG_LENGTH      (RG_LENGTH),
    .FW_LENGTH      (FW_LENGTH),
    .FP_LENGTH      (FP_LENGTH),
    .FICAM_ENTRIES  (FICAM_ENTRIES),
    .FI_PER_ENTRY   (FI_PER_ENTRY),
    .FICAM_LENGTH   (FICAM_LENGTH),
    .INDEX_WIDTH    (FICAM_INDEX_WIDTH),
    .COUNT_WIDTH    (FICAM_COUNT_WIDTH)
) u_ficam (
    .clk                 (clk),
    .rst_n               (rst_n),
    .Group_Is_Exp        (Group_Is_Exp),
    .FICAM_Entry         (FICAM_Entry),
    .Load_Lock           (evaluation_phase || eval_start_accept),
    .FICAM_Full          (ficam_full),
    .Entry_Count_Out     (ficam_entry_count),

    .Eval_Read_Enable    (eval_read_enable),
    .Eval_Read_Index     (eval_read_index),
    .Eval_Read_Valid     (eval_read_valid),
    .Eval_Read_PBG       (eval_read_pbg),
    .Eval_Read_Bank      (eval_read_bank),
    .Eval_Read_RG        (eval_read_rg),
    .Eval_Read_FI_Valid  (eval_read_fi_valid),
    .Eval_Read_FI_FP     (eval_read_fi_fp),
    .Eval_Read_Is_Exp    (eval_read_is_exp)
);

// -----------------------------------------------------------------------------
// Remap information memory
// -----------------------------------------------------------------------------
wire                              inter_write_enable;
wire [PBG_LENGTH-1:0]             inter_write_pbg;
wire [CW_LENGTH-1:0]              inter_shift_in;

wire                              intra_write_enable;
wire [PBG_LENGTH-1:0]             intra_write_pbg;
wire [BANK_LENGTH-1:0]            intra_write_bank;
wire [RG_LENGTH-1:0]              intra_write_rg;
wire [CW_LENGTH-1:0]              intra_cw_in;

// Normal address decode supplies remap-memory read address.
localparam integer D_DATA_LSB  = 0;
localparam integer D_ADDR_LSB  = D_DATA_LSB + DATA_WIDTH;
localparam integer D_VALID_BIT = D_ADDR_LSB + SRAM_ADDR_LENGTH;

wire [DATA_WIDTH-1:0]             data_in;
wire [SRAM_ADDR_LENGTH-1:0]       data_addr;
wire                              data_valid;

wire [WORD_ADDR_LENGTH-1:0]       normal_word;
wire [RG_LENGTH-1:0]              normal_rg;
wire [BANK_LENGTH-1:0]            normal_bank;
wire [PBG_LENGTH-1:0]             normal_pbg;

assign data_in    = Data_Write_Packet[D_DATA_LSB +: DATA_WIDTH];
assign data_addr  = Data_Write_Packet[D_ADDR_LSB +: SRAM_ADDR_LENGTH];
assign data_valid = Data_Write_Packet[D_VALID_BIT];

assign normal_word = data_addr[WORD_ADDR_LENGTH-1:0];
assign normal_rg   = data_addr[WORD_ADDR_LENGTH +: RG_LENGTH];
assign normal_bank = data_addr[WORD_ADDR_LENGTH+RG_LENGTH +: BANK_LENGTH];
assign normal_pbg  = data_addr[SRAM_ADDR_LENGTH-1 -: PBG_LENGTH];

wire [CW_LENGTH-1:0] selected_inter_shift;
wire [CW_LENGTH-1:0] selected_intra_cw;

Remap_Info_Memory_Page #(
    .PBG_LENGTH    (PBG_LENGTH),
    .BANK_LENGTH   (BANK_LENGTH),
    .RG_LENGTH     (RG_LENGTH),
    .CONTROL_WIDTH (CW_LENGTH)
) u_remap_info (
    .clk                (clk),

    .Inter_Write_Enable (inter_write_enable),
    .Inter_Write_PBG    (inter_write_pbg),
    .Inter_Shift_In     (inter_shift_in),

    .Intra_Write_Enable (intra_write_enable),
    .Intra_Write_PBG    (intra_write_pbg),
    .Intra_Write_Bank   (intra_write_bank),
    .Intra_Write_RG     (intra_write_rg),
    .Intra_CW_In        (intra_cw_in),

    .Read_PBG           (normal_pbg),
    .Read_Bank          (normal_bank),
    .Read_RG            (normal_rg),
    .Inter_Shift_Out    (selected_inter_shift),
    .Intra_CW_Out       (selected_intra_cw)
);

// -----------------------------------------------------------------------------
// Monte-Carlo threshold CU
// -----------------------------------------------------------------------------
MC_Control_Unit_PBG #(
    .PBG_LENGTH        (PBG_LENGTH),
    .BANK_LENGTH       (BANK_LENGTH),
    .RG_LENGTH         (RG_LENGTH),
    .FP_LENGTH         (FP_LENGTH),
    .CW_LENGTH         (CW_LENGTH),
    .ES_WIDTH          (ES_WIDTH),
    .TOTAL_COUNT_WIDTH (TOTAL_COUNT_WIDTH),
    .FI_PER_ENTRY      (FI_PER_ENTRY),
    .FICAM_ENTRIES     (FICAM_ENTRIES),
    .INDEX_WIDTH       (FICAM_INDEX_WIDTH),
    .COUNT_WIDTH       (FICAM_COUNT_WIDTH),
    .BASE_MAX_FAULTS   (BASE_MAX_FAULTS),
    .INTER_MAX_FAULTS  (INTER_MAX_FAULTS),
    .INTRA_MAX_FAULTS  (INTRA_MAX_FAULTS)
) u_mc_control_unit (
    .clk                 (clk),
    .rst_n               (rst_n),
    .Eval_Start          (eval_start_accept),
    .Eval_Entry_Count    (ficam_entry_count),

    .Eval_Read_Enable    (eval_read_enable),
    .Eval_Read_Index     (eval_read_index),
    .Eval_Read_Valid     (eval_read_valid),
    .Eval_Read_PBG       (eval_read_pbg),
    .Eval_Read_Bank      (eval_read_bank),
    .Eval_Read_RG        (eval_read_rg),
    .Eval_Read_FI_Valid  (eval_read_fi_valid),
    .Eval_Read_FI_FP     (eval_read_fi_fp),
    .Eval_Read_Is_Exp    (eval_read_is_exp),

    .Inter_Write_Enable  (inter_write_enable),
    .Inter_Write_PBG     (inter_write_pbg),
    .Inter_Shift_Out     (inter_shift_in),

    .Intra_Write_Enable  (intra_write_enable),
    .Intra_Write_PBG     (intra_write_pbg),
    .Intra_Write_Bank    (intra_write_bank),
    .Intra_Write_RG      (intra_write_rg),
    .Intra_CW_Out        (intra_cw_in),

    .Remap_Mode          (Remap_Mode),
    .Busy                (cu_busy),
    .Eval_Done           (cu_eval_done)
);

wire normal_path_enable;
assign normal_path_enable =
    evaluation_phase && !cu_busy &&
    (Remap_Mode != REMAP_UNREPAIRABLE);

// -----------------------------------------------------------------------------
// Inter data shifter: one selected shift for the packet's PBG.
// -----------------------------------------------------------------------------
wire [DATA_WIDTH-1:0] shifted_inter_data;
wire [DATA_WIDTH-1:0] unused_unshifted_data;

Data_Shifter #(
    .SM_LENGTH   (CW_LENGTH),
    .DATA_LENGTH (DATA_WIDTH)
) u_inter_shifter (
    .Shift_Mode        (selected_inter_shift),
    .Switched_Data_In  (data_in),
    .Shifted_Data_In   ({DATA_WIDTH{1'b0}}),
    .Switched_Data_Out (unused_unshifted_data),
    .Shifted_Data_Out  (shifted_inter_data)
);

// -----------------------------------------------------------------------------
// Intra 8x8 front-end: one CW selected by {PBG,Bank,RG}.
// -----------------------------------------------------------------------------
localparam [1:0] INTRA_IDLE  = 2'd0;
localparam [1:0] INTRA_FIRST = 2'd1;
localparam [1:0] INTRA_LOAD  = 2'd2;
localparam [1:0] INTRA_DRAIN = 2'd3;

reg [1:0] intra_state;
reg [DATA_WIDTH-1:0] first_ww;
reg [3:0] intra_input_count;
reg [WORD_ADDR_LENGTH-1:0] intra_output_count;
reg [CW_LENGTH-1:0] intra_cw_latched;
reg [SRAM_ADDR_LENGTH-WORD_ADDR_LENGTH-1:0] intra_addr_base;

wire trans_start;
wire trans_input_ready;
wire trans_output_valid;
wire trans_busy;
wire trans_done;
wire [TWW_LENGTH-1:0] transposed_data;
wire [WW_LENGTH-1:0] unused_ww_out;

wire feed_cached_first;
wire feed_external_ww;
wire trans_data_valid;
wire [WW_LENGTH-1:0] trans_ww_in;

assign feed_cached_first =
    (intra_state == INTRA_FIRST) && trans_input_ready;

assign feed_external_ww =
    (intra_state == INTRA_LOAD) &&
    trans_input_ready && normal_path_enable && data_valid;

assign trans_data_valid = feed_cached_first || feed_external_ww;
assign trans_ww_in      = feed_cached_first ? first_ww : data_in;

Transposer #(
    .WORD_COUNT  (8),
    .WW_LENGTH   (WW_LENGTH),
    .TWW_LENGTH  (TWW_LENGTH)
) u_transposer (
    .clk            (clk),
    .rst_n          (rst_n),
    .Start          (trans_start),
    .Direction      (1'b1),
    .Data_In_Valid  (trans_data_valid),
    .Input_Ready    (trans_input_ready),
    .WW_Data_In     (trans_ww_in),
    .TWW_Data_In    ({TWW_LENGTH{1'b0}}),
    .WW_Data_Out    (unused_ww_out),
    .TWW_Data_Out   (transposed_data),
    .Data_Out_Valid (trans_output_valid),
    .Busy           (trans_busy),
    .Done           (trans_done)
);

wire [WORD_ADDR_LENGTH-1:0] intra_physical_word;

Address_Remapper #(
    .WORD_ADDR_LENGTH (WORD_ADDR_LENGTH),
    .CW_LENGTH        (CW_LENGTH)
) u_intra_address_remapper (
    .Remap_Enable       (1'b1),
    .Control_Word       (intra_cw_latched),
    .Logical_Word_Addr  (intra_output_count),
    .Physical_Word_Addr (intra_physical_word)
);

// -----------------------------------------------------------------------------
// SRAM command pipeline
// -----------------------------------------------------------------------------
wire                        sram_command_ready;
wire                        sram_command_accepted;
wire                        sram_command_busy;
wire                        sram_write_done;
wire                        sram_read_data_valid;
wire [7:0]                  sram_read_data;
wire [2:0]                  sram_completion_tag;

reg                         sram_command_valid;
reg                         sram_command_write;
reg [13:0]                  sram_command_address;
reg [7:0]                   sram_command_write_data;
reg [2:0]                   sram_command_tag;

wire                        sram_cen_internal;
wire                        sram_wen_internal;
wire [13:0]                 sram_address_internal;
wire [7:0]                  sram_data_in_internal;
wire [7:0]                  sram_data_out_internal;
wire [2:0]                  sram_ema_internal;
wire [2:0]                  sram_macro_tag_internal;
wire [1:0]                  sram_pipeline_state_internal;

// -----------------------------------------------------------------------------
// Intra output FIFO
// -----------------------------------------------------------------------------
reg [13:0] intra_addr_fifo [0:INTRA_FIFO_DEPTH-1];
reg [7:0]  intra_data_fifo [0:INTRA_FIFO_DEPTH-1];

reg [INTRA_FIFO_PTR_WIDTH-1:0]   intra_fifo_write_ptr;
reg [INTRA_FIFO_PTR_WIDTH-1:0]   intra_fifo_read_ptr;
reg [INTRA_FIFO_COUNT_WIDTH-1:0] intra_fifo_count;

wire intra_fifo_empty;
wire intra_fifo_full;
wire intra_fifo_push;
wire intra_fifo_pop;
wire intra_fifo_push_accept;

assign intra_fifo_empty = (intra_fifo_count == 0);
assign intra_fifo_full  = (intra_fifo_count == INTRA_FIFO_DEPTH);

assign Data_Ready =
    normal_path_enable &&
    (
        ((intra_state == INTRA_IDLE) &&
         intra_fifo_empty && sram_command_ready)
        ||
        ((intra_state == INTRA_LOAD) && trans_input_ready)
    );

assign trans_start =
    (Remap_Mode == REMAP_INTRA_BANK) &&
    (intra_state == INTRA_IDLE) &&
    Data_Ready && data_valid;

wire [13:0] intra_generated_addr;
assign intra_generated_addr =
    {intra_addr_base, intra_physical_word};

assign intra_fifo_push =
    (intra_state == INTRA_DRAIN) && trans_output_valid;

assign intra_fifo_pop =
    sram_command_accepted &&
    (sram_command_tag == CMD_TAG_INTRA_WRITE);

assign intra_fifo_push_accept =
    intra_fifo_push && (!intra_fifo_full || intra_fifo_pop);

// -----------------------------------------------------------------------------
// Base / Inter direct write
// -----------------------------------------------------------------------------
wire direct_write_request;
wire [7:0] direct_write_data;

assign direct_write_request =
    normal_path_enable &&
    (Remap_Mode != REMAP_INTRA_BANK) &&
    (intra_state == INTRA_IDLE) &&
    intra_fifo_empty &&
    data_valid;

assign direct_write_data =
    (Remap_Mode == REMAP_INTER_BANK)
    ? shifted_inter_data[7:0]
    : data_in[7:0];

// Buffered Intra output has priority.
always @(*) begin
    sram_command_valid      = 1'b0;
    sram_command_write      = 1'b1;
    sram_command_address    = 14'b0;
    sram_command_write_data = 8'b0;
    sram_command_tag        = CMD_TAG_NONE;

    if (!intra_fifo_empty) begin
        sram_command_valid      = 1'b1;
        sram_command_address    = intra_addr_fifo[intra_fifo_read_ptr];
        sram_command_write_data = intra_data_fifo[intra_fifo_read_ptr];
        sram_command_tag        = CMD_TAG_INTRA_WRITE;
    end
    else if (direct_write_request) begin
        sram_command_valid      = 1'b1;
        sram_command_address    = data_addr[13:0];
        sram_command_write_data = direct_write_data;
        sram_command_tag        = CMD_TAG_DIRECT_WRITE;
    end
end

// -----------------------------------------------------------------------------
// Intra state / FIFO
// -----------------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        intra_state          <= INTRA_IDLE;
        first_ww             <= {DATA_WIDTH{1'b0}};
        intra_input_count    <= 4'd0;
        intra_output_count   <= {WORD_ADDR_LENGTH{1'b0}};
        intra_cw_latched     <= {CW_LENGTH{1'b0}};
        intra_addr_base      <= {(SRAM_ADDR_LENGTH-WORD_ADDR_LENGTH){1'b0}};
        intra_fifo_write_ptr <= {INTRA_FIFO_PTR_WIDTH{1'b0}};
        intra_fifo_read_ptr  <= {INTRA_FIFO_PTR_WIDTH{1'b0}};
        intra_fifo_count     <= {INTRA_FIFO_COUNT_WIDTH{1'b0}};
    end
    else begin
        case (intra_state)
            INTRA_IDLE: begin
                intra_input_count  <= 4'd0;
                intra_output_count <= {WORD_ADDR_LENGTH{1'b0}};

                if (trans_start) begin
                    first_ww         <= data_in;
                    intra_cw_latched <= selected_intra_cw;
                    intra_addr_base  <=
                        data_addr[SRAM_ADDR_LENGTH-1:WORD_ADDR_LENGTH];
                    intra_state      <= INTRA_FIRST;
                end
            end

            INTRA_FIRST: begin
                if (feed_cached_first) begin
                    intra_input_count <= 4'd1;
                    intra_state       <= INTRA_LOAD;
                end
            end

            INTRA_LOAD: begin
                if (feed_external_ww) begin
                    if (intra_input_count == 4'd7) begin
                        intra_input_count <= 4'd8;
                        intra_state       <= INTRA_DRAIN;
                    end
                    else begin
                        intra_input_count <= intra_input_count + 1'b1;
                    end
                end
            end

            INTRA_DRAIN: begin
                if (trans_output_valid) begin
                    if (intra_output_count == 3'd7)
                        intra_output_count <= {WORD_ADDR_LENGTH{1'b0}};
                    else
                        intra_output_count <= intra_output_count + 1'b1;
                end

                if (trans_done)
                    intra_state <= INTRA_IDLE;
            end

            default: intra_state <= INTRA_IDLE;
        endcase

        if (intra_fifo_push_accept) begin
            intra_addr_fifo[intra_fifo_write_ptr] <= intra_generated_addr;
            intra_data_fifo[intra_fifo_write_ptr] <= transposed_data[7:0];
            intra_fifo_write_ptr <= intra_fifo_write_ptr + 1'b1;
        end

        if (intra_fifo_pop)
            intra_fifo_read_ptr <= intra_fifo_read_ptr + 1'b1;

        case ({intra_fifo_push_accept, intra_fifo_pop})
            2'b10: intra_fifo_count <= intra_fifo_count + 1'b1;
            2'b01: intra_fifo_count <= intra_fifo_count - 1'b1;
            default: intra_fifo_count <= intra_fifo_count;
        endcase
    end
end

SRAM_SP_ADV_Command_Pipeline #(
    .TAG_WIDTH  (3),
    .EMA_NORMAL (3'b000)
) u_sram_command_pipeline (
    .clk                (clk),
    .rst_n              (rst_n),
    .Command_Valid      (sram_command_valid),
    .Command_Write      (sram_command_write),
    .Command_Address    (sram_command_address),
    .Command_Write_Data (sram_command_write_data),
    .Command_Tag        (sram_command_tag),

    .Command_Ready      (sram_command_ready),
    .Command_Accepted   (sram_command_accepted),
    .Busy               (sram_command_busy),
    .Write_Done         (sram_write_done),
    .Read_Data_Valid    (sram_read_data_valid),
    .Read_Data          (sram_read_data),
    .Completion_Tag     (sram_completion_tag),

    .CEN_Debug          (sram_cen_internal),
    .WEN_Debug          (sram_wen_internal),
    .Address_Debug      (sram_address_internal),
    .Data_In_Debug      (sram_data_in_internal),
    .Data_Out_Debug     (sram_data_out_internal),
    .EMA_Debug          (sram_ema_internal),
    .Macro_Tag_Debug    (sram_macro_tag_internal),
    .State_Debug        (sram_pipeline_state_internal)
);

wire _unused_internal_status;
assign _unused_internal_status = &{
    1'b0,
    ficam_full,
    cu_eval_done,
    trans_busy,
    intra_fifo_full,
    sram_command_busy,
    sram_write_done,
    sram_read_data_valid,
    sram_read_data,
    sram_completion_tag,
    sram_cen_internal,
    sram_wen_internal,
    sram_address_internal,
    sram_data_in_internal,
    sram_data_out_internal,
    sram_ema_internal,
    sram_macro_tag_internal,
    sram_pipeline_state_internal,
    normal_word
};

`ifndef SYNTHESIS
initial begin
    if (SRAM_ADDR_LENGTH != (PBG_LENGTH+BANK_LENGTH+RG_LENGTH+WORD_ADDR_LENGTH))
        $error("SRAM address partition must be {PBG,Bank,RG,Word}");
    if (SRAM_ADDR_LENGTH != 14)
        $error("Current SRAM macro requires 14-bit address");
    if (DATA_WIDTH != 8 || WW_LENGTH != 8 || TWW_LENGTH != 8)
        $error("Current datapath is specialized for 8-bit data");
end
`endif

endmodule


