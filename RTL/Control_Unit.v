module Control_Unit #(
    parameter integer NUM_MODEL_PAGES       = 1,
    parameter integer PAGE_INDEX_WIDTH      = 1,
    parameter integer NUM_PBG_PER_PAGE      = 32,
    parameter integer PBG_ID_WIDTH          = 5,
    parameter integer TOTAL_OCCUPIED_PBGS   = NUM_MODEL_PAGES * NUM_PBG_PER_PAGE,
    parameter integer DIRECTORY_ADDR_WIDTH  = PAGE_INDEX_WIDTH + PBG_ID_WIDTH,

    parameter integer BANK_SIZE             = 8,
    parameter integer BANK_ID_WIDTH         = 3,
    parameter integer RG_SIZE               = 8,
    parameter integer RG_ID_WIDTH           = 3,
    parameter integer WORD_ID_WIDTH         = 3,
    parameter integer BIT_ID_WIDTH          = 3,
    parameter integer FW_LENGTH             = 3,

    parameter integer CONTROL_WIDTH         = 3,
    parameter integer ES_WIDTH              = 20,
    parameter integer PBG_COUNT_WIDTH       = 13,
    parameter integer TOTAL_COUNT_WIDTH     = 24,

    // EXP + four S&M PBGs for bounding-box size 4.
    parameter integer EXP_PBG_PERIOD        = 5,
    parameter integer FIELD_PHASE_WIDTH     = 3,

    parameter [FW_LENGTH-1:0] FAULT_SA0_CODE = 3'b001,
    parameter [FW_LENGTH-1:0] FAULT_SA1_CODE = 3'b010
)(
    input  wire                              clk,
    input  wire                              rst_n,

    // Start a fresh model deployment. CU clears both fault and remap storage.
    input  wire                              Start,

    // Software-generated integer thresholds for this model/accuracy target.
    input  wire [TOTAL_COUNT_WIDTH-1:0]      Base_Max_Faults,
    input  wire [TOTAL_COUNT_WIDTH-1:0]      Inter_Max_Faults,
    input  wire [TOTAL_COUNT_WIDTH-1:0]      Intra_Max_Faults,

    // -------------------------------------------------------------------------
    // Streaming 64-bit BIST bitmap input.
    // Input order must be:
    //   Model Page -> PBG -> Bank -> RG
    // The 64 bits are {8 words x 8 bits}; index = Word_ID*8 + Bit_ID.
    // Fault_Value_Bitmap_In: 0=SA0, 1=SA1 for each valid fault position.
    // -------------------------------------------------------------------------
    input  wire                              Fault_Bitmap_Valid,
    input  wire [63:0]                       Fault_Valid_Bitmap_In,
    input  wire [63:0]                       Fault_Value_Bitmap_In,
    input  wire [PAGE_INDEX_WIDTH-1:0]       Bitmap_Page_Index_In,
    input  wire [PBG_ID_WIDTH-1:0]           Bitmap_PBG_ID_In,
    input  wire [BANK_ID_WIDTH-1:0]          Bitmap_Bank_ID_In,
    input  wire [RG_ID_WIDTH-1:0]            Bitmap_RG_ID_In,
    output wire                              Fault_Bitmap_Ready,

    // -------------------------------------------------------------------------
    // Sparse FICAM write interface
    // -------------------------------------------------------------------------
    output wire                              FICAM_Clear_All,
    output wire                              FICAM_PBG_Begin,
    output wire [PAGE_INDEX_WIDTH-1:0]       FICAM_Write_Page_Index,
    output wire [PBG_ID_WIDTH-1:0]           FICAM_Write_PBG_ID,

    output wire                              FICAM_Fault_Write_Enable,
    output wire [BANK_ID_WIDTH-1:0]          FICAM_Fault_Bank_ID_In,
    output wire [RG_ID_WIDTH-1:0]            FICAM_Fault_RG_ID_In,
    output wire [WORD_ID_WIDTH-1:0]          FICAM_Fault_Word_ID_In,
    output wire [BIT_ID_WIDTH-1:0]           FICAM_Fault_Bit_ID_In,
    output wire [FW_LENGTH-1:0]              FICAM_Fault_Code_In,
    output wire                              FICAM_PBG_End,

    input  wire                              FICAM_Fault_Write_Ready,
    input  wire                              FICAM_Fault_Entry_Full,
    input  wire [TOTAL_COUNT_WIDTH-1:0]      FICAM_Total_Fault_Count,

    // -------------------------------------------------------------------------
    // Sparse FICAM search/read interface
    // -------------------------------------------------------------------------
    output wire                              FICAM_Directory_Search_Enable,
    output wire [PAGE_INDEX_WIDTH-1:0]       FICAM_Search_Page_Index,
    output wire [PBG_ID_WIDTH-1:0]           FICAM_Search_PBG_ID,
    input  wire                              FICAM_Directory_Match,
    input  wire [PBG_COUNT_WIDTH-1:0]        FICAM_Directory_Fault_Count,

    output wire                              FICAM_Fault_Read_Enable,
    output wire [PBG_COUNT_WIDTH-1:0]        FICAM_Fault_Offset_In,
    input  wire                              FICAM_Fault_Read_Valid,
    input  wire [BANK_ID_WIDTH-1:0]          FICAM_Fault_Bank_ID_Out,
    input  wire [RG_ID_WIDTH-1:0]            FICAM_Fault_RG_ID_Out,
    input  wire [WORD_ID_WIDTH-1:0]          FICAM_Fault_Word_ID_Out,
    input  wire [BIT_ID_WIDTH-1:0]           FICAM_Fault_Bit_ID_Out,
    input  wire [FW_LENGTH-1:0]              FICAM_Fault_Code_Out,

    output wire                              FICAM_Filter_Enable,
    output wire [BANK_ID_WIDTH-1:0]          FICAM_Filter_Bank_ID,
    output wire [RG_ID_WIDTH-1:0]            FICAM_Filter_RG_ID,
    input  wire                              FICAM_Fault_Bank_RG_Match,

    // -------------------------------------------------------------------------
    // Remap_Info_Memory write interface
    // -------------------------------------------------------------------------
    output wire                              Remap_Info_Clear,

    output wire                              Inter_Write_Enable,
    output wire [PAGE_INDEX_WIDTH-1:0]       Inter_Write_Page_Index,
    output wire [PBG_ID_WIDTH-1:0]           Inter_Write_PBG_ID,
    output wire [CONTROL_WIDTH-1:0]          Inter_Shift_In,

    output wire                              Intra_Write_Enable,
    output wire [PAGE_INDEX_WIDTH-1:0]       Intra_Write_Page_Index,
    output wire [PBG_ID_WIDTH-1:0]           Intra_Write_PBG_ID,
    output wire [BANK_ID_WIDTH-1:0]          Intra_Write_Bank_ID,
    output wire [RG_ID_WIDTH-1:0]            Intra_Write_RG_ID,
    output wire [CONTROL_WIDTH-1:0]          Intra_CW_In,

    // -------------------------------------------------------------------------
    // Status/configuration outputs
    // -------------------------------------------------------------------------
    output reg  [1:0]                        Global_Remap_Mode,
    output wire                              Busy,
    output wire                              Done,
    output wire                              Model_Fault_Scan_Done,
    output reg                               Input_Error,
    output reg                               Constraint_Error,
    output reg                               Storage_Overflow_Error,
    output reg                               Directory_Missing_Error,
    output reg                               Fault_Read_Error,
    output reg                               Unsupported_Fault_Code_Error,

    // Debug
    output wire [5:0]                        State_Debug,
    output wire [TOTAL_COUNT_WIDTH-1:0]      Fault_Count_Debug,
    output wire [PAGE_INDEX_WIDTH-1:0]       Current_Page_Debug,
    output wire [PBG_ID_WIDTH-1:0]           Current_PBG_Debug,
    output wire [BANK_ID_WIDTH-1:0]          Current_Bank_Debug,
    output wire [RG_ID_WIDTH-1:0]            Current_RG_Debug,
    output wire [CONTROL_WIDTH-1:0]          Candidate_Control_Debug,
    output wire [PBG_COUNT_WIDTH-1:0]        Fault_Offset_Debug,
    output wire [ES_WIDTH-1:0]               Candidate_ES_Debug,
    output wire [ES_WIDTH-1:0]               Best_ES_Debug,
    output wire                              Current_PBG_Is_Exp_Debug
);

// -----------------------------------------------------------------------------
// Global remap modes
// -----------------------------------------------------------------------------
localparam [1:0] MODE_BASE         = 2'b00;
localparam [1:0] MODE_INTER        = 2'b01;
localparam [1:0] MODE_INTRA        = 2'b10;
localparam [1:0] MODE_UNREPAIRABLE = 2'b11;

// -----------------------------------------------------------------------------
// FSM states
// -----------------------------------------------------------------------------
localparam [5:0] ST_IDLE             = 6'd0;
localparam [5:0] ST_CLEAR            = 6'd1;
localparam [5:0] ST_LOAD_WAIT        = 6'd2;
localparam [5:0] ST_SERIALIZE        = 6'd3;
localparam [5:0] ST_BITMAP_FINISH    = 6'd4;
localparam [5:0] ST_SCAN_COMPLETE    = 6'd5;
localparam [5:0] ST_MODE_SELECT      = 6'd6;
localparam [5:0] ST_PBG_LOOKUP       = 6'd7;
localparam [5:0] ST_INTER_CAND_INIT  = 6'd8;
localparam [5:0] ST_INTER_SCAN       = 6'd9;
localparam [5:0] ST_INTER_WRITE      = 6'd10;
localparam [5:0] ST_INTRA_GROUP_INIT = 6'd11;
localparam [5:0] ST_INTRA_CAND_INIT  = 6'd12;
localparam [5:0] ST_INTRA_SCAN       = 6'd13;
localparam [5:0] ST_INTRA_WRITE      = 6'd14;
localparam [5:0] ST_ADVANCE_PBG      = 6'd15;
localparam [5:0] ST_DONE             = 6'd16;

localparam [CONTROL_WIDTH-1:0] LAST_CONTROL  = BANK_SIZE - 1;
localparam [BANK_ID_WIDTH-1:0] LAST_BANK     = BANK_SIZE - 1;
localparam [RG_ID_WIDTH-1:0]   LAST_RG       = RG_SIZE - 1;
localparam [ES_WIDTH-1:0]      ES_MAX        = {ES_WIDTH{1'b1}};
localparam [DIRECTORY_ADDR_WIDTH-1:0] LAST_OCCUPIED_PBG = TOTAL_OCCUPIED_PBGS - 1;

reg [5:0] State;

// -----------------------------------------------------------------------------
// Latched thresholds and model-wide count
// -----------------------------------------------------------------------------
reg [TOTAL_COUNT_WIDTH-1:0] Base_Max_Faults_Reg;
reg [TOTAL_COUNT_WIDTH-1:0] Inter_Max_Faults_Reg;
reg [TOTAL_COUNT_WIDTH-1:0] Intra_Max_Faults_Reg;
reg [TOTAL_COUNT_WIDTH-1:0] Total_Fault_Count_Reg;

// -----------------------------------------------------------------------------
// Load-phase counters and pending bitmap serialization
// -----------------------------------------------------------------------------
reg [DIRECTORY_ADDR_WIDTH-1:0] Load_PBG_Linear;
reg [BANK_ID_WIDTH-1:0]        Load_Bank;
reg [RG_ID_WIDTH-1:0]          Load_RG;

reg [63:0] Pending_Fault_Bitmap;
reg [63:0] Pending_Fault_Value;
reg [BANK_ID_WIDTH-1:0] Latched_Bitmap_Bank;
reg [RG_ID_WIDTH-1:0]   Latched_Bitmap_RG;

wire [PAGE_INDEX_WIDTH-1:0] Expected_Load_Page;
wire [PBG_ID_WIDTH-1:0]     Expected_Load_PBG;
wire                         Bitmap_Address_Match;
wire                         Accept_Bitmap;
wire                         First_Bitmap_Of_PBG;
wire                         Last_Bitmap_Of_PBG;
wire                         Last_PBG_Of_Model;

assign Expected_Load_Page = Load_PBG_Linear[DIRECTORY_ADDR_WIDTH-1:PBG_ID_WIDTH];
assign Expected_Load_PBG  = Load_PBG_Linear[PBG_ID_WIDTH-1:0];

assign Bitmap_Address_Match =
    (Bitmap_Page_Index_In == Expected_Load_Page) &&
    (Bitmap_PBG_ID_In     == Expected_Load_PBG)  &&
    (Bitmap_Bank_ID_In    == Load_Bank)          &&
    (Bitmap_RG_ID_In      == Load_RG);

assign Fault_Bitmap_Ready = (State == ST_LOAD_WAIT) && !FICAM_Fault_Entry_Full;

assign Accept_Bitmap = Fault_Bitmap_Valid && Fault_Bitmap_Ready && Bitmap_Address_Match;

assign First_Bitmap_Of_PBG =
    (Load_Bank == {BANK_ID_WIDTH{1'b0}}) &&
    (Load_RG   == {RG_ID_WIDTH{1'b0}});

assign Last_Bitmap_Of_PBG = (Load_Bank == LAST_BANK) && (Load_RG == LAST_RG);

assign Last_PBG_Of_Model = (Load_PBG_Linear == LAST_OCCUPIED_PBG);

// Priority encoder over one 64-bit Bank/RG bitmap.
reg        Priority_Valid;
reg [5:0]  Priority_Index;
reg        Priority_Found;
integer    Priority_Scan;

always @(*) begin
    Priority_Valid = 1'b0;
    Priority_Index = 6'd0;
    Priority_Found = 1'b0;

    for (Priority_Scan = 0; Priority_Scan < 64;
         Priority_Scan = Priority_Scan + 1) begin
        if (!Priority_Found && Pending_Fault_Bitmap[Priority_Scan]) begin
            Priority_Valid = 1'b1;
            Priority_Index = Priority_Scan[5:0];
            Priority_Found = 1'b1;
        end
    end
end

wire [63:0] Priority_One_Hot;
wire [63:0] Pending_After_Accepted_Write;
wire        Accepted_Sparse_Write;

assign Priority_One_Hot = 64'b1 << Priority_Index;
assign Accepted_Sparse_Write =
    (State == ST_SERIALIZE) && Priority_Valid &&
    FICAM_Fault_Write_Ready && !FICAM_Fault_Entry_Full;
assign Pending_After_Accepted_Write =
    Pending_Fault_Bitmap & ~Priority_One_Hot;

// -----------------------------------------------------------------------------
// Search traversal
// -----------------------------------------------------------------------------
reg [DIRECTORY_ADDR_WIDTH-1:0] Search_PBG_Linear;
reg [FIELD_PHASE_WIDTH-1:0]    PBG_Field_Phase;
reg [PBG_COUNT_WIDTH-1:0]      Current_PBG_Fault_Count;
reg [BANK_ID_WIDTH-1:0]        Search_Bank;
reg [RG_ID_WIDTH-1:0]          Search_RG;
reg [PBG_COUNT_WIDTH-1:0]      Fault_Offset;

wire [PAGE_INDEX_WIDTH-1:0] Current_Search_Page;
wire [PBG_ID_WIDTH-1:0]     Current_Search_PBG;
wire                         Current_PBG_Is_Exp;
wire                         Last_Search_PBG;

assign Current_Search_Page = Search_PBG_Linear[DIRECTORY_ADDR_WIDTH-1:PBG_ID_WIDTH];
assign Current_Search_PBG = Search_PBG_Linear[PBG_ID_WIDTH-1:0];
assign Current_PBG_Is_Exp = (PBG_Field_Phase == {FIELD_PHASE_WIDTH{1'b0}});
assign Last_Search_PBG = (Search_PBG_Linear == LAST_OCCUPIED_PBG);

// -----------------------------------------------------------------------------
// Candidate search registers
// -----------------------------------------------------------------------------
reg [CONTROL_WIDTH-1:0] Candidate_Control;
reg [CONTROL_WIDTH-1:0] Best_Control;
reg [ES_WIDTH-1:0]      Candidate_ES_Accumulator;
reg [ES_WIDTH-1:0]      Best_ES_Reg;

// -----------------------------------------------------------------------------
// ES profile
// RTL bit indices are conventional byte indices: bit 7 is MSB, bit 0 is LSB.
// EXP bit 7 receives zero cost because masking is always active for EXP.
// -----------------------------------------------------------------------------
function [ES_WIDTH-1:0] Exp_Weight;
    input [2:0] Logical_Index;
    begin
        case (Logical_Index)
            3'd7: Exp_Weight = 0;   // Masked EXP MSB
            3'd6: Exp_Weight = 7;
            3'd5: Exp_Weight = 7;
            3'd4: Exp_Weight = 7;
            3'd3: Exp_Weight = 7;
            3'd2: Exp_Weight = 15;
            3'd1: Exp_Weight = 15;
            3'd0: Exp_Weight = 3;
            default: Exp_Weight = {ES_WIDTH{1'b0}};
        endcase
    end
endfunction

function [ES_WIDTH-1:0] SM_Weight;
    input [2:0] Logical_Index;
    begin
        case (Logical_Index)
            3'd7: SM_Weight = 31;
            3'd6: SM_Weight = 15;
            3'd5: SM_Weight = 7;
            3'd4: SM_Weight = 3;
            3'd3: SM_Weight = 1;
            3'd2: SM_Weight = 1;
            3'd1: SM_Weight = 1;
            3'd0: SM_Weight = 1;
            default: SM_Weight = {ES_WIDTH{1'b0}};
        endcase
    end
endfunction

reg [3:0] Inter_Logical_Bank_Temp;
reg [2:0] Inter_Logical_Bank;
reg [2:0] Intra_Logical_Bit;
reg [ES_WIDTH-1:0] Current_Entry_ES;

always @(*) begin
    Inter_Logical_Bank_Temp =
        FICAM_Fault_Bank_ID_Out + Candidate_Control;

    if (Inter_Logical_Bank_Temp >= BANK_SIZE)
        Inter_Logical_Bank = Inter_Logical_Bank_Temp - BANK_SIZE;
    else
        Inter_Logical_Bank = Inter_Logical_Bank_Temp[2:0];

    Intra_Logical_Bit = FICAM_Fault_Bit_ID_Out ^ Candidate_Control;

    Current_Entry_ES = {ES_WIDTH{1'b0}};

    if (FICAM_Fault_Read_Valid) begin
        if (State == ST_INTER_SCAN) begin
            if (Current_PBG_Is_Exp)
                Current_Entry_ES = Exp_Weight(Inter_Logical_Bank);
            else
                Current_Entry_ES = SM_Weight(Inter_Logical_Bank);
        end
        else if ((State == ST_INTRA_SCAN) && FICAM_Fault_Bank_RG_Match) begin
            if (Current_PBG_Is_Exp)
                Current_Entry_ES = Exp_Weight(Intra_Logical_Bit);
            else
                Current_Entry_ES = SM_Weight(Intra_Logical_Bit);
        end
    end
end

wire [ES_WIDTH-1:0] Candidate_ES_After_Entry;
wire                 Candidate_Better;
wire [CONTROL_WIDTH-1:0] Best_Control_After_Candidate;
wire [ES_WIDTH-1:0]      Best_ES_After_Candidate;

assign Candidate_ES_After_Entry = Candidate_ES_Accumulator + Current_Entry_ES;
assign Candidate_Better = (Candidate_ES_After_Entry < Best_ES_Reg);
assign Best_Control_After_Candidate = Candidate_Better ? Candidate_Control : Best_Control;
assign Best_ES_After_Candidate = Candidate_Better ? Candidate_ES_After_Entry : Best_ES_Reg;

// -----------------------------------------------------------------------------
// External control signals
// -----------------------------------------------------------------------------
assign FICAM_Clear_All = (State == ST_CLEAR);
assign Remap_Info_Clear = (State == ST_CLEAR);

assign FICAM_PBG_Begin = Accept_Bitmap && First_Bitmap_Of_PBG;
assign FICAM_Write_Page_Index = Expected_Load_Page;
assign FICAM_Write_PBG_ID     = Expected_Load_PBG;

assign FICAM_Fault_Write_Enable = Accepted_Sparse_Write;
assign FICAM_Fault_Bank_ID_In   = Latched_Bitmap_Bank;
assign FICAM_Fault_RG_ID_In     = Latched_Bitmap_RG;
assign FICAM_Fault_Word_ID_In   = Priority_Index[5:3];
assign FICAM_Fault_Bit_ID_In    = Priority_Index[2:0];
assign FICAM_Fault_Code_In      = Pending_Fault_Value[Priority_Index]
                                ? FAULT_SA1_CODE : FAULT_SA0_CODE;
assign FICAM_PBG_End = (State == ST_BITMAP_FINISH) && Last_Bitmap_Of_PBG;

assign FICAM_Search_Page_Index = Current_Search_Page;
assign FICAM_Search_PBG_ID     = Current_Search_PBG;
assign FICAM_Directory_Search_Enable = (State == ST_PBG_LOOKUP);

assign FICAM_Fault_Read_Enable =
    ((State == ST_INTER_SCAN) || (State == ST_INTRA_SCAN)) &&
    (Current_PBG_Fault_Count != {PBG_COUNT_WIDTH{1'b0}});
assign FICAM_Fault_Offset_In = Fault_Offset;

assign FICAM_Filter_Enable = (State == ST_INTRA_SCAN);
assign FICAM_Filter_Bank_ID = Search_Bank;
assign FICAM_Filter_RG_ID   = Search_RG;

assign Inter_Write_Enable     = (State == ST_INTER_WRITE);
assign Inter_Write_Page_Index = Current_Search_Page;
assign Inter_Write_PBG_ID     = Current_Search_PBG;
assign Inter_Shift_In         = Best_Control;

assign Intra_Write_Enable     = (State == ST_INTRA_WRITE);
assign Intra_Write_Page_Index = Current_Search_Page;
assign Intra_Write_PBG_ID     = Current_Search_PBG;
assign Intra_Write_Bank_ID    = Search_Bank;
assign Intra_Write_RG_ID      = Search_RG;
assign Intra_CW_In            = Best_Control;

assign Busy = (State != ST_IDLE);
assign Done = (State == ST_DONE);
assign Model_Fault_Scan_Done = (State == ST_SCAN_COMPLETE);

assign State_Debug              = State;
assign Fault_Count_Debug        = Total_Fault_Count_Reg;
assign Current_Page_Debug       = Current_Search_Page;
assign Current_PBG_Debug        = Current_Search_PBG;
assign Current_Bank_Debug       = Search_Bank;
assign Current_RG_Debug         = Search_RG;
assign Candidate_Control_Debug  = Candidate_Control;
assign Fault_Offset_Debug       = Fault_Offset;
assign Candidate_ES_Debug       = Candidate_ES_Accumulator;
assign Best_ES_Debug            = Best_ES_Reg;
assign Current_PBG_Is_Exp_Debug = Current_PBG_Is_Exp;

// -----------------------------------------------------------------------------
// Main sequential controller
// -----------------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        State                         <= ST_IDLE;
        Global_Remap_Mode             <= MODE_BASE;

        Base_Max_Faults_Reg           <= {TOTAL_COUNT_WIDTH{1'b0}};
        Inter_Max_Faults_Reg          <= {TOTAL_COUNT_WIDTH{1'b0}};
        Intra_Max_Faults_Reg          <= {TOTAL_COUNT_WIDTH{1'b0}};
        Total_Fault_Count_Reg         <= {TOTAL_COUNT_WIDTH{1'b0}};

        Load_PBG_Linear               <= {DIRECTORY_ADDR_WIDTH{1'b0}};
        Load_Bank                     <= {BANK_ID_WIDTH{1'b0}};
        Load_RG                       <= {RG_ID_WIDTH{1'b0}};
        Pending_Fault_Bitmap          <= 64'b0;
        Pending_Fault_Value           <= 64'b0;
        Latched_Bitmap_Bank           <= {BANK_ID_WIDTH{1'b0}};
        Latched_Bitmap_RG             <= {RG_ID_WIDTH{1'b0}};

        Search_PBG_Linear             <= {DIRECTORY_ADDR_WIDTH{1'b0}};
        PBG_Field_Phase               <= {FIELD_PHASE_WIDTH{1'b0}};
        Current_PBG_Fault_Count       <= {PBG_COUNT_WIDTH{1'b0}};
        Search_Bank                   <= {BANK_ID_WIDTH{1'b0}};
        Search_RG                     <= {RG_ID_WIDTH{1'b0}};
        Fault_Offset                  <= {PBG_COUNT_WIDTH{1'b0}};

        Candidate_Control             <= {CONTROL_WIDTH{1'b0}};
        Best_Control                  <= {CONTROL_WIDTH{1'b0}};
        Candidate_ES_Accumulator      <= {ES_WIDTH{1'b0}};
        Best_ES_Reg                   <= ES_MAX;

        Input_Error                   <= 1'b0;
        Constraint_Error              <= 1'b0;
        Storage_Overflow_Error        <= 1'b0;
        Directory_Missing_Error       <= 1'b0;
        Fault_Read_Error              <= 1'b0;
        Unsupported_Fault_Code_Error  <= 1'b0;
    end
    else begin
        // Input_Error is a one-cycle protocol pulse; other errors are sticky
        // for the current deployment and are cleared by Start/ST_CLEAR.
        Input_Error <= 1'b0;

        case (State)
            ST_IDLE: begin
                if (Start) begin
                    Base_Max_Faults_Reg          <= Base_Max_Faults;
                    Inter_Max_Faults_Reg         <= Inter_Max_Faults;
                    Intra_Max_Faults_Reg         <= Intra_Max_Faults;
                    Total_Fault_Count_Reg        <= {TOTAL_COUNT_WIDTH{1'b0}};
                    Global_Remap_Mode            <= MODE_BASE;

                    Load_PBG_Linear              <= {DIRECTORY_ADDR_WIDTH{1'b0}};
                    Load_Bank                    <= {BANK_ID_WIDTH{1'b0}};
                    Load_RG                      <= {RG_ID_WIDTH{1'b0}};
                    Pending_Fault_Bitmap         <= 64'b0;
                    Pending_Fault_Value          <= 64'b0;

                    Search_PBG_Linear            <= {DIRECTORY_ADDR_WIDTH{1'b0}};
                    PBG_Field_Phase              <= {FIELD_PHASE_WIDTH{1'b0}};
                    Search_Bank                  <= {BANK_ID_WIDTH{1'b0}};
                    Search_RG                    <= {RG_ID_WIDTH{1'b0}};

                    Constraint_Error             <= 1'b0;
                    Storage_Overflow_Error       <= 1'b0;
                    Directory_Missing_Error      <= 1'b0;
                    Fault_Read_Error             <= 1'b0;
                    Unsupported_Fault_Code_Error <= 1'b0;
                    State                        <= ST_CLEAR;
                end
            end

            ST_CLEAR: begin
                // One complete clock with both Clear outputs asserted.
                State <= ST_LOAD_WAIT;
            end

            ST_LOAD_WAIT: begin
                if (FICAM_Fault_Entry_Full) begin
                    Storage_Overflow_Error <= 1'b1;
                    Global_Remap_Mode      <= MODE_UNREPAIRABLE;
                    State                  <= ST_DONE;
                end
                else if (Fault_Bitmap_Valid) begin
                    if (!Bitmap_Address_Match) begin
                        Input_Error <= 1'b1;
                    end
                    else begin
                        Pending_Fault_Bitmap <= Fault_Valid_Bitmap_In;
                        Pending_Fault_Value  <= Fault_Value_Bitmap_In;
                        Latched_Bitmap_Bank  <= Bitmap_Bank_ID_In;
                        Latched_Bitmap_RG    <= Bitmap_RG_ID_In;

                        if (Fault_Valid_Bitmap_In == 64'b0)
                            State <= ST_BITMAP_FINISH;
                        else
                            State <= ST_SERIALIZE;
                    end
                end
            end

            ST_SERIALIZE: begin
                if (FICAM_Fault_Entry_Full) begin
                    Storage_Overflow_Error <= 1'b1;
                    Global_Remap_Mode      <= MODE_UNREPAIRABLE;
                    State                  <= ST_DONE;
                end
                else if (Priority_Valid) begin
                    if (FICAM_Fault_Write_Ready) begin
                        Pending_Fault_Bitmap <= Pending_After_Accepted_Write;

                        if (Pending_After_Accepted_Write == 64'b0)
                            State <= ST_BITMAP_FINISH;
                    end
                end
                else begin
                    State <= ST_BITMAP_FINISH;
                end
            end

            ST_BITMAP_FINISH: begin
                if (Last_Bitmap_Of_PBG) begin
                    Load_Bank <= {BANK_ID_WIDTH{1'b0}};
                    Load_RG   <= {RG_ID_WIDTH{1'b0}};

                    if (Last_PBG_Of_Model) begin
                        State <= ST_SCAN_COMPLETE;
                    end
                    else begin
                        Load_PBG_Linear <= Load_PBG_Linear + 1'b1;
                        State           <= ST_LOAD_WAIT;
                    end
                end
                else if (Load_RG == LAST_RG) begin
                    Load_RG   <= {RG_ID_WIDTH{1'b0}};
                    Load_Bank <= Load_Bank + 1'b1;
                    State     <= ST_LOAD_WAIT;
                end
                else begin
                    Load_RG <= Load_RG + 1'b1;
                    State   <= ST_LOAD_WAIT;
                end
            end

            ST_SCAN_COMPLETE: begin
                // FICAM committed the final Directory entry at this edge.
                Total_Fault_Count_Reg <= FICAM_Total_Fault_Count;
                State                 <= ST_MODE_SELECT;
            end

            ST_MODE_SELECT: begin
                if ((Base_Max_Faults_Reg > Inter_Max_Faults_Reg) ||
                    (Inter_Max_Faults_Reg > Intra_Max_Faults_Reg)) begin
                    Constraint_Error  <= 1'b1;
                    Global_Remap_Mode <= MODE_UNREPAIRABLE;
                    State             <= ST_DONE;
                end
                else if (Total_Fault_Count_Reg <= Base_Max_Faults_Reg) begin
                    Global_Remap_Mode <= MODE_BASE;
                    State             <= ST_DONE;
                end
                else if (Total_Fault_Count_Reg <= Inter_Max_Faults_Reg) begin
                    Global_Remap_Mode         <= MODE_INTER;
                    Search_PBG_Linear         <= {DIRECTORY_ADDR_WIDTH{1'b0}};
                    PBG_Field_Phase           <= {FIELD_PHASE_WIDTH{1'b0}};
                    State                     <= ST_PBG_LOOKUP;
                end
                else if (Total_Fault_Count_Reg <= Intra_Max_Faults_Reg) begin
                    Global_Remap_Mode         <= MODE_INTRA;
                    Search_PBG_Linear         <= {DIRECTORY_ADDR_WIDTH{1'b0}};
                    PBG_Field_Phase           <= {FIELD_PHASE_WIDTH{1'b0}};
                    Search_Bank               <= {BANK_ID_WIDTH{1'b0}};
                    Search_RG                 <= {RG_ID_WIDTH{1'b0}};
                    State                     <= ST_PBG_LOOKUP;
                end
                else begin
                    Global_Remap_Mode <= MODE_UNREPAIRABLE;
                    State             <= ST_DONE;
                end
            end

            ST_PBG_LOOKUP: begin
                if (!FICAM_Directory_Match) begin
                    Directory_Missing_Error <= 1'b1;
                    Current_PBG_Fault_Count <= {PBG_COUNT_WIDTH{1'b0}};
                end
                else begin
                    Current_PBG_Fault_Count <= FICAM_Directory_Fault_Count;
                end

                if (Global_Remap_Mode == MODE_INTER) begin
                    Candidate_Control        <= {CONTROL_WIDTH{1'b0}};
                    Best_Control             <= {CONTROL_WIDTH{1'b0}};
                    Candidate_ES_Accumulator <= {ES_WIDTH{1'b0}};
                    Best_ES_Reg              <= ES_MAX;

                    if (!FICAM_Directory_Match ||
                        (FICAM_Directory_Fault_Count == {PBG_COUNT_WIDTH{1'b0}})) begin
                        Best_Control <= {CONTROL_WIDTH{1'b0}};
                        Best_ES_Reg  <= {ES_WIDTH{1'b0}};
                        State        <= ST_INTER_WRITE;
                    end
                    else begin
                        State <= ST_INTER_CAND_INIT;
                    end
                end
                else begin
                    Search_Bank <= {BANK_ID_WIDTH{1'b0}};
                    Search_RG   <= {RG_ID_WIDTH{1'b0}};
                    State       <= ST_INTRA_GROUP_INIT;
                end
            end

            ST_INTER_CAND_INIT: begin
                Fault_Offset             <= {PBG_COUNT_WIDTH{1'b0}};
                Candidate_ES_Accumulator <= {ES_WIDTH{1'b0}};
                State                    <= ST_INTER_SCAN;
            end

            ST_INTER_SCAN: begin
                if (!FICAM_Fault_Read_Valid)
                    Fault_Read_Error <= 1'b1;

                if (FICAM_Fault_Read_Valid &&
                    (FICAM_Fault_Code_Out != FAULT_SA0_CODE) &&
                    (FICAM_Fault_Code_Out != FAULT_SA1_CODE))
                    Unsupported_Fault_Code_Error <= 1'b1;

                if (Fault_Offset == (Current_PBG_Fault_Count - 1'b1)) begin
                    Best_Control <= Best_Control_After_Candidate;
                    Best_ES_Reg  <= Best_ES_After_Candidate;

                    if (Candidate_Control == LAST_CONTROL) begin
                        State <= ST_INTER_WRITE;
                    end
                    else begin
                        Candidate_Control <= Candidate_Control + 1'b1;
                        State             <= ST_INTER_CAND_INIT;
                    end
                end
                else begin
                    Candidate_ES_Accumulator <= Candidate_ES_After_Entry;
                    Fault_Offset             <= Fault_Offset + 1'b1;
                end
            end

            ST_INTER_WRITE: begin
                State <= ST_ADVANCE_PBG;
            end

            ST_INTRA_GROUP_INIT: begin
                Candidate_Control        <= {CONTROL_WIDTH{1'b0}};
                Best_Control             <= {CONTROL_WIDTH{1'b0}};
                Candidate_ES_Accumulator <= {ES_WIDTH{1'b0}};
                Best_ES_Reg              <= ES_MAX;

                if (Current_PBG_Fault_Count == {PBG_COUNT_WIDTH{1'b0}}) begin
                    Best_Control <= {CONTROL_WIDTH{1'b0}};
                    Best_ES_Reg  <= {ES_WIDTH{1'b0}};
                    State        <= ST_INTRA_WRITE;
                end
                else begin
                    State <= ST_INTRA_CAND_INIT;
                end
            end

            ST_INTRA_CAND_INIT: begin
                Fault_Offset             <= {PBG_COUNT_WIDTH{1'b0}};
                Candidate_ES_Accumulator <= {ES_WIDTH{1'b0}};
                State                    <= ST_INTRA_SCAN;
            end

            ST_INTRA_SCAN: begin
                if (!FICAM_Fault_Read_Valid)
                    Fault_Read_Error <= 1'b1;

                if (FICAM_Fault_Read_Valid &&
                    (FICAM_Fault_Code_Out != FAULT_SA0_CODE) &&
                    (FICAM_Fault_Code_Out != FAULT_SA1_CODE))
                    Unsupported_Fault_Code_Error <= 1'b1;

                if (Fault_Offset == (Current_PBG_Fault_Count - 1'b1)) begin
                    Best_Control <= Best_Control_After_Candidate;
                    Best_ES_Reg  <= Best_ES_After_Candidate;

                    if (Candidate_Control == LAST_CONTROL) begin
                        State <= ST_INTRA_WRITE;
                    end
                    else begin
                        Candidate_Control <= Candidate_Control + 1'b1;
                        State             <= ST_INTRA_CAND_INIT;
                    end
                end
                else begin
                    Candidate_ES_Accumulator <= Candidate_ES_After_Entry;
                    Fault_Offset             <= Fault_Offset + 1'b1;
                end
            end

            ST_INTRA_WRITE: begin
                if (Search_RG == LAST_RG) begin
                    Search_RG <= {RG_ID_WIDTH{1'b0}};

                    if (Search_Bank == LAST_BANK) begin
                        Search_Bank <= {BANK_ID_WIDTH{1'b0}};
                        State       <= ST_ADVANCE_PBG;
                    end
                    else begin
                        Search_Bank <= Search_Bank + 1'b1;
                        State       <= ST_INTRA_GROUP_INIT;
                    end
                end
                else begin
                    Search_RG <= Search_RG + 1'b1;
                    State     <= ST_INTRA_GROUP_INIT;
                end
            end

            ST_ADVANCE_PBG: begin
                if (Last_Search_PBG) begin
                    State <= ST_DONE;
                end
                else begin
                    Search_PBG_Linear <= Search_PBG_Linear + 1'b1;

                    if (PBG_Field_Phase == (EXP_PBG_PERIOD - 1))
                        PBG_Field_Phase <= {FIELD_PHASE_WIDTH{1'b0}};
                    else
                        PBG_Field_Phase <= PBG_Field_Phase + 1'b1;

                    State <= ST_PBG_LOOKUP;
                end
            end

            ST_DONE: begin
                // Done is a one-cycle pulse. Global_Remap_Mode is retained.
                State <= ST_IDLE;
            end

            default: begin
                State <= ST_IDLE;
            end
        endcase
    end
end

endmodule
