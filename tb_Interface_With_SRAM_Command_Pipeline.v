`timescale 1ns / 1ps

// =============================================================================
// Stage-6B timing-safe SRAM command-pipeline regression
// =============================================================================
// Verifies the complete chain through the generated 16384 x 8 SRAM macro:
//   repair -> mode selection -> remap-info lookup -> AR/Shifter/Transposer/
//   EXP masker -> synchronous SRAM macro write/read.
// =============================================================================

module tb_Interface_With_SRAM_Command_Pipeline;

localparam integer NUM_MODEL_PAGES      = 1;
localparam integer PAGE_INDEX_WIDTH     = 1;
localparam integer NUM_FBB_PER_PAGE     = 32;
localparam integer FBB_ID_WIDTH         = 5;
localparam integer TOTAL_OCCUPIED_FBBS  = 2;
localparam integer TOTAL_COUNT_WIDTH    = 8;
localparam integer NUM_FAULT_ENTRIES    = 64;
localparam integer FAULT_ADDR_WIDTH     = 6;
localparam integer FBB_COUNT_WIDTH      = 13;
localparam integer ES_WIDTH             = 20;

localparam [1:0] MODE_BASE         = 2'b00;
localparam [1:0] MODE_INTER        = 2'b01;
localparam [1:0] MODE_INTRA        = 2'b10;
localparam [1:0] MODE_UNREPAIRABLE = 2'b11;
localparam [2:0] FAULT_SA0 = 3'b001;
localparam [2:0] FAULT_SA1 = 3'b010;
localparam integer EXPECTED_TOTAL_FAULTS = 18;

reg clk;
reg rst_n;
reg Repair_Start;
reg [TOTAL_COUNT_WIDTH-1:0] Base_Max_Faults;
reg [TOTAL_COUNT_WIDTH-1:0] Inter_Max_Faults;
reg [TOTAL_COUNT_WIDTH-1:0] Intra_Max_Faults;

reg Fault_Bitmap_Valid;
reg [63:0] Fault_Valid_Bitmap_In;
reg [63:0] Fault_Value_Bitmap_In;
reg [PAGE_INDEX_WIDTH-1:0] Bitmap_Page_Index_In;
reg [FBB_ID_WIDTH-1:0] Bitmap_FBB_ID_In;
reg [2:0] Bitmap_Bank_ID_In;
reg [2:0] Bitmap_RG_ID_In;
wire Fault_Bitmap_Ready;

reg Normal_Lookup_Enable;
reg [PAGE_INDEX_WIDTH-1:0] Normal_Model_Page_Index;
reg [13:0] Normal_Page_Offset;
reg [2:0] Intra_Logical_TWW_Addr;
wire Normal_Access_Ready;
wire Remap_Control_Valid;
wire Remap_Control_Missing;
wire [2:0] Selected_Shift_Mode;
wire [2:0] Selected_Address_CW;
wire Address_Remap_Enable;
wire [2:0] Intra_Physical_TWW_Addr;
wire Data_Shift_Enable;
wire Normal_FBB_Is_Exp;

reg Normal_Buffer_Write_Enable;
reg Normal_Buffer_Read_Enable;
reg [7:0] Normal_Buffer_Data_In;
wire Normal_Buffer_Request_Ready;
wire [7:0] Normal_Buffer_Data_Out;
wire Normal_Buffer_Data_Valid;
wire Normal_Buffer_Operation_Done;

reg Intra_Transaction_Start;
reg Intra_Transaction_Direction;
wire Intra_Transaction_Ready;
wire Intra_Transaction_Busy;
wire Intra_Transaction_Done;
wire Intra_Transaction_Error;
reg Intra_Write_WW_Valid;
reg [7:0] Intra_Write_WW_Data;
wire Intra_Write_WW_Ready;
wire Intra_Read_WW_Valid;
wire [7:0] Intra_Read_WW_Data;
wire Intra_Read_WW_Is_Exp;

wire Buffer_Busy;
wire Buffer_Access_Error;
wire Buffer_Write_Enable_Debug;
wire Buffer_Read_Enable_Debug;
wire [2:0] Buffer_Bank_Addr_Debug;
wire [10:0] Buffer_Local_Addr_Debug;
wire [7:0] Buffer_Data_In_Debug;
wire [7:0] Buffer_Data_Out_Debug;
wire SRAM_CEN_Debug;
wire SRAM_WEN_Debug;
wire [13:0] SRAM_Address_Debug;
wire [7:0] SRAM_Data_In_Debug;
wire [7:0] SRAM_Data_Out_Debug;
wire [2:0] SRAM_EMA_Debug;
wire Intra_Buffer_Write_Valid_Debug;
wire [2:0] Intra_Buffer_Physical_TWW_Debug;
wire Intra_Buffer_Read_Request_Debug;

wire [4:0] Normal_FBB_ID;
wire [2:0] Normal_Bank_ID;
wire [2:0] Normal_RG_ID;
wire [2:0] Normal_Word_ID;

wire [1:0] Global_Remap_Mode;
wire Repair_Busy;
wire Repair_Done;
wire Model_Fault_Scan_Done;
wire Unrepairable;
wire [TOTAL_COUNT_WIDTH-1:0] Total_Fault_Count;
wire Input_Error;
wire Constraint_Error;
wire Storage_Overflow_Error;
wire Directory_Missing_Error;
wire Fault_Read_Error;
wire Unsupported_Fault_Code_Error;
wire FICAM_Overflow_Error;
wire FICAM_Protocol_Error;
wire FICAM_Duplicate_FBB_Error;
wire Remap_Write_Address_Error;
wire Remap_Simultaneous_Write_Error;
wire [5:0] CU_State_Debug;
wire [PAGE_INDEX_WIDTH-1:0] CU_Current_Page_Debug;
wire [4:0] CU_Current_FBB_Debug;
wire [2:0] CU_Current_Bank_Debug;
wire [2:0] CU_Current_RG_Debug;
wire [2:0] CU_Candidate_Control_Debug;
wire [ES_WIDTH-1:0] CU_Candidate_ES_Debug;
wire [ES_WIDTH-1:0] CU_Best_ES_Debug;

integer errors;
integer bank_i;
integer rg_i;
integer fbb_i;
integer inter_write_count;
integer intra_write_count;
integer sparse_write_count;
integer scan_done_count;
integer saw_any_error;
integer run_active;
reg [63:0] vmap;
reg [63:0] fmap;

reg [7:0] intra_source_ww [0:7];
reg [7:0] intra_expected_tww [0:7];
reg [7:0] intra_restored_ww [0:7];
integer intra_read_output_count;
integer intra_buffer_write_count;
integer tx_i;
integer tx_j;
integer tx_timeout;

integer sram_write_count;
integer sram_read_count;
integer intra_sram_write_count;
reg [13:0] last_sram_write_address;
reg [7:0]  last_sram_write_data;
reg [13:0] last_sram_read_address;
reg [13:0] intra_sram_write_address [0:7];
reg [7:0]  intra_sram_write_data [0:7];
integer write_count_before;
reg [13:0] expected_sram_address;

Interface_With_SRAM_Command_Pipeline #(
    .NUM_MODEL_PAGES      (NUM_MODEL_PAGES),
    .PAGE_INDEX_WIDTH     (PAGE_INDEX_WIDTH),
    .NUM_FBB_PER_PAGE     (NUM_FBB_PER_PAGE),
    .FBB_ID_WIDTH         (FBB_ID_WIDTH),
    .TOTAL_OCCUPIED_FBBS  (TOTAL_OCCUPIED_FBBS),
    .DIRECTORY_ADDR_WIDTH (PAGE_INDEX_WIDTH + FBB_ID_WIDTH),
    .ES_WIDTH             (ES_WIDTH),
    .FBB_COUNT_WIDTH      (FBB_COUNT_WIDTH),
    .TOTAL_COUNT_WIDTH    (TOTAL_COUNT_WIDTH),
    .NUM_FAULT_ENTRIES    (NUM_FAULT_ENTRIES),
    .FAULT_ADDR_WIDTH     (FAULT_ADDR_WIDTH)
) dut (
    .clk                            (clk),
    .rst_n                          (rst_n),
    .Repair_Start                   (Repair_Start),
    .Base_Max_Faults                (Base_Max_Faults),
    .Inter_Max_Faults               (Inter_Max_Faults),
    .Intra_Max_Faults               (Intra_Max_Faults),
    .Fault_Bitmap_Valid             (Fault_Bitmap_Valid),
    .Fault_Valid_Bitmap_In          (Fault_Valid_Bitmap_In),
    .Fault_Value_Bitmap_In          (Fault_Value_Bitmap_In),
    .Bitmap_Page_Index_In           (Bitmap_Page_Index_In),
    .Bitmap_FBB_ID_In               (Bitmap_FBB_ID_In),
    .Bitmap_Bank_ID_In              (Bitmap_Bank_ID_In),
    .Bitmap_RG_ID_In                (Bitmap_RG_ID_In),
    .Fault_Bitmap_Ready             (Fault_Bitmap_Ready),
    .Normal_Lookup_Enable           (Normal_Lookup_Enable),
    .Normal_Model_Page_Index        (Normal_Model_Page_Index),
    .Normal_Page_Offset             (Normal_Page_Offset),
    .Intra_Logical_TWW_Addr         (Intra_Logical_TWW_Addr),
    .Normal_Access_Ready            (Normal_Access_Ready),
    .Remap_Control_Valid            (Remap_Control_Valid),
    .Remap_Control_Missing          (Remap_Control_Missing),
    .Selected_Shift_Mode            (Selected_Shift_Mode),
    .Selected_Address_CW            (Selected_Address_CW),
    .Address_Remap_Enable           (Address_Remap_Enable),
    .Intra_Physical_TWW_Addr        (Intra_Physical_TWW_Addr),
    .Data_Shift_Enable              (Data_Shift_Enable),
    .Normal_FBB_Is_Exp              (Normal_FBB_Is_Exp),
    .Normal_Buffer_Write_Enable     (Normal_Buffer_Write_Enable),
    .Normal_Buffer_Read_Enable      (Normal_Buffer_Read_Enable),
    .Normal_Buffer_Data_In          (Normal_Buffer_Data_In),
    .Normal_Buffer_Request_Ready    (Normal_Buffer_Request_Ready),
    .Normal_Buffer_Data_Out         (Normal_Buffer_Data_Out),
    .Normal_Buffer_Data_Valid       (Normal_Buffer_Data_Valid),
    .Normal_Buffer_Operation_Done   (Normal_Buffer_Operation_Done),
    .Intra_Transaction_Start        (Intra_Transaction_Start),
    .Intra_Transaction_Direction    (Intra_Transaction_Direction),
    .Intra_Transaction_Ready        (Intra_Transaction_Ready),
    .Intra_Transaction_Busy         (Intra_Transaction_Busy),
    .Intra_Transaction_Done         (Intra_Transaction_Done),
    .Intra_Transaction_Error        (Intra_Transaction_Error),
    .Intra_Write_WW_Valid           (Intra_Write_WW_Valid),
    .Intra_Write_WW_Data            (Intra_Write_WW_Data),
    .Intra_Write_WW_Ready           (Intra_Write_WW_Ready),
    .Intra_Read_WW_Valid            (Intra_Read_WW_Valid),
    .Intra_Read_WW_Data             (Intra_Read_WW_Data),
    .Intra_Read_WW_Is_Exp           (Intra_Read_WW_Is_Exp),
    .Buffer_Busy                    (Buffer_Busy),
    .Buffer_Access_Error            (Buffer_Access_Error),
    .Buffer_Write_Enable_Debug      (Buffer_Write_Enable_Debug),
    .Buffer_Read_Enable_Debug       (Buffer_Read_Enable_Debug),
    .Buffer_Bank_Addr_Debug         (Buffer_Bank_Addr_Debug),
    .Buffer_Local_Addr_Debug        (Buffer_Local_Addr_Debug),
    .Buffer_Data_In_Debug           (Buffer_Data_In_Debug),
    .Buffer_Data_Out_Debug          (Buffer_Data_Out_Debug),
    .SRAM_CEN_Debug                 (SRAM_CEN_Debug),
    .SRAM_WEN_Debug                 (SRAM_WEN_Debug),
    .SRAM_Address_Debug             (SRAM_Address_Debug),
    .SRAM_Data_In_Debug             (SRAM_Data_In_Debug),
    .SRAM_Data_Out_Debug            (SRAM_Data_Out_Debug),
    .SRAM_EMA_Debug                 (SRAM_EMA_Debug),
    .Intra_Buffer_Write_Valid_Debug (Intra_Buffer_Write_Valid_Debug),
    .Intra_Buffer_Physical_TWW_Debug(Intra_Buffer_Physical_TWW_Debug),
    .Intra_Buffer_Read_Request_Debug(Intra_Buffer_Read_Request_Debug),
    .Normal_FBB_ID                  (Normal_FBB_ID),
    .Normal_Bank_ID                 (Normal_Bank_ID),
    .Normal_RG_ID                   (Normal_RG_ID),
    .Normal_Word_ID                 (Normal_Word_ID),
    .Global_Remap_Mode              (Global_Remap_Mode),
    .Repair_Busy                    (Repair_Busy),
    .Repair_Done                    (Repair_Done),
    .Model_Fault_Scan_Done          (Model_Fault_Scan_Done),
    .Unrepairable                   (Unrepairable),
    .Total_Fault_Count              (Total_Fault_Count),
    .Input_Error                    (Input_Error),
    .Constraint_Error               (Constraint_Error),
    .Storage_Overflow_Error         (Storage_Overflow_Error),
    .Directory_Missing_Error        (Directory_Missing_Error),
    .Fault_Read_Error               (Fault_Read_Error),
    .Unsupported_Fault_Code_Error   (Unsupported_Fault_Code_Error),
    .FICAM_Overflow_Error           (FICAM_Overflow_Error),
    .FICAM_Protocol_Error           (FICAM_Protocol_Error),
    .FICAM_Duplicate_FBB_Error      (FICAM_Duplicate_FBB_Error),
    .Remap_Write_Address_Error      (Remap_Write_Address_Error),
    .Remap_Simultaneous_Write_Error (Remap_Simultaneous_Write_Error),
    .CU_State_Debug                 (CU_State_Debug),
    .CU_Current_Page_Debug          (CU_Current_Page_Debug),
    .CU_Current_FBB_Debug           (CU_Current_FBB_Debug),
    .CU_Current_Bank_Debug          (CU_Current_Bank_Debug),
    .CU_Current_RG_Debug            (CU_Current_RG_Debug),
    .CU_Candidate_Control_Debug     (CU_Candidate_Control_Debug),
    .CU_Candidate_ES_Debug          (CU_Candidate_ES_Debug),
    .CU_Best_ES_Debug               (CU_Best_ES_Debug)
);

always #5 clk = ~clk;

always @(posedge clk) begin
    if (run_active) begin
        if (dut.u_core.inter_write_enable)
            inter_write_count = inter_write_count + 1;
        if (dut.u_core.intra_write_enable)
            intra_write_count = intra_write_count + 1;
        if (dut.u_core.ficam_fault_write_enable)
            sparse_write_count = sparse_write_count + 1;
        if (Model_Fault_Scan_Done)
            scan_done_count = scan_done_count + 1;

        if (Input_Error || Constraint_Error || Storage_Overflow_Error ||
            Directory_Missing_Error || Fault_Read_Error ||
            Unsupported_Fault_Code_Error || FICAM_Overflow_Error ||
            FICAM_Protocol_Error || FICAM_Duplicate_FBB_Error ||
            Remap_Write_Address_Error || Remap_Simultaneous_Write_Error)
            saw_any_error = 1;
    end

    // Observe the generated macro's active-low pins on every access edge.
    if (!SRAM_CEN_Debug) begin
        if (SRAM_EMA_Debug !== 3'b000) begin
            $display("[FAIL] SRAM EMA=%b expected=000", SRAM_EMA_Debug);
            errors = errors + 1;
        end

        if (!SRAM_WEN_Debug) begin
            last_sram_write_address = SRAM_Address_Debug;
            last_sram_write_data    = SRAM_Data_In_Debug;
            sram_write_count        = sram_write_count + 1;

            if (Intra_Buffer_Write_Valid_Debug) begin
                if (intra_sram_write_count < 8) begin
                    intra_sram_write_address[intra_sram_write_count] = SRAM_Address_Debug;
                    intra_sram_write_data[intra_sram_write_count]    = SRAM_Data_In_Debug;
                end
                intra_sram_write_count = intra_sram_write_count + 1;
            end
        end
        else begin
            last_sram_read_address = SRAM_Address_Debug;
            sram_read_count = sram_read_count + 1;
        end
    end

    if ((Buffer_Write_Enable_Debug || Buffer_Read_Enable_Debug) && SRAM_CEN_Debug) begin
        $display("[FAIL] logical buffer request did not enable SRAM CEN");
        errors = errors + 1;
    end
    if (Buffer_Write_Enable_Debug && SRAM_WEN_Debug) begin
        $display("[FAIL] SRAM WEN was not low during write");
        errors = errors + 1;
    end
    if (Buffer_Read_Enable_Debug && !SRAM_WEN_Debug) begin
        $display("[FAIL] SRAM WEN was not high during read");
        errors = errors + 1;
    end

    if (Intra_Buffer_Write_Valid_Debug)
        intra_buffer_write_count = intra_buffer_write_count + 1;

    if (Intra_Read_WW_Valid) begin
        if (intra_read_output_count < 8)
            intra_restored_ww[intra_read_output_count] = Intra_Read_WW_Data;
        else begin
            $display("[FAIL] unexpected extra Intra WW output");
            errors = errors + 1;
        end
        intra_read_output_count = intra_read_output_count + 1;
    end
end

function [13:0] make_page_offset;
    input [4:0] fbb;
    input [2:0] bank;
    input [2:0] rg;
    input [2:0] word_id;
    begin
        make_page_offset = {fbb, bank, rg, word_id};
    end
endfunction

function [10:0] make_local_addr;
    input [4:0] fbb;
    input [2:0] rg;
    input [2:0] word_id;
    begin
        make_local_addr = {fbb, rg, word_id};
    end
endfunction

function [7:0] rotate_right8;
    input [7:0] value;
    input [2:0] amount;
    begin
        if (amount == 0)
            rotate_right8 = value;
        else
            rotate_right8 = (value >> amount) | (value << (8 - amount));
    end
endfunction

function [15:0] pack_fault_entry;
    input [2:0] bank;
    input [2:0] rg;
    input [2:0] word_id;
    input [2:0] bit_id;
    input [2:0] fault_code;
    begin
        pack_fault_entry = {bank, rg, word_id, bit_id, fault_code, 1'b1};
    end
endfunction

task reset_run_counters;
    begin
        inter_write_count = 0;
        intra_write_count = 0;
        sparse_write_count = 0;
        scan_done_count = 0;
        saw_any_error = 0;
        intra_read_output_count = 0;
        intra_buffer_write_count = 0;
        sram_write_count = 0;
        sram_read_count = 0;
        intra_sram_write_count = 0;
        last_sram_write_address = 14'b0;
        last_sram_write_data = 8'b0;
        last_sram_read_address = 14'b0;
    end
endtask

task apply_reset;
    begin
        rst_n = 1'b0;
        Repair_Start = 1'b0;
        Fault_Bitmap_Valid = 1'b0;
        Normal_Lookup_Enable = 1'b0;
        Normal_Buffer_Write_Enable = 1'b0;
        Normal_Buffer_Read_Enable = 1'b0;
        Intra_Transaction_Start = 1'b0;
        Intra_Write_WW_Valid = 1'b0;
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        reset_run_counters();
    end
endtask

task pulse_start;
    input [TOTAL_COUNT_WIDTH-1:0] base_th;
    input [TOTAL_COUNT_WIDTH-1:0] inter_th;
    input [TOTAL_COUNT_WIDTH-1:0] intra_th;
    begin
        @(negedge clk);
        Base_Max_Faults  = base_th;
        Inter_Max_Faults = inter_th;
        Intra_Max_Faults = intra_th;
        Repair_Start     = 1'b1;
        @(negedge clk);
        Repair_Start     = 1'b0;
    end
endtask

task send_bitmap;
    input [PAGE_INDEX_WIDTH-1:0] page_index;
    input [FBB_ID_WIDTH-1:0] fbb_id;
    input [2:0] bank_id;
    input [2:0] rg_id;
    input [63:0] valid_map;
    input [63:0] value_map;
    begin
        Fault_Bitmap_Valid = 1'b0;
        while (!Fault_Bitmap_Ready)
            @(negedge clk);
        Bitmap_Page_Index_In  = page_index;
        Bitmap_FBB_ID_In      = fbb_id;
        Bitmap_Bank_ID_In     = bank_id;
        Bitmap_RG_ID_In       = rg_id;
        Fault_Valid_Bitmap_In = valid_map;
        Fault_Value_Bitmap_In = value_map;
        Fault_Bitmap_Valid    = 1'b1;
        @(negedge clk);
        Fault_Bitmap_Valid    = 1'b0;
        Fault_Valid_Bitmap_In = 64'b0;
        Fault_Value_Bitmap_In = 64'b0;
    end
endtask

task stream_multi_fault_model;
    begin
        for (fbb_i = 0; fbb_i < TOTAL_OCCUPIED_FBBS; fbb_i = fbb_i + 1) begin
            for (bank_i = 0; bank_i < 8; bank_i = bank_i + 1) begin
                for (rg_i = 0; rg_i < 8; rg_i = rg_i + 1) begin
                    vmap = 64'b0;
                    fmap = 64'b0;
                    if (fbb_i == 0) begin
                        if ((bank_i == 0) && (rg_i == 0)) begin
                            vmap = (64'b1 << 2) | (64'b1 << 13) | (64'b1 << 23);
                            fmap = (64'b1 << 13);
                        end
                        else if ((bank_i == 1) && (rg_i == 1)) begin
                            vmap = (64'b1 << 25); fmap = (64'b1 << 25);
                        end
                        else if ((bank_i == 2) && (rg_i == 2)) begin
                            vmap = (64'b1 << 36);
                        end
                        else if ((bank_i == 3) && (rg_i == 3)) begin
                            vmap = (64'b1 << 47); fmap = (64'b1 << 47);
                        end
                    end
                    else if (fbb_i == 1) begin
                        if ((bank_i == 4) && (rg_i == 0)) begin
                            vmap = (64'b1 << 0) | (64'b1 << 12) | (64'b1 << 23);
                            fmap = (64'b1 << 0) | (64'b1 << 23);
                        end
                        else if ((bank_i == 5) && (rg_i == 1)) begin
                            vmap = (64'b1 << 1) | (64'b1 << 11) | (64'b1 << 23);
                            fmap = (64'b1 << 11);
                        end
                        else if ((bank_i == 6) && (rg_i == 2)) begin
                            vmap = (64'b1 << 2) | (64'b1 << 13) | (64'b1 << 23);
                            fmap = (64'b1 << 2) | (64'b1 << 13);
                        end
                        else if ((bank_i == 7) && (rg_i == 3)) begin
                            vmap = (64'b1 << 3) | (64'b1 << 12) | (64'b1 << 21);
                            fmap = (64'b1 << 21);
                        end
                    end
                    send_bitmap(0, fbb_i[4:0], bank_i[2:0], rg_i[2:0], vmap, fmap);
                end
            end
        end
    end
endtask

task wait_for_done;
    integer timeout;
    begin
        timeout = 0;
        while (!Repair_Done && timeout < 100000) begin
            @(negedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 100000) begin
            $display("[FAIL] repair timeout state=%0d fbb=%0d bank=%0d rg=%0d",
                     CU_State_Debug, CU_Current_FBB_Debug,
                     CU_Current_Bank_Debug, CU_Current_RG_Debug);
            errors = errors + 1;
        end
        @(negedge clk);
    end
endtask

task check_common_results;
    input [1:0] expected_mode;
    input integer expected_inter_writes;
    input integer expected_intra_writes;
    begin
        if (Global_Remap_Mode !== expected_mode) begin
            $display("[FAIL] mode=%0d expected=%0d", Global_Remap_Mode, expected_mode);
            errors = errors + 1;
        end
        if (Total_Fault_Count !== EXPECTED_TOTAL_FAULTS) begin
            $display("[FAIL] total faults=%0d expected=%0d", Total_Fault_Count, EXPECTED_TOTAL_FAULTS);
            errors = errors + 1;
        end
        if (sparse_write_count != EXPECTED_TOTAL_FAULTS || scan_done_count != 1) begin
            $display("[FAIL] sparse writes=%0d scan_done=%0d", sparse_write_count, scan_done_count);
            errors = errors + 1;
        end
        if (inter_write_count != expected_inter_writes ||
            intra_write_count != expected_intra_writes) begin
            $display("[FAIL] remap writes inter=%0d/%0d intra=%0d/%0d",
                     inter_write_count, expected_inter_writes,
                     intra_write_count, expected_intra_writes);
            errors = errors + 1;
        end
        if (saw_any_error) begin
            $display("[FAIL] repair error flag asserted");
            errors = errors + 1;
        end
        if (dut.u_core.u_ficam.Directory_Start_Mem[0] !== 0 ||
            dut.u_core.u_ficam.Directory_Count_Mem[0] !== 6 ||
            dut.u_core.u_ficam.Directory_Start_Mem[1] !== 6 ||
            dut.u_core.u_ficam.Directory_Count_Mem[1] !== 12) begin
            $display("[FAIL] directory range mismatch");
            errors = errors + 1;
        end
        if (dut.u_core.u_ficam.Fault_Entry_Mem[0] !==
            pack_fault_entry(0,0,0,2,FAULT_SA0) ||
            dut.u_core.u_ficam.Fault_Entry_Mem[17] !==
            pack_fault_entry(7,3,2,5,FAULT_SA1)) begin
            $display("[FAIL] sparse entry content mismatch");
            errors = errors + 1;
        end
    end
endtask

task set_lookup;
    input [4:0] fbb;
    input [2:0] bank;
    input [2:0] rg;
    input [2:0] word_id;
    begin
        Normal_Model_Page_Index = 0;
        Normal_Page_Offset = make_page_offset(fbb, bank, rg, word_id);
        Normal_Lookup_Enable = 1'b1;
        #1;
        if (!Remap_Control_Valid || Remap_Control_Missing) begin
            $display("[FAIL] lookup invalid mode=%0d FBB=%0d Bank=%0d RG=%0d",
                     Global_Remap_Mode, fbb, bank, rg);
            errors = errors + 1;
        end
    end
endtask

task normal_buffer_write;
    input [4:0] fbb;
    input [2:0] bank;
    input [2:0] rg;
    input [2:0] word_id;
    input [7:0] data_value;
    integer timeout;
    begin
        set_lookup(fbb, bank, rg, word_id);
        timeout = 0;
        while (!Normal_Buffer_Request_Ready && timeout < 50) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!Normal_Buffer_Request_Ready) begin
            $display("[FAIL] normal write request never ready");
            errors = errors + 1;
        end
        Normal_Buffer_Data_In = data_value;
        Normal_Buffer_Write_Enable = 1'b1;
        @(negedge clk);
        Normal_Buffer_Write_Enable = 1'b0;

        timeout = 0;
        while (!Normal_Buffer_Operation_Done && timeout < 30) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!Normal_Buffer_Operation_Done) begin
            $display("[FAIL] normal write done timeout");
            errors = errors + 1;
        end
        Normal_Lookup_Enable = 1'b0;
        @(negedge clk);
    end
endtask

task normal_buffer_read_check;
    input [4:0] fbb;
    input [2:0] bank;
    input [2:0] rg;
    input [2:0] word_id;
    input [7:0] expected_data;
    integer timeout;
    begin
        set_lookup(fbb, bank, rg, word_id);
        timeout = 0;
        while (!Normal_Buffer_Request_Ready && timeout < 50) begin
            @(negedge clk); timeout = timeout + 1;
        end
        Normal_Buffer_Read_Enable = 1'b1;
        @(negedge clk);
        Normal_Buffer_Read_Enable = 1'b0;
        timeout = 0;
        while (!Normal_Buffer_Data_Valid && timeout < 20) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!Normal_Buffer_Data_Valid) begin
            $display("[FAIL] normal read data-valid timeout");
            errors = errors + 1;
        end
        else if (Normal_Buffer_Data_Out !== expected_data) begin
            $display("[FAIL] normal read got=%h expected=%h FBB=%0d", Normal_Buffer_Data_Out, expected_data, fbb);
            errors = errors + 1;
        end
        if (!Normal_Buffer_Operation_Done) begin
            $display("[FAIL] normal read done pulse missing");
            errors = errors + 1;
        end
        Normal_Lookup_Enable = 1'b0;
        @(negedge clk);
    end
endtask

task check_normal_buffer_roundtrip;
    input [4:0] fbb;
    input [2:0] bank;
    input [2:0] rg;
    input [2:0] word_id;
    input [7:0] logical_data;
    input [2:0] expected_shift;
    input       expected_is_exp;
    reg [7:0] expected_raw;
    reg [7:0] expected_read;
    reg [10:0] local_addr;
    begin
        expected_raw = rotate_right8(logical_data, expected_shift);
        expected_read = expected_is_exp ? {1'b0, logical_data[6:0]} : logical_data;
        local_addr = make_local_addr(fbb, rg, word_id);

        expected_sram_address = make_page_offset(fbb, bank, rg, word_id);
        write_count_before = sram_write_count;
        normal_buffer_write(fbb, bank, rg, word_id, logical_data);
        if (sram_write_count != (write_count_before + 1)) begin
            $display("[FAIL] normal SRAM write-count did not advance exactly once");
            errors = errors + 1;
        end
        if (last_sram_write_address !== expected_sram_address ||
            last_sram_write_data !== expected_raw) begin
            $display("[FAIL] raw SRAM write A=%h/%h D=%h/%h",
                     last_sram_write_address, expected_sram_address,
                     last_sram_write_data, expected_raw);
            errors = errors + 1;
        end
        normal_buffer_read_check(fbb, bank, rg, word_id, expected_read);
        if (last_sram_read_address !== expected_sram_address) begin
            $display("[FAIL] raw SRAM read A=%h expected=%h",
                     last_sram_read_address, expected_sram_address);
            errors = errors + 1;
        end
    end
endtask

task build_intra_matrix;
    input integer pattern_select;
    begin
        if (pattern_select == 0) begin
            intra_source_ww[0]=8'h80; intra_source_ww[1]=8'hFF;
            intra_source_ww[2]=8'hA5; intra_source_ww[3]=8'hC3;
            intra_source_ww[4]=8'h5A; intra_source_ww[5]=8'h96;
            intra_source_ww[6]=8'hE1; intra_source_ww[7]=8'h7E;
        end
        else begin
            intra_source_ww[0]=8'h91; intra_source_ww[1]=8'h42;
            intra_source_ww[2]=8'hE7; intra_source_ww[3]=8'h18;
            intra_source_ww[4]=8'hBD; intra_source_ww[5]=8'h64;
            intra_source_ww[6]=8'hCA; intra_source_ww[7]=8'h3F;
        end
        for (tx_j=0; tx_j<8; tx_j=tx_j+1) begin
            intra_expected_tww[tx_j]=8'b0;
            for (tx_i=0; tx_i<8; tx_i=tx_i+1)
                intra_expected_tww[tx_j][tx_i]=intra_source_ww[tx_i][tx_j];
        end
    end
endtask

task start_intra_transaction;
    input direction_value;
    begin
        tx_timeout=0;
        while (!Intra_Transaction_Ready && tx_timeout<100) begin
            @(negedge clk); tx_timeout=tx_timeout+1;
        end
        if (!Intra_Transaction_Ready) begin
            $display("[FAIL] Intra transaction not ready"); errors=errors+1;
        end
        Intra_Transaction_Direction=direction_value;
        Intra_Transaction_Start=1'b1;
        @(negedge clk);
        Intra_Transaction_Start=1'b0;
    end
endtask

task stream_intra_write_words;
    begin
        for (tx_i=0; tx_i<8; tx_i=tx_i+1) begin
            tx_timeout=0;
            while (!Intra_Write_WW_Ready && tx_timeout<100) begin
                @(negedge clk); tx_timeout=tx_timeout+1;
            end
            if (!Intra_Write_WW_Ready) begin
                $display("[FAIL] Intra WW ready timeout row=%0d", tx_i);
                errors=errors+1;
            end
            Intra_Write_WW_Data=intra_source_ww[tx_i];
            Intra_Write_WW_Valid=1'b1;
            @(negedge clk);
            Intra_Write_WW_Valid=1'b0;
        end
    end
endtask

task wait_intra_done;
    begin
        tx_timeout=0;
        while (!Intra_Transaction_Done && tx_timeout<600) begin
            @(negedge clk); tx_timeout=tx_timeout+1;
        end
        if (!Intra_Transaction_Done) begin
            $display("[FAIL] Intra transaction timeout"); errors=errors+1;
        end
        @(negedge clk);
    end
endtask

task check_intra_buffer_roundtrip;
    input [4:0] fbb;
    input [2:0] bank;
    input [2:0] rg;
    input [2:0] expected_cw;
    input       expected_is_exp;
    input integer pattern_select;
    reg [2:0] physical_index;
    reg [10:0] local_addr;
    reg [7:0] expected_word;
    begin
        build_intra_matrix(pattern_select);
        intra_read_output_count=0;
        intra_buffer_write_count=0;
        intra_sram_write_count=0;
        for (tx_i=0; tx_i<8; tx_i=tx_i+1) begin
            intra_restored_ww[tx_i]=8'b0;
            intra_sram_write_address[tx_i]=14'b0;
            intra_sram_write_data[tx_i]=8'b0;
        end

        set_lookup(fbb, bank, rg, 0);
        if (Selected_Address_CW !== expected_cw || Normal_FBB_Is_Exp !== expected_is_exp) begin
            $display("[FAIL] Intra lookup CW=%0d/%0d EXP=%b/%b",
                     Selected_Address_CW, expected_cw, Normal_FBB_Is_Exp, expected_is_exp);
            errors=errors+1;
        end

        start_intra_transaction(1'b1);
        stream_intra_write_words();
        wait_intra_done();
        if (intra_buffer_write_count != 8) begin
            $display("[FAIL] Intra Buffer writes=%0d expected=8", intra_buffer_write_count);
            errors=errors+1;
        end

        if (intra_sram_write_count != 8) begin
            $display("[FAIL] Intra SRAM writes=%0d expected=8", intra_sram_write_count);
            errors=errors+1;
        end
        for (tx_i=0; tx_i<8; tx_i=tx_i+1) begin
            physical_index = tx_i[2:0] ^ expected_cw;
            expected_sram_address = make_page_offset(fbb, bank, rg, physical_index);
            if (intra_sram_write_address[tx_i] !== expected_sram_address ||
                intra_sram_write_data[tx_i] !== intra_expected_tww[tx_i]) begin
                $display("[FAIL] Intra SRAM write logical=%0d A=%h/%h D=%h/%h",
                         tx_i,
                         intra_sram_write_address[tx_i], expected_sram_address,
                         intra_sram_write_data[tx_i], intra_expected_tww[tx_i]);
                errors=errors+1;
            end
        end

        intra_read_output_count=0;
        start_intra_transaction(1'b0);
        wait_intra_done();
        if (intra_read_output_count != 8) begin
            $display("[FAIL] restored WW count=%0d expected=8", intra_read_output_count);
            errors=errors+1;
        end
        for (tx_i=0; tx_i<8; tx_i=tx_i+1) begin
            expected_word = expected_is_exp ? {1'b0,intra_source_ww[tx_i][6:0]} : intra_source_ww[tx_i];
            if (intra_restored_ww[tx_i] !== expected_word) begin
                $display("[FAIL] restored WW row=%0d got=%h expected=%h",
                         tx_i, intra_restored_ww[tx_i], expected_word);
                errors=errors+1;
            end
        end
        if (Intra_Transaction_Error || Buffer_Access_Error) begin
            $display("[FAIL] Intra/Buffer protocol error asserted");
            errors=errors+1;
        end
        Normal_Lookup_Enable=1'b0;
        @(negedge clk);
    end
endtask

task run_base_case;
    begin
        $display("\n[CASE] Stage-6B BASE timing-safe SRAM");
        apply_reset(); run_active=1;
        pulse_start(18,18,18); stream_multi_fault_model(); wait_for_done();
        run_active=0; check_common_results(MODE_BASE,0,0);
        check_normal_buffer_roundtrip(0,2,3,4,8'hD3,3'd0,1'b1);
        check_normal_buffer_roundtrip(1,7,3,6,8'hA6,3'd0,1'b0);
        if (Buffer_Access_Error) begin $display("[FAIL] BASE buffer error"); errors=errors+1; end
        $display("[PASS] Stage-6 BASE SRAM macro roundtrip completed");
    end
endtask

task run_inter_case;
    begin
        $display("\n[CASE] Stage-6B INTER timing-safe SRAM");
        apply_reset(); run_active=1;
        pulse_start(0,18,18); stream_multi_fault_model(); wait_for_done();
        run_active=0; check_common_results(MODE_INTER,2,0);
        // Previously verified expected shifts: FBB0=5, FBB1=4.
        check_normal_buffer_roundtrip(0,2,3,4,8'hD3,3'd5,1'b1);
        check_normal_buffer_roundtrip(1,7,3,6,8'hA6,3'd4,1'b0);
        if (Buffer_Access_Error) begin $display("[FAIL] INTER buffer error"); errors=errors+1; end
        $display("[PASS] Stage-6 INTER shift/store/restore SRAM macro completed");
    end
endtask

task run_intra_case;
    begin
        $display("\n[CASE] Stage-6B INTRA timing-safe SRAM");
        apply_reset(); run_active=1;
        pulse_start(0,17,18); stream_multi_fault_model(); wait_for_done();
        run_active=0; check_common_results(MODE_INTRA,0,128);
        check_intra_buffer_roundtrip(0,0,0,3'd2,1'b1,0);
        check_intra_buffer_roundtrip(1,4,0,3'd4,1'b0,1);
        $display("[PASS] Stage-6 INTRA transpose/CW/SRAM/restore completed");
    end
endtask

task run_unrepairable_case;
    begin
        $display("\n[CASE] Stage-6B UNREPAIRABLE");
        apply_reset(); run_active=1;
        pulse_start(0,10,17); stream_multi_fault_model(); wait_for_done();
        run_active=0; check_common_results(MODE_UNREPAIRABLE,0,0);
        Normal_Model_Page_Index=0;
        Normal_Page_Offset=make_page_offset(0,0,0,0);
        Normal_Lookup_Enable=1'b1;
        #1;
        if (Normal_Access_Ready || Remap_Control_Valid || Normal_Buffer_Request_Ready) begin
            $display("[FAIL] Unrepairable mode allowed buffer access"); errors=errors+1;
        end
        Normal_Lookup_Enable=1'b0;
        $display("[PASS] Stage-6 UNREPAIRABLE access block completed");
    end
endtask

initial begin
    clk=1'b0; rst_n=1'b0; Repair_Start=1'b0;
    Base_Max_Faults=0; Inter_Max_Faults=0; Intra_Max_Faults=0;
    Fault_Bitmap_Valid=1'b0; Fault_Valid_Bitmap_In=0; Fault_Value_Bitmap_In=0;
    Bitmap_Page_Index_In=0; Bitmap_FBB_ID_In=0; Bitmap_Bank_ID_In=0; Bitmap_RG_ID_In=0;
    Normal_Lookup_Enable=1'b0; Normal_Model_Page_Index=0; Normal_Page_Offset=0;
    Intra_Logical_TWW_Addr=0;
    Normal_Buffer_Write_Enable=1'b0; Normal_Buffer_Read_Enable=1'b0;
    Normal_Buffer_Data_In=0;
    Intra_Transaction_Start=1'b0; Intra_Transaction_Direction=1'b0;
    Intra_Write_WW_Valid=1'b0; Intra_Write_WW_Data=0;
    errors=0; run_active=0; reset_run_counters();

    run_base_case();
    run_inter_case();
    run_intra_case();
    run_unrepairable_case();

    if (errors==0)
        $display("\nPASS: Stage-6B timing-safe SRAM command-pipeline regression");
    else
        $display("\nFAIL: Stage-6B SRAM command-pipeline errors=%0d", errors);
    #20; $finish;
end

endmodule
