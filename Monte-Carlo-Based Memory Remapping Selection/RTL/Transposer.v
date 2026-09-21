module Transposer #(
    parameter integer WORD_COUNT  = 8,
    parameter integer WW_LENGTH   = 8,
    parameter integer TWW_LENGTH  = 8,
    parameter integer MATRIX_SIZE = WORD_COUNT*WW_LENGTH
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  Start,

    // 1: WW -> TWW, 0: TWW -> WW.
    input  wire                  Direction,
    input  wire                  Data_In_Valid,
    output wire                  Input_Ready,

    input  wire [WW_LENGTH-1:0]  WW_Data_In,
    input  wire [TWW_LENGTH-1:0] TWW_Data_In,
    output reg  [WW_LENGTH-1:0]  WW_Data_Out,
    output reg  [TWW_LENGTH-1:0] TWW_Data_Out,

    output wire                  Data_Out_Valid,
    output wire                  Busy,
    output reg                   Done
);

localparam [1:0] IDLE        = 2'd0;
localparam [1:0] LOAD        = 2'd1;
localparam [1:0] OUTPUT_DATA = 2'd2;
localparam [1:0] FINISH      = 2'd3;

reg [1:0]             State;
reg [2:0]             Load_Counter;
reg [2:0]             Output_Counter;
reg                   Direction_Latched;
reg [MATRIX_SIZE-1:0] Transposer_Storage;
integer               bit_idx;

assign Input_Ready    = (State == LOAD);
assign Data_Out_Valid = (State == OUTPUT_DATA);
assign Busy           = (State != IDLE);

always @(*) begin
    WW_Data_Out  = {WW_LENGTH{1'b0}};
    TWW_Data_Out = {TWW_LENGTH{1'b0}};

    if (State == OUTPUT_DATA) begin
        if (Direction_Latched) begin
            TWW_Data_Out = Transposer_Storage[Output_Counter*TWW_LENGTH +: TWW_LENGTH];
        end
        else begin
            for (bit_idx = 0; bit_idx < WW_LENGTH; bit_idx = bit_idx + 1) begin
                WW_Data_Out[bit_idx] = Transposer_Storage[bit_idx*WORD_COUNT + Output_Counter];
            end
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        State               <= IDLE;
        Load_Counter        <= 3'd0;
        Output_Counter      <= 3'd0;
        Direction_Latched   <= 1'b0;
        Transposer_Storage  <= {MATRIX_SIZE{1'b0}};
        Done                <= 1'b0;
    end
    else begin
        Done <= 1'b0;
        case (State)
            IDLE: begin
                Load_Counter   <= 3'd0;
                Output_Counter <= 3'd0;

                if (Start) begin
                    Direction_Latched <= Direction;
                    State             <= LOAD;
                end
            end

            LOAD: begin
                if (Data_In_Valid) begin
                    if (Direction_Latched) begin
                        // WW -> TWW: store one WW as a matrix row.
                        for (bit_idx = 0; bit_idx < WW_LENGTH; bit_idx = bit_idx + 1) begin
                            Transposer_Storage[bit_idx*WORD_COUNT + Load_Counter] <= WW_Data_In[bit_idx];
                        end
                    end
                    else begin
                        // TWW -> WW: store one TWW as a contiguous column block.
                        for (bit_idx = 0; bit_idx < TWW_LENGTH; bit_idx = bit_idx + 1) begin
                            Transposer_Storage[Load_Counter*TWW_LENGTH + bit_idx] <= TWW_Data_In[bit_idx];
                        end
                    end

                    if (Load_Counter == WORD_COUNT-1) begin
                        Load_Counter   <= 3'd0;
                        Output_Counter <= 3'd0;
                        State          <= OUTPUT_DATA;
                    end
                    else begin
                        Load_Counter <= Load_Counter + 1'b1;
                    end
                end
            end

            OUTPUT_DATA: begin
                if (Output_Counter == WORD_COUNT-1) begin
                    Output_Counter <= 3'd0;
                    State          <= FINISH;
                end
                else begin
                    Output_Counter <= Output_Counter + 1'b1;
                end
            end

            FINISH: begin
                Done  <= 1'b1;
                State <= IDLE;
            end

            default: begin
                State <= IDLE;
            end
        endcase
    end
end

endmodule
