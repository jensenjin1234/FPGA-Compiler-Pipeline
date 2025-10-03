// ===== BASELINE NON-PIPELINED DESIGN =====
module foo(
    input wire [31:0] a,
    input wire [31:0] b,
    input wire clk,
    output wire [31:0] c
);

    wire [31:0] t2;
    wire [31:0] t4;
    wire [31:0] t5;
    wire [31:0] t7;
    wire [31:0] t9;
    wire [31:0] t10;

    assign t2 = a + 4;
    assign t4 = b + 7;
    assign t5 = t2 * t4;
    assign t7 = t5 / 3;
    assign t9 = t7 + 120;
    assign t10 = t9 * t9;
    assign c = t10;

endmodule

// ===== FSM-BASED PIPELINED DESIGN =====
module foo_fsm_pipelined(
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [31:0] clk,
    input wire [31:0] rst,
    output reg [31:0] c,
    output reg [31:0] done
);

    localparam IDLE = 4'd0;
    localparam STAGE1 = 4'd1;
    localparam STAGE2 = 4'd2;
    localparam STAGE3 = 4'd3;
    localparam STAGE4 = 4'd4;
    localparam STAGE5 = 4'd5;
    localparam STAGE6 = 4'd6;
    localparam DONE = 4'd7;

    reg [31:0] t2;
    reg [31:0] t4;
    reg [31:0] t5;
    reg [31:0] t7;
    reg [31:0] t9;
    reg [31:0] t10;
    reg [3:0] state;

    always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        c <= 0;
        done <= 0;
        t2 <= 0; t4 <= 0; t5 <= 0; t7 <= 0; t9 <= 0; t10 <= 0;
    end else begin
        case (state)
            IDLE: begin
                done <= 0;
                state <= STAGE1;
            end

            STAGE1: begin
                t2 <= a + 4;
                t4 <= b + 7;
                state <= STAGE2;
            end

            STAGE2: begin
                t5 <= t2 * t4;  // MUL latency abstracted
                state <= STAGE3;
            end

            STAGE3: begin
                t7 <= t5 / 3;  // DIV latency abstracted
                state <= STAGE4;
            end

            STAGE4: begin
                t9 <= t7 + 120;
                state <= STAGE5;
            end

            STAGE5: begin
                t10 <= t9 * t9;
                state <= STAGE6;
            end

            STAGE6: begin
                c <= t10;
                state <= DONE;
            end

            DONE: begin
                done <= 1;
            end
        endcase
    end
end

endmodule

// ===== TRUE PIPELINED DESIGN =====
module foo_true_pipelined(
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [31:0] clk,
    input wire [31:0] rst,
    output reg [31:0] c
);

    reg [31:0] t2_stage1;
    reg [31:0] t4_stage1;
    reg [31:0] t5_stage2;
    reg [31:0] t5_stage3;
    reg [31:0] t7_stage4;
    reg [31:0] t7_stage5;
    reg [31:0] t7_stage6;
    reg [31:0] t7_stage7;
    reg [31:0] t9_stage8;
    reg [31:0] t10_stage9;
    reg [31:0] t10_stage10;

    always @(posedge clk or posedge rst) begin
    if (rst) begin
        t2_stage1 <= 0; t4_stage1 <= 0;
        t5_stage2 <= 0; t5_stage3 <= 0;
        t7_stage4 <= 0; t7_stage5 <= 0; t7_stage6 <= 0; t7_stage7 <= 0;
        t9_stage8 <= 0;
        t10_stage9 <= 0; t10_stage10 <= 0;
        c <= 0;
    end else begin
        // Stage 1
        t2_stage1 <= a + 4;
        t4_stage1 <= b + 7;

        // Stage 2-3 (MUL pipeline)
        t5_stage2 <= t2_stage1 * t4_stage1;
        t5_stage3 <= t5_stage2;

        // Stage 4-7 (DIV pipeline)
        t7_stage4 <= t5_stage3 / 3;
        t7_stage5 <= t7_stage4;
        t7_stage6 <= t7_stage5;
        t7_stage7 <= t7_stage6;

        // Stage 8
        t9_stage8 <= t7_stage7 + 120;

        // Stage 9-10 (MUL pipeline)
        t10_stage9 <= t9_stage8 * t9_stage8;
        t10_stage10 <= t10_stage9;

        // Stage 11 (Output)
        c <= t10_stage10;
    end
end

endmodule