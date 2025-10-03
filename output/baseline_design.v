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