`timescale 1ns / 1ps

module tb_top;

    // Parameters
    parameter DATA_WIDTH = 32;
    parameter CLK_PERIOD = 10;
    
    // Testbench signals
    reg clk;
    reg rst;
    reg start;
    reg [DATA_WIDTH-1:0] a, b;
    
    // DUT outputs - Baseline
    wire [DATA_WIDTH-1:0] c_baseline;
    
    // DUT outputs - FSM Pipelined
    wire [DATA_WIDTH-1:0] c_fsm_pipelined;
    wire done_fsm;
    
    // DUT outputs - True Pipelined
    wire [DATA_WIDTH-1:0] c_true_pipelined;
    
    // Test vectors
    reg [DATA_WIDTH-1:0] test_a [0:15];
    reg [DATA_WIDTH-1:0] test_b [0:15];
    reg [DATA_WIDTH-1:0] expected_c [0:15];
    
    // Test control
    integer test_count;
    integer current_test;
    integer errors;
    integer passed_tests;
    
    // Result monitoring
    reg [DATA_WIDTH-1:0] baseline_results [0:15];
    reg [DATA_WIDTH-1:0] fsm_results [0:15];
    reg [DATA_WIDTH-1:0] true_pipe_results [0:15];
    integer result_count;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation - Baseline design
    foo dut_baseline (
        .a(a),
        .b(b),
        .clk(clk),
        .c(c_baseline)
    );
    
    // DUT instantiation - FSM Pipelined design
    foo_fsm_pipelined dut_fsm_pipelined (
        .a(a),
        .b(b),
        .clk(clk),
        .rst(rst),
        .c(c_fsm_pipelined),
        .done(done_fsm)
    );
    
    // DUT instantiation - True Pipelined design
    foo_true_pipelined dut_true_pipelined (
        .a(a),
        .b(b),
        .clk(clk),
        .rst(rst),
        .c(c_true_pipelined)
    );
    
    // Test vector initialization
    initial begin
        // Initialize test vectors based on the function: c = ((a+4)*(b+7)/3 + 120)^2
        
        // Test case 0: a=1, b=2
        test_a[0] = 32'd1;   test_b[0] = 32'd2;   
        
        // Test case 1: a=0, b=0  
        test_a[1] = 32'd0;   test_b[1] = 32'd0;   
        
        // Test case 2: a=5, b=3
        test_a[2] = 32'd5;   test_b[2] = 32'd3;   
        
        // Test case 3: a=10, b=5
        test_a[3] = 32'd10;  test_b[3] = 32'd5;   
        
        // Test case 4: a=2, b=1
        test_a[4] = 32'd2;   test_b[4] = 32'd1;   
        
        // Test case 5: a=3, b=6
        test_a[5] = 32'd3;   test_b[5] = 32'd6;   
        
        // Additional test cases
        test_a[6] = 32'd7;   test_b[6] = 32'd4;   
        test_a[7] = 32'd4;   test_b[7] = 32'd8;   
        test_a[8] = 32'd6;   test_b[8] = 32'd9;   
        test_a[9] = 32'd8;   test_b[9] = 32'd2;   
        
        // Edge cases
        test_a[10] = 32'd0;  test_b[10] = 32'd1;  
        test_a[11] = 32'd1;  test_b[11] = 32'd0;  
        test_a[12] = 32'd15; test_b[12] = 32'd10; 
        test_a[13] = 32'd20; test_b[13] = 32'd15; 
        test_a[14] = 32'd25; test_b[14] = 32'd20; 
        test_a[15] = 32'd30; test_b[15] = 32'd25; 
        
        test_count = 16;
        current_test = 0;
        errors = 0;
        passed_tests = 0;
        result_count = 0;
    end
    
    // Calculate expected results more accurately
    function [31:0] calculate_expected;
        input [31:0] in_a, in_b;
        reg [31:0] temp1, temp2, temp3, temp4, temp5;
        begin
            temp1 = in_a + 4;           // a + 4
            temp2 = in_b + 7;           // b + 7  
            temp3 = temp1 * temp2;      // (a+4) * (b+7)
            temp4 = temp3 / 3;          // ((a+4) * (b+7)) / 3
            temp5 = temp4 + 120;        // result + 120
            calculate_expected = temp5 * temp5;  // square the result
        end
    endfunction
    
    // Update expected values with calculated results
    initial begin
        #1; // Wait for initialization
        for (integer i = 0; i < test_count; i = i + 1) begin
            expected_c[i] = calculate_expected(test_a[i], test_b[i]);
            $display("Test %0d: a=%0d, b=%0d, expected=%0d", i, test_a[i], test_b[i], expected_c[i]);
        end
    end
    
    // Main test sequence
    initial begin
        $display("========================================");
        $display("Starting Verification Testbench");
        $display("Testing Professor's Example Implementations");
        $display("========================================");
        
        // Initialize
        rst = 1;
        start = 0;
        a = 0;
        b = 0;
        
        // Reset sequence
        repeat(5) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);
        
        // Run all test cases
        for (current_test = 0; current_test < test_count; current_test = current_test + 1) begin
            run_single_test(current_test);
        end
        
        // Wait for any remaining pipeline results
        repeat(20) @(posedge clk);
        
        // Final verification and reporting
        verify_all_results();
        generate_final_report();
        
        $finish;
    end
    
    // Task to run a single test case
    task run_single_test;
        input integer test_idx;
        begin
            $display("\n--- Running Test Case %0d ---", test_idx);
            $display("Inputs: a=%0d, b=%0d", test_a[test_idx], test_b[test_idx]);
            $display("Expected output: %0d", expected_c[test_idx]);
            
            // Apply inputs
            a = test_a[test_idx];
            b = test_b[test_idx];
            
            // Test baseline (combinational)
            @(posedge clk);
            baseline_results[test_idx] = c_baseline;
            $display("Baseline result: %0d", c_baseline);
            
            // Check baseline immediately
            if (c_baseline == expected_c[test_idx]) begin
                $display("✓ Baseline PASSED");
            end else begin
                $display("✗ Baseline FAILED - Expected: %0d, Got: %0d", expected_c[test_idx], c_baseline);
                errors = errors + 1;
            end
            
            // Reset and test FSM pipelined
            rst = 1;
            @(posedge clk);
            rst = 0;
            
            // Wait for FSM completion
            wait_for_fsm_result(test_idx);
            
            // Reset and test true pipelined (wait for pipeline to fill)
            rst = 1;
            repeat(2) @(posedge clk);
            rst = 0;
            
            // Wait for true pipeline result (11 cycles for full pipeline)
            repeat(12) @(posedge clk);
            true_pipe_results[test_idx] = c_true_pipelined;
            $display("True Pipeline result: %0d (after 11 cycles)", c_true_pipelined);
            
            if (c_true_pipelined == expected_c[test_idx]) begin
                $display("✓ True Pipeline PASSED");
                passed_tests = passed_tests + 1;
            end else begin
                $display("✗ True Pipeline FAILED - Expected: %0d, Got: %0d", expected_c[test_idx], c_true_pipelined);
                errors = errors + 1;
            end
            
            // Small delay between tests
            repeat(2) @(posedge clk);
        end
    endtask
    
    // Task to wait for FSM result
    task wait_for_fsm_result;
        input integer test_idx;
        integer timeout_counter;
        begin
            timeout_counter = 0;
            
            // Wait for done signal with timeout
            while (!done_fsm && timeout_counter < 50) begin
                @(posedge clk);
                timeout_counter = timeout_counter + 1;
            end
            
            if (done_fsm) begin
                fsm_results[test_idx] = c_fsm_pipelined;
                $display("FSM Pipeline result: %0d (cycles: %0d)", c_fsm_pipelined, timeout_counter);
                
                if (c_fsm_pipelined == expected_c[test_idx]) begin
                    $display("✓ FSM Pipeline PASSED");
                    passed_tests = passed_tests + 1;
                end else begin
                    $display("✗ FSM Pipeline FAILED - Expected: %0d, Got: %0d", expected_c[test_idx], c_fsm_pipelined);
                    errors = errors + 1;
                end
            end else begin
                $display("✗ FSM Pipeline TIMEOUT - No done signal received");
                fsm_results[test_idx] = 0;
                errors = errors + 1;
            end
        end
    endtask
    
    // Task to verify all results
    task verify_all_results;
        integer i;
        integer baseline_errors, fsm_errors, pipe_errors;
        begin
            $display("\n========================================");
            $display("VERIFICATION SUMMARY");
            $display("========================================");
            
            baseline_errors = 0;
            fsm_errors = 0;
            pipe_errors = 0;
            
            $display("\nDetailed Results:");
            $display("Test# | Inputs    | Expected | Baseline | FSM Pipe | True Pipe| Status");
            $display("------|-----------|----------|----------|----------|----------|--------");
            
            for (i = 0; i < test_count; i = i + 1) begin
                $display("%2d    | a=%2d b=%2d | %8d | %8d | %8d | %8d | %s", 
                    i, test_a[i], test_b[i], expected_c[i], 
                    baseline_results[i], fsm_results[i], true_pipe_results[i],
                    (baseline_results[i] == expected_c[i] && 
                     fsm_results[i] == expected_c[i] && 
                     true_pipe_results[i] == expected_c[i]) ? "PASS" : "FAIL");
                
                if (baseline_results[i] != expected_c[i]) baseline_errors = baseline_errors + 1;
                if (fsm_results[i] != expected_c[i]) fsm_errors = fsm_errors + 1;
                if (true_pipe_results[i] != expected_c[i]) pipe_errors = pipe_errors + 1;
            end
            
            $display("\nError Summary:");
            $display("Baseline errors: %0d/%0d", baseline_errors, test_count);
            $display("FSM Pipeline errors: %0d/%0d", fsm_errors, test_count);
            $display("True Pipeline errors: %0d/%0d", pipe_errors, test_count);
        end
    endtask
    
    // Task to generate final report
    task generate_final_report;
        begin
            $display("\n========================================");
            $display("FINAL TEST REPORT");
            $display("========================================");
            $display("Total tests run: %0d", test_count);
            $display("Design implementations tested: 3");
            $display("  - Baseline (combinational)");
            $display("  - FSM-based pipelined");
            $display("  - True pipelined (11-stage)");
            
            if (errors == 0) begin
                $display("\n🎉 ALL TESTS PASSED! 🎉");
                $display("All three implementations are functionally correct.");
            end else begin
                $display("\n❌ SOME TESTS FAILED");
                $display("Total errors: %0d", errors);
                $display("Please check the design implementations.");
            end
            
            $display("\nPerformance Comparison:");
            $display("- Baseline: 1 cycle (combinational)");
            $display("- FSM Pipeline: ~6-7 cycles per operation (sequential)");
            $display("- True Pipeline: 11 cycles latency, 1 cycle throughput");
            
            $display("\nSimulation completed at time: %0t", $time);
        end
    endtask
    
    // Monitor for debugging
    always @(posedge clk) begin
        if (done_fsm) begin
            $display("FSM done signal at time %0t: c=%0d", $time, c_fsm_pipelined);
        end
    end
    
    // Timeout watchdog
    initial begin
        #100000; // 100us timeout
        $display("\n⚠️  SIMULATION TIMEOUT ⚠️");
        $display("Simulation has been running for too long.");
        generate_final_report();
        $finish;
    end
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);
    end

endmodule

// Additional test utilities module
module test_utils;
    
    // Function to compute reference result
    function [31:0] compute_reference;
        input [31:0] a, b;
        reg [31:0] t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
        begin
            t1 = 4;
            t2 = a + t1;
            t3 = 7; 
            t4 = b + t3;
            t5 = t2 * t4;
            t6 = 3;
            t7 = t5 / t6;
            t8 = 120;
            t9 = t7 + t8;
            t10 = t9 * t9;
            compute_reference = t10;
        end
    endfunction
    
endmodule
