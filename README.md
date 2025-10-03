# ECEN513 Final Exam - Hardware Compiler Pipeline

## Overview
This project implements a comprehensive compiler infrastructure for hardware generation targeting FPGA with pipelining, parallelism, and loop unrolling capabilities.

## Project Structure
```
├── main.py                 # Main entry point
├── src/                    # Source code directory
│   ├── dfg_optimizer.py   # DFG optimization (Problem 1)
│   ├── unroller.py        # Loop unrolling (Problem 2)
│   ├── scheduler.py       # Parallelism-aware scheduling (Problem 4)
│   ├── verilog_gen.py     # Pipelined Verilog backend (Problem 3)
│   └── utils.py           # Integration utilities
├── ir/                     # Intermediate representation files
│   └── inputir.txt        # Input IR code
├── tb/                     # Testbench files (Problem 5)
│   ├── tb_top.v           # Main testbench
│   └── test_vectors.txt   # Test vectors
├── output/                 # Generated output files
├── latecncy.json          # Operation latencies
└── resources.json         # Resource constraints

## Usage

### Running the Complete Pipeline
```bash
python main.py
```

### Running Individual Components
```bash
cd src
python dfg_optimizer.py    # Test DFG optimization
python scheduler.py         # Test scheduling
python verilog_gen.py      # Test Verilog generation
```

## Example Workflow

1. **Input**: Professor's C code example
   ```c
   void foo(int a, int b, int c) {
       int d, e;
       d = (a + 4) * (b + 7);
       d = d / 3;
       e = d + 120;
       c = e * e;
   }
   ```

2. **IR**: Generated intermediate representation
   ```
   t1 = CONST 4
   t2 = ADD a, t1
   t3 = CONST 7
   t4 = ADD b, t3
   t5 = MUL t2, t4
   t6 = CONST 3
   t7 = DIV t5, t6
   t8 = CONST 120
   t9 = ADD t7, t8
   t10 = MUL t9, t9
   ```

3. **Output**: Three Verilog implementations
   - Baseline (non-pipelined combinational)
   - FSM-based pipelined 
   - True 11-stage pipelined

## Features Implemented

### Problem 1: DFG Optimization (20 marks)
- [x] Constant folding
- [x] Common subexpression elimination
- [x] Dead code elimination
- [x] Critical path analysis
- [x] Dependency/hazard detection

### Problem 2: Loop Unrolling (20 marks)
- [x] Fixed-trip-count loop unrolling
- [x] Full and partial unrolling
- [x] Instruction flattening
- [x] Area vs performance analysis

### Problem 3: Pipelined Verilog Backend (20 marks)
- [x] Baseline non-pipelined design
- [x] FSM-based pipelined design
- [x] True 11-stage pipelined design
- [x] Multi-cycle operation support

### Problem 4: Parallelism-Aware Scheduling (20 marks)
- [x] List scheduling algorithm
- [x] Greedy load balancing
- [x] Resource constraint handling
- [x] JSON-configurable constraints

### Problem 5: Simulation and Verification (10 marks)
- [x] Comprehensive testbench
- [x] Input/output vector files
- [x] Three-design verification
- [x] Correctness verification

### Problem 6: Documentation (10 marks)
- [x] Algorithm descriptions
- [x] Performance analysis
- [x] Tradeoff discussions
- [x] Complete README

## Generated Verilog Implementations

### 1. Baseline (Non-pipelined)
- Combinational logic implementation
- 1 cycle latency
- Matches professor's example exactly

### 2. FSM-based Pipelined  
- State machine controlled pipeline
- 6-7 cycles per operation
- Sequential stage execution

### 3. True Pipelined
- 11-stage pipeline with register stages
- 11 cycles latency, 1 cycle throughput
- Full parallel execution

## Output Files
- `output/baseline_design.v` - Non-pipelined design
- `output/pipelined_design.v` - All three implementations
- `output/schedule.json` - Scheduling results
- `output/optimized_ir.txt` - Optimized IR
- `output/analysis_report.txt` - Performance analysis

## Dependencies
- Python 3.7+
- For Vivado simulation: Xilinx Vivado tools

## Author
Implementation for ECEN513 Final Exam
Santa Clara University, 2025
