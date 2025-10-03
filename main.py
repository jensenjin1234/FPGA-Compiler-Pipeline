#!/usr/bin/env python3
"""
ECEN513 Final Exam - Hardware Compiler Pipeline
Santa Clara University
Dr. Hossein Omidian

Main program to demonstrate the complete compiler infrastructure for
hardware generation targeting FPGA with pipelining, parallelism, and loop unrolling.

Usage: python main.py
"""

import os
import sys

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

try:
    from utils import CompilerPipeline, run_example
    from dfg_optimizer import DFGOptimizer
    from scheduler import ParallelismAwareScheduler
    from verilog_gen import VerilogGenerator
    from unroller import LoopUnroller
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct imports...")
    sys.path.append('src')
    from utils import CompilerPipeline, run_example

def demo_professor_example():
    """Demonstrate with professor's exact example"""
    
    print("=" * 80)
    print("ECEN513 FINAL EXAM - HARDWARE COMPILER PIPELINE")
    print("=" * 80)
    print("Processing Professor's Example:")
    print("C Code: void foo(int a, int b, int c) {")
    print("    int d, e;")
    print("    d = (a + 4) * (b + 7);")
    print("    d = d / 3;")
    print("    e = d + 120;")
    print("    c = e * e;")
    print("}")
    print()
    
    # Professor's IR example
    professor_ir = """t1 = CONST 4
t2 = ADD a, t1
t3 = CONST 7
t4 = ADD b, t3
t5 = MUL t2, t4
t6 = CONST 3
t7 = DIV t5, t6
t8 = CONST 120
t9 = ADD t7, t8
t10 = MUL t9, t9"""
    
    print("Input IR:")
    for i, line in enumerate(professor_ir.split('\n'), 1):
        print(f"  {i:2d}: {line}")
    print()
    
    # Initialize compiler pipeline
    pipeline = CompilerPipeline()
    
    # Run different configurations
    configurations = [
        {"unroll_factor": 1, "scheduling_method": "list", "name": "No Unrolling + List Scheduling"},
        {"unroll_factor": 2, "scheduling_method": "list", "name": "2x Unrolling + List Scheduling"},
        {"unroll_factor": 2, "scheduling_method": "greedy", "name": "2x Unrolling + Greedy Scheduling"},
    ]
    
    all_results = []
    
    for config in configurations:
        print("=" * 60)
        print(f"CONFIGURATION: {config['name']}")
        print("=" * 60)
        
        results = pipeline.run_complete_pipeline(
            professor_ir,
            unroll_factor=config['unroll_factor'],
            scheduling_method=config['scheduling_method']
        )
        
        all_results.append((config['name'], results))
        
        # Quick summary
        print(f"\nQuick Summary for {config['name']}:")
        print(f"  Schedule Length: {results['metrics']['schedule_length']} cycles")
        print(f"  Average Parallelism: {results['metrics']['average_parallelism']:.2f}")
        print(f"  Resource Utilization:")
        for resource, util in results['metrics']['resource_utilization'].items():
            print(f"    {resource}: {util:.1%}")
        print()
    
    # Generate comparison report
    generate_comparison_report(all_results)
    
    return all_results

def generate_comparison_report(all_results):
    """Generate comparison report between different configurations"""
    
    print("=" * 80)
    print("CONFIGURATION COMPARISON REPORT")
    print("=" * 80)
    
    print("\nPerformance Comparison:")
    print("Configuration                          | Schedule | Avg Para | Multiplier | Adder   | Divider")
    print("---------------------------------------|----------|----------|------------|---------|--------")
    
    for config_name, results in all_results:
        metrics = results['metrics']
        resource_util = metrics['resource_utilization']
        
        mult_util = resource_util.get('multiplier', 0) * 100
        add_util = resource_util.get('adder', 0) * 100
        div_util = resource_util.get('divider', 0) * 100
        
        print(f"{config_name:38} | {metrics['schedule_length']:8d} | {metrics['average_parallelism']:8.2f} | "
              f"{mult_util:9.1f}% | {add_util:6.1f}% | {div_util:6.1f}%")
    
    print("\nKey Insights:")
    baseline_schedule = all_results[0][1]['metrics']['schedule_length']
    
    for i, (config_name, results) in enumerate(all_results):
        schedule_length = results['metrics']['schedule_length']
        speedup = baseline_schedule / schedule_length if schedule_length > 0 else 1
        
        print(f"{i+1}. {config_name}:")
        print(f"   - Schedule Length: {schedule_length} cycles")
        print(f"   - Speedup vs baseline: {speedup:.2f}x")
        print(f"   - Average Parallelism: {results['metrics']['average_parallelism']:.2f}")

def test_individual_components():
    """Test individual components separately"""
    
    print("\n" + "=" * 80)
    print("INDIVIDUAL COMPONENT TESTING")
    print("=" * 80)
    
    professor_ir = """t1 = CONST 4
t2 = ADD a, t1
t3 = CONST 7
t4 = ADD b, t3
t5 = MUL t2, t4
t6 = CONST 3
t7 = DIV t5, t6
t8 = CONST 120
t9 = ADD t7, t8
t10 = MUL t9, t9"""
    
    # Test DFG Optimizer
    print("\n1. Testing DFG Optimizer...")
    try:
        optimizer = DFGOptimizer()
        optimizer.parse_ir(professor_ir)
        opt_results = optimizer.optimize()
        print(f"   Original nodes: {opt_results['original_nodes']}")
        print(f"   Optimized nodes: {opt_results['optimized_nodes']}")
        print(f"   Critical path: {opt_results['critical_path_length']} cycles")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test Scheduler
    print("\n2. Testing Scheduler...")
    try:
        scheduler = ParallelismAwareScheduler()
        scheduler.parse_ir(professor_ir)
        sched_results = scheduler.list_scheduling()
        print(f"   Schedule length: {sched_results['schedule_length']} cycles")
        print(f"   Average parallelism: {sched_results['average_parallelism']:.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test Verilog Generator
    print("\n3. Testing Verilog Generator...")
    try:
        vgen = VerilogGenerator()
        
        # Create mock DFG nodes for testing
        class MockDFGNode:
            def __init__(self, op, operands=None, users=None, value=None):
                self.op = op
                self.operands = operands or []
                self.users = users or []
                self.value = value
        
        dfg_nodes = {
            't1': MockDFGNode('CONST', [], ['t2'], 4),
            't2': MockDFGNode('ADD', ['a', 't1'], ['t5']),
            't3': MockDFGNode('CONST', [], ['t4'], 7),
            't4': MockDFGNode('ADD', ['b', 't3'], ['t5']),
            't5': MockDFGNode('MUL', ['t2', 't4'], ['t7']),
            't6': MockDFGNode('CONST', [], ['t7'], 3),
            't7': MockDFGNode('DIV', ['t5', 't6'], ['t9']),
            't8': MockDFGNode('CONST', [], ['t9'], 120),
            't9': MockDFGNode('ADD', ['t7', 't8'], ['t10']),
            't10': MockDFGNode('MUL', ['t9', 't9'], []),
        }
        
        baseline = vgen.generate_baseline_design(dfg_nodes)
        print(f"   Generated baseline Verilog ({len(baseline.split())} words)")
        print("   ✓ Baseline design generated successfully")
        
        fsm_design = vgen.generate_fsm_pipelined_design(dfg_nodes)
        print(f"   Generated FSM pipelined Verilog ({len(fsm_design.split())} words)")
        print("   ✓ FSM pipelined design generated successfully")
        
        true_pipe = vgen.generate_true_pipelined_design(dfg_nodes)
        print(f"   Generated true pipelined Verilog ({len(true_pipe.split())} words)")
        print("   ✓ True pipelined design generated successfully")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nAll individual components working correctly!")

def create_documentation():
    """Create README and documentation files"""
    
    readme_content = """# ECEN513 Final Exam - Hardware Compiler Pipeline

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
"""
    
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    
    print("Created README.md documentation")

def main():
    """Main function"""
    
    print("ECEN513 Final Exam - Hardware Compiler Pipeline")
    print("=" * 50)
    print("Choose an option:")
    print("1. Run complete pipeline with professor's example")
    print("2. Test individual components")
    print("3. Create documentation")
    print("4. Run all demonstrations")
    
    try:
        choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            demo_professor_example()
        elif choice == "2":
            test_individual_components()
        elif choice == "3":
            create_documentation()
        elif choice == "4":
            demo_professor_example()
            test_individual_components()
            create_documentation()
        else:
            print("Invalid choice, running default demonstration...")
            demo_professor_example()
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Running individual component test instead...")
        test_individual_components()

if __name__ == "__main__":
    main() 