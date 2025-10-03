import json
import os
import sys
from typing import Dict, List, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dfg_optimizer import DFGOptimizer
from unroller import LoopUnroller
from scheduler import ParallelismAwareScheduler
from verilog_gen import VerilogGenerator

class CompilerPipeline:
    def __init__(self, latency_file: str = "latecncy.json", resource_file: str = "resources.json"):
        self.latency_file = latency_file
        self.resource_file = resource_file
        self.optimizer = DFGOptimizer(latency_file)
        self.unroller = LoopUnroller()
        self.scheduler = ParallelismAwareScheduler(resource_file, latency_file)
        self.verilog_gen = VerilogGenerator(latency_file)
        
        # Pipeline state
        self.original_ir = ""
        self.optimized_ir = ""
        self.unrolled_ir = ""
        self.schedule_results = {}
        self.verilog_code = ""
        
    def load_ir_from_file(self, filename: str) -> str:
        """Load IR from text file"""
        try:
            with open(filename, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Could not find IR file {filename}")
            return ""
    
    def save_to_file(self, content: str, filename: str) -> bool:
        """Save content to file"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error saving to {filename}: {e}")
            return False
    
    def run_complete_pipeline(self, ir_input: str, unroll_factor: int = 2, 
                            scheduling_method: str = "list") -> Dict[str, Any]:
        """Run the complete compiler pipeline"""
        results = {
            'stages': {},
            'metrics': {},
            'files_generated': []
        }
        
        print("=== Starting Compiler Pipeline ===")
        
        # Stage 1: DFG Optimization
        print("\n1. DFG Optimization...")
        self.original_ir = ir_input
        self.optimizer.parse_ir(ir_input)
        
        optimization_results = self.optimizer.optimize()
        self.optimized_ir = self.optimizer.to_ir_string()
        
        results['stages']['optimization'] = optimization_results
        results['metrics']['nodes_eliminated'] = optimization_results['nodes_eliminated']
        results['metrics']['critical_path_original'] = optimization_results['critical_path_length']
        
        print(f"   - Eliminated {optimization_results['nodes_eliminated']} nodes")
        print(f"   - Critical path: {optimization_results['critical_path_length']} cycles")
        
        # Stage 2: Loop Unrolling (if applicable)
        print("\n2. Loop Unrolling...")
        if unroll_factor > 1:
            unrolling_results = self.unroller.analyze_unrolling_impact(
                self.optimized_ir, self.optimized_ir, unroll_factor)
            self.unrolled_ir = self.unroller.apply_loop_unrolling(self.optimized_ir, unroll_factor)
            
            results['stages']['unrolling'] = unrolling_results
            results['metrics']['code_expansion'] = unrolling_results['code_expansion']
            results['metrics']['estimated_parallelism'] = unrolling_results['estimated_parallelism']
            
            print(f"   - Code expansion: {unrolling_results['code_expansion']:.2f}x")
            print(f"   - Estimated parallelism: {unrolling_results['estimated_parallelism']}")
        else:
            self.unrolled_ir = self.optimized_ir
            results['stages']['unrolling'] = {'applied': False}
        
        # Stage 3: Scheduling
        print(f"\n3. Scheduling ({scheduling_method})...")
        ir_for_scheduling = self.unrolled_ir if unroll_factor > 1 else self.optimized_ir
        self.scheduler.parse_ir(ir_for_scheduling)
        
        if scheduling_method == "list":
            self.schedule_results = self.scheduler.list_scheduling()
        elif scheduling_method == "greedy":
            self.schedule_results = self.scheduler.greedy_balancing()
        else:
            self.schedule_results = self.scheduler.list_scheduling()
        
        results['stages']['scheduling'] = self.schedule_results
        results['metrics']['schedule_length'] = self.schedule_results['schedule_length']
        results['metrics']['average_parallelism'] = self.schedule_results['average_parallelism']
        results['metrics']['resource_utilization'] = self.schedule_results['resource_utilization']
        
        print(f"   - Schedule length: {self.schedule_results['schedule_length']} cycles")
        print(f"   - Average parallelism: {self.schedule_results['average_parallelism']:.2f}")
        
        # Stage 4: Verilog Generation
        print("\n4. Verilog Generation...")
        dfg_nodes = self.optimizer.nodes
        self.verilog_code = self.verilog_gen.generate_complete_design(
            self.schedule_results, dfg_nodes)
        
        results['stages']['verilog_generation'] = {
            'modules_generated': list(self.verilog_gen.modules.keys()),
            'pipeline_stages': self.verilog_gen.pipeline_stages
        }
        
        print(f"   - Generated {len(self.verilog_gen.modules)} Verilog modules")
        
        # Save intermediate and final results
        self.save_pipeline_results(results)
        
        return results
    
    def save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Save all pipeline results to files"""
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Save optimized IR
        if self.save_to_file(self.optimized_ir, "output/optimized_ir.txt"):
            results['files_generated'].append("output/optimized_ir.txt")
        
        # Save unrolled IR (if different)
        if self.unrolled_ir != self.optimized_ir:
            if self.save_to_file(self.unrolled_ir, "output/unrolled_ir.txt"):
                results['files_generated'].append("output/unrolled_ir.txt")
        
        # Save scheduling results
        schedule_json = json.dumps(self.schedule_results, indent=2)
        if self.save_to_file(schedule_json, "output/schedule.json"):
            results['files_generated'].append("output/schedule.json")
        
        # Save Verilog code
        if self.save_to_file(self.verilog_code, "output/pipelined_design.v"):
            results['files_generated'].append("output/pipelined_design.v")
        
        # Save baseline Verilog for comparison
        baseline_verilog = self.verilog_gen.generate_non_pipelined_baseline(self.optimizer.nodes)
        if self.save_to_file(baseline_verilog, "output/baseline_design.v"):
            results['files_generated'].append("output/baseline_design.v")
        
        # Save pipeline results summary
        results_json = json.dumps(results, indent=2, default=str)
        if self.save_to_file(results_json, "output/pipeline_results.json"):
            results['files_generated'].append("output/pipeline_results.json")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("COMPILER PIPELINE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        
        # Overview
        report_lines.append("\n## OVERVIEW")
        report_lines.append(f"Original IR instructions: {len([l for l in self.original_ir.split('\n') if l.strip()])}")
        report_lines.append(f"Optimized IR instructions: {len([l for l in self.optimized_ir.split('\n') if l.strip()])}")
        if 'unrolling' in results['stages'] and results['stages']['unrolling'].get('applied', True):
            report_lines.append(f"Unrolled IR instructions: {len([l for l in self.unrolled_ir.split('\n') if l.strip()])}")
        
        # Optimization Results
        if 'optimization' in results['stages']:
            opt_results = results['stages']['optimization']
            report_lines.append("\n## DFG OPTIMIZATION")
            report_lines.append(f"Nodes eliminated: {opt_results['nodes_eliminated']}")
            report_lines.append(f"Constant folding applied: {opt_results['constant_folding_applied']}")
            report_lines.append(f"CSE applied: {opt_results['cse_applied']}")
            report_lines.append(f"DCE applied: {opt_results['dce_applied']}")
            report_lines.append(f"Critical path length: {opt_results['critical_path_length']} cycles")
            report_lines.append(f"Critical path: {' -> '.join(opt_results['critical_path'])}")
        
        # Loop Unrolling Results
        if 'unrolling' in results['stages']:
            unroll_results = results['stages']['unrolling']
            if unroll_results.get('applied', True):
                report_lines.append("\n## LOOP UNROLLING")
                report_lines.append(f"Code expansion: {unroll_results['code_expansion']:.2f}x")
                report_lines.append(f"Estimated parallelism: {unroll_results['estimated_parallelism']}")
                area_perf = unroll_results['area_vs_performance']
                report_lines.append(f"Area increase: {area_perf['area_increase']:.2f}x")
                report_lines.append(f"Potential speedup: {area_perf['potential_speedup']:.2f}x")
                report_lines.append(f"Efficiency: {area_perf['efficiency']:.2f}")
        
        # Scheduling Results
        if 'scheduling' in results['stages']:
            sched_results = results['stages']['scheduling']
            report_lines.append("\n## SCHEDULING ANALYSIS")
            report_lines.append(f"Schedule length: {sched_results['schedule_length']} cycles")
            report_lines.append(f"Average parallelism: {sched_results['average_parallelism']:.2f}")
            report_lines.append(f"Maximum parallelism: {sched_results['maximum_parallelism']}")
            
            report_lines.append("\nResource Utilization:")
            for resource, util in sched_results['resource_utilization'].items():
                report_lines.append(f"  {resource}: {util:.1%}")
            
            report_lines.append(f"\nSchedule by time:")
            for time in sorted(sched_results['schedule'].keys()):
                nodes = sched_results['schedule'][time]
                report_lines.append(f"  Cycle {time}: {', '.join(nodes)}")
        
        # Verilog Generation
        if 'verilog_generation' in results['stages']:
            verilog_results = results['stages']['verilog_generation']
            report_lines.append("\n## VERILOG GENERATION")
            report_lines.append(f"Modules generated: {', '.join(verilog_results['modules_generated'])}")
            report_lines.append(f"Pipeline stages: {verilog_results['pipeline_stages']}")
        
        # Performance Analysis
        report_lines.append("\n## PERFORMANCE ANALYSIS")
        if 'critical_path_original' in results['metrics']:
            report_lines.append(f"Original critical path: {results['metrics']['critical_path_original']} cycles")
        if 'schedule_length' in results['metrics']:
            report_lines.append(f"Scheduled length: {results['metrics']['schedule_length']} cycles")
            if 'critical_path_original' in results['metrics']:
                speedup = results['metrics']['critical_path_original'] / results['metrics']['schedule_length']
                report_lines.append(f"Theoretical speedup: {speedup:.2f}x")
        
        # Area vs Performance Tradeoffs
        report_lines.append("\n## AREA VS PERFORMANCE TRADEOFFS")
        if 'code_expansion' in results['metrics']:
            report_lines.append(f"Code expansion factor: {results['metrics']['code_expansion']:.2f}x")
        if 'average_parallelism' in results['metrics']:
            report_lines.append(f"Parallelism achieved: {results['metrics']['average_parallelism']:.2f}")
        
        report_lines.append("\n## FILES GENERATED")
        for filename in results.get('files_generated', []):
            report_lines.append(f"  {filename}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)

def create_sample_input_files():
    """Create sample input files for testing"""
    
    # Sample IR based on the professor's example
    sample_ir = """t1 = CONST 4
t2 = ADD a, t1
t3 = CONST 7
t4 = ADD b, t3
t5 = MUL t2, t4
t6 = CONST 3
t7 = DIV t5, t6
t8 = CONST 120
t9 = ADD t7, t8
t10 = MUL t9, t9"""
    
    # Create directories if they don't exist
    os.makedirs("ir", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Write sample IR file
    with open("ir/inputir.txt", "w") as f:
        f.write(sample_ir)
    
    print("Created sample input files:")
    print("  ir/inputir.txt - Sample IR code")

def run_example():
    """Run a complete example of the compiler pipeline"""
    
    # Create sample files if they don't exist
    if not os.path.exists("ir/inputir.txt"):
        create_sample_input_files()
    
    # Initialize compiler pipeline
    pipeline = CompilerPipeline()
    
    # Load input IR
    ir_code = pipeline.load_ir_from_file("ir/inputir.txt")
    if not ir_code:
        print("Error: Could not load input IR")
        return
    
    print("Loaded IR code:")
    print(ir_code)
    
    # Run pipeline with different configurations
    print("\n" + "=" * 60)
    print("RUNNING PIPELINE WITH UNROLL_FACTOR=2, LIST SCHEDULING")
    print("=" * 60)
    
    results = pipeline.run_complete_pipeline(
        ir_code, 
        unroll_factor=2, 
        scheduling_method="list"
    )
    
    # Generate and save report
    report = pipeline.generate_report(results)
    pipeline.save_to_file(report, "output/analysis_report.txt")
    
    print(f"\n{report}")
    
    return results

if __name__ == "__main__":
    # Run example pipeline
    results = run_example()
    print("\nPipeline execution completed!")
    print("Check the 'output/' directory for generated files.")
