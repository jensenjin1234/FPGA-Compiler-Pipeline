import re
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

class LoopInfo:
    def __init__(self, start_line: int, end_line: int, induction_var: str, 
                 initial_value: int, final_value: int, step: int = 1):
        self.start_line = start_line
        self.end_line = end_line
        self.induction_var = induction_var
        self.initial_value = initial_value
        self.final_value = final_value
        self.step = step
        self.trip_count = max(0, (final_value - initial_value) // step)
        self.body_instructions = []

class LoopUnroller:
    def __init__(self):
        self.loops = []
        self.ir_lines = []
        self.unrolled_ir = []
        
    def parse_ir_with_loops(self, ir_text: str) -> None:
        """Parse IR text and identify loop structures"""
        self.ir_lines = [line.strip() for line in ir_text.split('\n') if line.strip()]
        self.detect_loops()
        
    def detect_loops(self) -> None:
        """Detect simple for-loop patterns in IR"""
        i = 0
        while i < len(self.ir_lines):
            line = self.ir_lines[i]
            
            # Look for loop initialization pattern
            # Example: i = CONST 0
            if '=' in line and 'CONST' in line:
                parts = line.split('=')
                var_name = parts[0].strip()
                value_part = parts[1].strip()
                
                if 'CONST' in value_part:
                    initial_val = int(value_part.split()[1])
                    
                    # Look ahead for loop condition and body
                    loop_info = self.analyze_loop_structure(i, var_name, initial_val)
                    if loop_info:
                        self.loops.append(loop_info)
                        i = loop_info.end_line + 1
                        continue
            i += 1
    
    def analyze_loop_structure(self, start_idx: int, induction_var: str, 
                             initial_value: int) -> Optional[LoopInfo]:
        """Analyze loop structure starting from induction variable initialization"""
        # This is a simplified loop detection for demonstration
        # In practice, you'd need more sophisticated analysis
        
        # Look for loop body pattern (simplified)
        body_start = start_idx + 1
        body_end = start_idx + 1
        
        # Find potential loop body by looking for increment pattern
        for j in range(start_idx + 1, len(self.ir_lines)):
            line = self.ir_lines[j]
            if f"{induction_var} = ADD {induction_var}" in line:
                # Found increment, this might be end of loop body
                body_end = j
                break
        
        if body_end > body_start:
            # Extract trip count (simplified - assume fixed bounds)
            trip_count = 4  # Default for demonstration
            final_value = initial_value + trip_count
            
            loop_info = LoopInfo(start_idx, body_end, induction_var, 
                               initial_value, final_value, 1)
            
            # Extract body instructions
            for idx in range(body_start, body_end):
                if idx < len(self.ir_lines):
                    loop_info.body_instructions.append(self.ir_lines[idx])
            
            return loop_info
        
        return None
    
    def unroll_loop(self, loop: LoopInfo, unroll_factor: int = 0) -> List[str]:
        """Unroll a single loop with specified unroll factor"""
        if unroll_factor == 0:
            # Full unrolling
            unroll_factor = loop.trip_count
        
        unrolled_instructions = []
        
        # Determine how many times to unroll
        full_unrolls = loop.trip_count // unroll_factor
        remainder = loop.trip_count % unroll_factor
        
        iteration = 0
        
        # Generate fully unrolled iterations
        for unroll_group in range(full_unrolls):
            for unroll_iter in range(unroll_factor):
                current_iter = iteration
                
                # Generate unrolled instructions for this iteration
                for instruction in loop.body_instructions:
                    unrolled_inst = self.rename_instruction(instruction, 
                                                          loop.induction_var, 
                                                          current_iter)
                    unrolled_instructions.append(unrolled_inst)
                
                iteration += 1
        
        # Handle remainder iterations
        for rem_iter in range(remainder):
            current_iter = iteration
            
            for instruction in loop.body_instructions:
                unrolled_inst = self.rename_instruction(instruction, 
                                                      loop.induction_var, 
                                                      current_iter)
                unrolled_instructions.append(unrolled_inst)
            
            iteration += 1
        
        return unrolled_instructions
    
    def rename_instruction(self, instruction: str, induction_var: str, 
                          iteration: int) -> str:
        """Rename variables in instruction for specific loop iteration"""
        # Replace induction variable with its value
        if induction_var in instruction:
            instruction = instruction.replace(induction_var, str(iteration))
        
        # Rename temporary variables to avoid conflicts
        # Pattern: t1 -> t1_iter0, t2 -> t2_iter0, etc.
        temp_pattern = r'\bt(\d+)\b'
        
        def replace_temp(match):
            temp_num = match.group(1)
            return f"t{temp_num}_iter{iteration}"
        
        renamed_instruction = re.sub(temp_pattern, replace_temp, instruction)
        
        return renamed_instruction
    
    def create_synthetic_loop_example(self) -> str:
        """Create a synthetic loop example for demonstration"""
        return """
// Loop: for(i = 0; i < 4; i++)
i = CONST 0
loop_start:
t1 = CONST 2
t2 = MUL i, t1
t3 = ADD a, t2
t4 = STORE t3, array[i]
i_next = ADD i, 1
i = MOVE i_next
BRANCH_LT i, 4, loop_start
"""
    
    def unroll_synthetic_loop(self, unroll_factor: int = 2) -> List[str]:
        """Unroll the synthetic loop example"""
        # Manually create loop body for demonstration
        loop_body = [
            "t1 = CONST 2",
            "t2 = MUL i, t1", 
            "t3 = ADD a, t2",
            "t4 = STORE t3, array[i]"
        ]
        
        unrolled_instructions = []
        
        # Unroll 4 iterations with factor of 2
        for iteration in range(4):
            for instruction in loop_body:
                # Replace 'i' with actual iteration value
                renamed = instruction.replace('i', str(iteration))
                
                # Rename temporaries
                temp_pattern = r'\bt(\d+)\b'
                def replace_temp(match):
                    temp_num = match.group(1)
                    return f"t{temp_num}_iter{iteration}"
                
                renamed = re.sub(temp_pattern, replace_temp, renamed)
                unrolled_instructions.append(renamed)
        
        return unrolled_instructions
    
    def apply_loop_unrolling(self, ir_text: str, unroll_factor: int = 2) -> str:
        """Apply loop unrolling to entire IR"""
        self.parse_ir_with_loops(ir_text)
        
        if not self.loops:
            # No loops detected, try synthetic example
            print("No loops detected in IR, generating synthetic loop example...")
            unrolled = self.unroll_synthetic_loop(unroll_factor)
            return '\n'.join(unrolled)
        
        result_lines = []
        current_line = 0
        
        for loop in self.loops:
            # Add instructions before loop
            while current_line < loop.start_line:
                result_lines.append(self.ir_lines[current_line])
                current_line += 1
            
            # Add unrolled loop
            unrolled_loop = self.unroll_loop(loop, unroll_factor)
            result_lines.extend(unrolled_loop)
            
            # Skip original loop instructions
            current_line = loop.end_line + 1
        
        # Add remaining instructions
        while current_line < len(self.ir_lines):
            result_lines.append(self.ir_lines[current_line])
            current_line += 1
        
        return '\n'.join(result_lines)
    
    def analyze_unrolling_impact(self, original_ir: str, unrolled_ir: str, 
                               unroll_factor: int) -> Dict[str, Any]:
        """Analyze the impact of loop unrolling"""
        original_lines = len([l for l in original_ir.split('\n') if l.strip()])
        unrolled_lines = len([l for l in unrolled_ir.split('\n') if l.strip()])
        
        # Count operations
        def count_operations(ir_text):
            ops = defaultdict(int)
            for line in ir_text.split('\n'):
                if '=' in line and line.strip():
                    parts = line.split('=')[1].strip().split()
                    if parts:
                        op = parts[0]
                        ops[op] += 1
            return ops
        
        original_ops = count_operations(original_ir)
        unrolled_ops = count_operations(unrolled_ir)
        
        # Calculate parallelism potential
        parallelism_factor = self.estimate_parallelism(unrolled_ir)
        
        return {
            'original_instructions': original_lines,
            'unrolled_instructions': unrolled_lines,
            'code_expansion': unrolled_lines / max(original_lines, 1),
            'unroll_factor': unroll_factor,
            'original_operations': dict(original_ops),
            'unrolled_operations': dict(unrolled_ops),
            'estimated_parallelism': parallelism_factor,
            'area_vs_performance': {
                'area_increase': unrolled_lines / max(original_lines, 1),
                'potential_speedup': min(unroll_factor, parallelism_factor),
                'efficiency': min(unroll_factor, parallelism_factor) / (unrolled_lines / max(original_lines, 1))
            }
        }
    
    def estimate_parallelism(self, ir_text: str) -> int:
        """Estimate potential parallelism in unrolled code"""
        lines = [l.strip() for l in ir_text.split('\n') if l.strip()]
        
        # Build dependency graph
        dependencies = defaultdict(set)
        
        for line in lines:
            if '=' in line:
                parts = line.split('=')
                dest = parts[0].strip()
                rhs = parts[1].strip()
                
                # Find dependencies
                for other_line in lines:
                    if other_line != line and '=' in other_line:
                        other_dest = other_line.split('=')[0].strip()
                        if other_dest in rhs:
                            dependencies[dest].add(other_dest)
        
        # Estimate maximum parallelism using topological levels
        levels = {}
        
        def compute_level(node):
            if node in levels:
                return levels[node]
            
            if not dependencies[node]:
                levels[node] = 0
            else:
                max_dep_level = max(compute_level(dep) for dep in dependencies[node])
                levels[node] = max_dep_level + 1
            
            return levels[node]
        
        for line in lines:
            if '=' in line:
                dest = line.split('=')[0].strip()
                compute_level(dest)
        
        # Count nodes at each level
        level_counts = defaultdict(int)
        for level in levels.values():
            level_counts[level] += 1
        
        # Maximum parallelism is the maximum nodes at any level
        max_parallelism = max(level_counts.values()) if level_counts else 1
        
        return max_parallelism

# Example usage and testing
if __name__ == "__main__":
    unroller = LoopUnroller()
    
    # Test with synthetic loop
    print("=== Loop Unrolling Example ===")
    
    synthetic_loop = unroller.create_synthetic_loop_example()
    print("Original Loop IR:")
    print(synthetic_loop)
    
    # Unroll with factor 2
    unrolled_ir = unroller.unroll_synthetic_loop(2)
    print(f"\nUnrolled IR (factor=2):")
    for line in unrolled_ir:
        print(f"  {line}")
    
    # Analyze impact
    original_ir_simple = "\n".join([
        "t1 = CONST 2",
        "t2 = MUL i, t1", 
        "t3 = ADD a, t2",
        "t4 = STORE t3, array[i]"
    ])
    
    analysis = unroller.analyze_unrolling_impact(original_ir_simple, 
                                               '\n'.join(unrolled_ir), 2)
    
    print(f"\n=== Unrolling Analysis ===")
    print(f"Code expansion: {analysis['code_expansion']:.2f}x")
    print(f"Estimated parallelism: {analysis['estimated_parallelism']}")
    print(f"Area vs Performance:")
    print(f"  Area increase: {analysis['area_vs_performance']['area_increase']:.2f}x")
    print(f"  Potential speedup: {analysis['area_vs_performance']['potential_speedup']:.2f}x")
    print(f"  Efficiency: {analysis['area_vs_performance']['efficiency']:.2f}")
