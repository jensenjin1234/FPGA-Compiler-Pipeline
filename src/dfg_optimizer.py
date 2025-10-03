import json
import re
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, deque

class DFGNode:
    def __init__(self, name: str, op: str, operands: List[str], value=None):
        self.name = name
        self.op = op
        self.operands = operands
        self.value = value  # For constants
        self.users = []  # Nodes that use this node
        self.latency = 0
        self.earliest_time = 0
        self.latest_time = float('inf')
        self.critical_path = False

    def __repr__(self):
        if self.op == 'CONST':
            return f"{self.name} = {self.op} {self.value}"
        return f"{self.name} = {self.op} {', '.join(self.operands)}"

class DFGOptimizer:
    def __init__(self, latency_file: str = "latecncy.json"):
        self.nodes = {}
        self.constants = {}
        self.latencies = self.load_latencies(latency_file)
        
    def load_latencies(self, filename: str) -> Dict[str, int]:
        """Load operation latencies from JSON file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default latencies if file not found
            return {
                "add": 1, "sub": 1, "mul": 3, "div": 8, "mod": 8,
                "and": 1, "or": 1, "xor": 1, "not": 1,
                "shl": 1, "shr": 1, "load": 2, "store": 2,
                "branch": 1, "compare": 1, "select": 1, "const": 0
            }

    def parse_ir(self, ir_text: str) -> None:
        """Parse IR text and build DFG"""
        lines = [line.strip() for line in ir_text.split('\n') if line.strip()]
        
        for line in lines:
            if '=' in line:
                parts = line.split('=')
                dest = parts[0].strip()
                rhs = parts[1].strip().split()
                
                op = rhs[0]
                operands = rhs[1:] if len(rhs) > 1 else []
                
                # Handle constants
                if op == 'CONST':
                    value = int(operands[0]) if operands else 0
                    node = DFGNode(dest, op, [], value)
                    self.constants[dest] = value
                else:
                    # Clean operands (remove commas)
                    operands = [op.rstrip(',') for op in operands]
                    node = DFGNode(dest, op, operands)
                
                # Set latency
                node.latency = self.latencies.get(op.lower(), 1)
                self.nodes[dest] = node
        
        # Build use-def chains
        self.build_dependencies()

    def build_dependencies(self):
        """Build dependency relationships between nodes"""
        for node in self.nodes.values():
            for operand in node.operands:
                if operand in self.nodes:
                    self.nodes[operand].users.append(node.name)

    def constant_folding(self) -> bool:
        """Perform constant folding optimization"""
        changed = False
        
        # Iterate until no more changes
        while True:
            round_changed = False
            
            for node_name, node in list(self.nodes.items()):
                if node.op == 'CONST':
                    continue
                    
                # Check if all operands are constants
                operand_values = []
                all_const = True
                
                for operand in node.operands:
                    if operand in self.constants:
                        operand_values.append(self.constants[operand])
                    elif operand.isdigit() or (operand.startswith('-') and operand[1:].isdigit()):
                        operand_values.append(int(operand))
                    else:
                        all_const = False
                        break
                
                if all_const and len(operand_values) > 0:
                    # Perform constant folding
                    result = self.evaluate_constant_expression(node.op, operand_values)
                    if result is not None:
                        # Replace node with constant
                        self.constants[node_name] = result
                        new_node = DFGNode(node_name, 'CONST', [], result)
                        new_node.users = node.users
                        self.nodes[node_name] = new_node
                        round_changed = True
                        changed = True
            
            if not round_changed:
                break
                
        return changed

    def evaluate_constant_expression(self, op: str, operands: List[int]) -> int:
        """Evaluate constant expressions"""
        if op == 'ADD' and len(operands) == 2:
            return operands[0] + operands[1]
        elif op == 'SUB' and len(operands) == 2:
            return operands[0] - operands[1]
        elif op == 'MUL' and len(operands) == 2:
            return operands[0] * operands[1]
        elif op == 'DIV' and len(operands) == 2 and operands[1] != 0:
            return operands[0] // operands[1]
        elif op == 'MOD' and len(operands) == 2 and operands[1] != 0:
            return operands[0] % operands[1]
        elif op == 'AND' and len(operands) == 2:
            return operands[0] & operands[1]
        elif op == 'OR' and len(operands) == 2:
            return operands[0] | operands[1]
        elif op == 'XOR' and len(operands) == 2:
            return operands[0] ^ operands[1]
        elif op == 'SHL' and len(operands) == 2:
            return operands[0] << operands[1]
        elif op == 'SHR' and len(operands) == 2:
            return operands[0] >> operands[1]
        return None

    def common_subexpression_elimination(self) -> bool:
        """Eliminate common subexpressions"""
        changed = False
        expression_map = {}
        
        for node_name, node in list(self.nodes.items()):
            if node.op == 'CONST':
                continue
                
            # Create expression signature
            expr_sig = (node.op, tuple(sorted(node.operands)) if node.op in ['ADD', 'MUL', 'AND', 'OR', 'XOR'] else tuple(node.operands))
            
            if expr_sig in expression_map:
                # Found common subexpression
                original_node = expression_map[expr_sig]
                # Redirect all users of current node to original node
                for user_name in node.users:
                    if user_name in self.nodes:
                        user_node = self.nodes[user_name]
                        user_node.operands = [original_node if op == node_name else op for op in user_node.operands]
                        self.nodes[original_node].users.append(user_name)
                
                # Remove redundant node
                del self.nodes[node_name]
                if node_name in self.constants:
                    del self.constants[node_name]
                changed = True
            else:
                expression_map[expr_sig] = node_name
                
        return changed

    def dead_code_elimination(self) -> bool:
        """Remove dead code (unused computations)"""
        changed = False
        
        # Find all live nodes (nodes that contribute to outputs)
        live_nodes = set()
        
        # Assume all nodes without users are outputs (simplification)
        outputs = [name for name, node in self.nodes.items() if not node.users]
        
        # Mark all nodes reachable from outputs as live
        def mark_live(node_name):
            if node_name in live_nodes or node_name not in self.nodes:
                return
            live_nodes.add(node_name)
            node = self.nodes[node_name]
            for operand in node.operands:
                mark_live(operand)
        
        for output in outputs:
            mark_live(output)
        
        # Remove dead nodes
        dead_nodes = set(self.nodes.keys()) - live_nodes
        for dead_node in dead_nodes:
            del self.nodes[dead_node]
            if dead_node in self.constants:
                del self.constants[dead_node]
            changed = True
            
        return changed

    def compute_critical_path(self) -> Tuple[int, List[str]]:
        """Compute critical path using topological sort and longest path"""
        # Topological sort
        in_degree = defaultdict(int)
        for node in self.nodes.values():
            for operand in node.operands:
                if operand in self.nodes:
                    in_degree[node.name] += 1
        
        queue = deque([name for name in self.nodes.keys() if in_degree[name] == 0])
        topo_order = []
        
        while queue:
            node_name = queue.popleft()
            topo_order.append(node_name)
            
            for user_name in self.nodes[node_name].users:
                if user_name in self.nodes:
                    in_degree[user_name] -= 1
                    if in_degree[user_name] == 0:
                        queue.append(user_name)
        
        # Forward pass - compute earliest times
        for node_name in topo_order:
            node = self.nodes[node_name]
            max_pred_time = 0
            for operand in node.operands:
                if operand in self.nodes:
                    pred_time = self.nodes[operand].earliest_time + self.nodes[operand].latency
                    max_pred_time = max(max_pred_time, pred_time)
            node.earliest_time = max_pred_time
        
        # Find critical path length
        max_finish_time = 0
        critical_end = None
        for node_name, node in self.nodes.items():
            finish_time = node.earliest_time + node.latency
            if finish_time > max_finish_time:
                max_finish_time = finish_time
                critical_end = node_name
        
        # Backward pass - find critical path
        critical_path = []
        current = critical_end
        
        while current:
            critical_path.append(current)
            self.nodes[current].critical_path = True
            
            # Find critical predecessor
            next_node = None
            target_time = self.nodes[current].earliest_time
            
            for operand in self.nodes[current].operands:
                if operand in self.nodes:
                    pred_node = self.nodes[operand]
                    if pred_node.earliest_time + pred_node.latency == target_time:
                        next_node = operand
                        break
            
            current = next_node
        
        critical_path.reverse()
        return max_finish_time, critical_path

    def detect_hazards(self) -> List[Dict[str, Any]]:
        """Detect data hazards and dependencies"""
        hazards = []
        
        for node_name, node in self.nodes.items():
            for operand in node.operands:
                if operand in self.nodes:
                    hazards.append({
                        'type': 'RAW',  # Read After Write
                        'producer': operand,
                        'consumer': node_name,
                        'latency': self.nodes[operand].latency
                    })
        
        return hazards

    def optimize(self) -> Dict[str, Any]:
        """Perform all optimizations and return results"""
        original_node_count = len(self.nodes)
        
        # Apply optimizations
        cf_changed = self.constant_folding()
        cse_changed = self.common_subexpression_elimination()
        dce_changed = self.dead_code_elimination()
        
        # Compute critical path and detect hazards
        critical_length, critical_path = self.compute_critical_path()
        hazards = self.detect_hazards()
        
        optimized_node_count = len(self.nodes)
        
        return {
            'original_nodes': original_node_count,
            'optimized_nodes': optimized_node_count,
            'nodes_eliminated': original_node_count - optimized_node_count,
            'constant_folding_applied': cf_changed,
            'cse_applied': cse_changed,
            'dce_applied': dce_changed,
            'critical_path_length': critical_length,
            'critical_path': critical_path,
            'hazards': hazards,
            'optimized_dfg': self.nodes
        }

    def to_ir_string(self) -> str:
        """Convert optimized DFG back to IR string"""
        lines = []
        
        # Topological sort for proper ordering
        in_degree = defaultdict(int)
        for node in self.nodes.values():
            for operand in node.operands:
                if operand in self.nodes:
                    in_degree[node.name] += 1
        
        queue = deque([name for name in self.nodes.keys() if in_degree[name] == 0])
        
        while queue:
            node_name = queue.popleft()
            node = self.nodes[node_name]
            
            if node.op == 'CONST':
                lines.append(f"{node.name} = CONST {node.value}")
            else:
                operands_str = ', '.join(node.operands)
                lines.append(f"{node.name} = {node.op} {operands_str}")
            
            for user_name in node.users:
                if user_name in self.nodes:
                    in_degree[user_name] -= 1
                    if in_degree[user_name] == 0:
                        queue.append(user_name)
        
        return '\n'.join(lines)

# Example usage and testing
if __name__ == "__main__":
    # Test with the provided example
    ir_code = """
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
    """
    
    optimizer = DFGOptimizer()
    optimizer.parse_ir(ir_code)
    
    print("Original DFG:")
    for node in optimizer.nodes.values():
        print(f"  {node}")
    
    results = optimizer.optimize()
    
    print(f"\nOptimization Results:")
    print(f"  Original nodes: {results['original_nodes']}")
    print(f"  Optimized nodes: {results['optimized_nodes']}")
    print(f"  Nodes eliminated: {results['nodes_eliminated']}")
    print(f"  Critical path length: {results['critical_path_length']}")
    print(f"  Critical path: {' -> '.join(results['critical_path'])}")
    
    print(f"\nOptimized IR:")
    print(optimizer.to_ir_string())
