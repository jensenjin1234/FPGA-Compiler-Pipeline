import json
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, deque
import heapq

class ScheduleNode:
    def __init__(self, name: str, op: str, operands: List[str], latency: int):
        self.name = name
        self.op = op
        self.operands = operands
        self.latency = latency
        self.dependencies = set()  # Nodes this depends on
        self.dependents = set()    # Nodes that depend on this
        self.scheduled_time = -1
        self.completion_time = -1
        self.resource_type = self.get_resource_type()
        self.priority = 0
        
    def get_resource_type(self) -> str:
        """Map operation to resource type"""
        op_to_resource = {
            'ADD': 'adder', 'SUB': 'adder',
            'MUL': 'multiplier',
            'DIV': 'divider', 'MOD': 'divider',
            'AND': 'logic_unit', 'OR': 'logic_unit', 'XOR': 'logic_unit', 'NOT': 'logic_unit',
            'SHL': 'logic_unit', 'SHR': 'logic_unit',
            'LOAD': 'memory_unit', 'STORE': 'memory_unit',
            'COMPARE': 'comparator', 'SELECT': 'comparator',
            'CONST': 'none'
        }
        return op_to_resource.get(self.op, 'adder')  # Default to adder
    
    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first

class Resource:
    def __init__(self, name: str, count: int):
        self.name = name
        self.count = count
        self.available = count
        self.usage_schedule = []  # List of (start_time, end_time, node_name)
        
    def is_available(self, start_time: int, duration: int) -> bool:
        """Check if resource is available for given time period"""
        end_time = start_time + duration
        available_count = self.count
        
        for sched_start, sched_end, _ in self.usage_schedule:
            if not (end_time <= sched_start or start_time >= sched_end):
                available_count -= 1
                if available_count <= 0:
                    return False
        return True
    
    def reserve(self, start_time: int, duration: int, node_name: str):
        """Reserve resource for given time period"""
        end_time = start_time + duration
        self.usage_schedule.append((start_time, end_time, node_name))
        self.usage_schedule.sort()  # Keep sorted by start time

class ParallelismAwareScheduler:
    def __init__(self, resource_file: str = "resources.json", latency_file: str = "latecncy.json"):
        self.nodes = {}
        self.resources = {}
        self.scheduled_nodes = []
        self.schedule = {}  # time -> list of nodes
        self.max_time = 0
        
        # Load configurations
        self.load_resources(resource_file)
        self.load_latencies(latency_file)
        
    def load_resources(self, filename: str):
        """Load resource constraints from JSON file"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
                
            functional_units = config.get('functional_units', {})
            for unit_name, count in functional_units.items():
                self.resources[unit_name] = Resource(unit_name, count)
                
        except FileNotFoundError:
            # Default resources if file not found
            default_resources = {
                'adder': 2, 'multiplier': 1, 'divider': 1,
                'logic_unit': 2, 'memory_unit': 1, 'comparator': 1
            }
            for unit_name, count in default_resources.items():
                self.resources[unit_name] = Resource(unit_name, count)
    
    def load_latencies(self, filename: str):
        """Load operation latencies from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.latencies = json.load(f)
        except FileNotFoundError:
            self.latencies = {
                "add": 1, "sub": 1, "mul": 3, "div": 8, "mod": 8,
                "and": 1, "or": 1, "xor": 1, "not": 1,
                "shl": 1, "shr": 1, "load": 2, "store": 2,
                "branch": 1, "compare": 1, "select": 1, "const": 0
            }
    
    def parse_dfg(self, dfg_nodes: Dict[str, Any]) -> None:
        """Parse DFG nodes from optimizer output"""
        for node_name, dfg_node in dfg_nodes.items():
            latency = self.latencies.get(dfg_node.op.lower(), 1)
            sched_node = ScheduleNode(node_name, dfg_node.op, dfg_node.operands, latency)
            self.nodes[node_name] = sched_node
        
        # Build dependency graph
        self.build_dependencies()
        
    def parse_ir(self, ir_text: str) -> None:
        """Parse IR text directly"""
        lines = [line.strip() for line in ir_text.split('\n') if line.strip()]
        
        for line in lines:
            if '=' in line:
                parts = line.split('=')
                dest = parts[0].strip()
                rhs = parts[1].strip().split()
                
                op = rhs[0]
                operands = [op.rstrip(',') for op in rhs[1:]] if len(rhs) > 1 else []
                
                latency = self.latencies.get(op.lower(), 1)
                node = ScheduleNode(dest, op, operands, latency)
                self.nodes[dest] = node
        
        self.build_dependencies()
    
    def build_dependencies(self):
        """Build dependency relationships between nodes"""
        for node in self.nodes.values():
            for operand in node.operands:
                if operand in self.nodes:
                    node.dependencies.add(operand)
                    self.nodes[operand].dependents.add(node.name)
    
    def compute_priorities(self):
        """Compute scheduling priorities using critical path"""
        # Use longest path to end as priority
        def compute_longest_path(node_name):
            if node_name not in self.nodes:
                return 0
                
            node = self.nodes[node_name]
            if node.priority > 0:  # Already computed
                return node.priority
            
            if not node.dependents:
                # Leaf node
                node.priority = node.latency
            else:
                # Max path through dependents
                max_dependent_path = 0
                for dependent in node.dependents:
                    dep_path = compute_longest_path(dependent)
                    max_dependent_path = max(max_dependent_path, dep_path)
                node.priority = node.latency + max_dependent_path
            
            return node.priority
        
        for node_name in self.nodes:
            compute_longest_path(node_name)
    
    def list_scheduling(self) -> Dict[str, Any]:
        """Perform list scheduling with resource constraints"""
        self.compute_priorities()
        
        # Initialize ready queue with nodes that have no dependencies
        ready_queue = []
        for node_name, node in self.nodes.items():
            if not node.dependencies:
                heapq.heappush(ready_queue, node)
        
        current_time = 0
        scheduled_count = 0
        
        while ready_queue or scheduled_count < len(self.nodes):
            # Schedule all ready nodes that can be scheduled at current time
            time_scheduled = []
            
            # Create a temporary queue to check what can be scheduled
            temp_queue = []
            while ready_queue:
                node = heapq.heappop(ready_queue)
                
                # Check if all dependencies are satisfied
                deps_satisfied = True
                earliest_start = current_time
                
                for dep_name in node.dependencies:
                    dep_node = self.nodes[dep_name]
                    if dep_node.completion_time == -1:
                        deps_satisfied = False
                        break
                    earliest_start = max(earliest_start, dep_node.completion_time)
                
                if deps_satisfied:
                    # Check resource availability
                    resource_type = node.resource_type
                    if resource_type != 'none' and resource_type in self.resources:
                        resource = self.resources[resource_type]
                        
                        # Find earliest time when resource is available
                        schedule_time = earliest_start
                        while not resource.is_available(schedule_time, node.latency):
                            schedule_time += 1
                        
                        # Schedule the node
                        node.scheduled_time = schedule_time
                        node.completion_time = schedule_time + node.latency
                        resource.reserve(schedule_time, node.latency, node.name)
                        
                        time_scheduled.append(node)
                        scheduled_count += 1
                        
                        # Add to schedule
                        if schedule_time not in self.schedule:
                            self.schedule[schedule_time] = []
                        self.schedule[schedule_time].append(node.name)
                        
                        self.max_time = max(self.max_time, node.completion_time)
                    else:
                        # No resource constraint (like constants)
                        node.scheduled_time = earliest_start
                        node.completion_time = earliest_start + node.latency
                        time_scheduled.append(node)
                        scheduled_count += 1
                        
                        if earliest_start not in self.schedule:
                            self.schedule[earliest_start] = []
                        self.schedule[earliest_start].append(node.name)
                        
                        self.max_time = max(self.max_time, node.completion_time)
                else:
                    temp_queue.append(node)
            
            # Re-add unscheduled nodes to ready queue
            for node in temp_queue:
                heapq.heappush(ready_queue, node)
            
            # Add newly ready nodes to queue
            for node in time_scheduled:
                for dependent_name in node.dependents:
                    dependent = self.nodes[dependent_name]
                    
                    # Check if all dependencies of dependent are now satisfied
                    all_deps_done = True
                    for dep_name in dependent.dependencies:
                        if self.nodes[dep_name].completion_time == -1:
                            all_deps_done = False
                            break
                    
                    if all_deps_done and dependent.scheduled_time == -1:
                        heapq.heappush(ready_queue, dependent)
            
            current_time += 1
            
            # Safety check to prevent infinite loops
            if current_time > 1000:
                break
        
        return self.generate_schedule_report()
    
    def greedy_balancing(self) -> Dict[str, Any]:
        """Alternative greedy load balancing scheduler"""
        self.compute_priorities()
        
        # Sort nodes by priority
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.priority, reverse=True)
        
        # Track resource usage over time
        resource_timelines = {name: [] for name in self.resources.keys()}
        
        for node in sorted_nodes:
            # Find earliest time when all dependencies are satisfied
            earliest_start = 0
            for dep_name in node.dependencies:
                dep_node = self.nodes[dep_name]
                if dep_node.completion_time != -1:
                    earliest_start = max(earliest_start, dep_node.completion_time)
            
            # Find earliest available slot for required resource
            resource_type = node.resource_type
            if resource_type != 'none' and resource_type in self.resources:
                resource = self.resources[resource_type]
                
                schedule_time = earliest_start
                while not resource.is_available(schedule_time, node.latency):
                    schedule_time += 1
                
                node.scheduled_time = schedule_time
                node.completion_time = schedule_time + node.latency
                resource.reserve(schedule_time, node.latency, node.name)
            else:
                node.scheduled_time = earliest_start
                node.completion_time = earliest_start + node.latency
            
            # Add to schedule
            if node.scheduled_time not in self.schedule:
                self.schedule[node.scheduled_time] = []
            self.schedule[node.scheduled_time].append(node.name)
            
            self.max_time = max(self.max_time, node.completion_time)
        
        return self.generate_schedule_report()
    
    def generate_schedule_report(self) -> Dict[str, Any]:
        """Generate comprehensive scheduling report"""
        # Calculate resource utilization
        resource_utilization = {}
        for res_name, resource in self.resources.items():
            total_usage = sum(end - start for start, end, _ in resource.usage_schedule)
            utilization = total_usage / (self.max_time * resource.count) if self.max_time > 0 else 0
            resource_utilization[res_name] = min(utilization, 1.0)
        
        # Calculate parallelism achieved
        parallelism_by_time = {}
        for time, nodes in self.schedule.items():
            parallelism_by_time[time] = len(nodes)
        
        avg_parallelism = sum(parallelism_by_time.values()) / len(parallelism_by_time) if parallelism_by_time else 0
        max_parallelism = max(parallelism_by_time.values()) if parallelism_by_time else 0
        
        # Identify critical path
        critical_nodes = []
        critical_length = 0
        
        for node in self.nodes.values():
            if node.completion_time == self.max_time:
                # Trace back critical path
                current = node
                path = []
                while current:
                    path.append(current.name)
                    # Find critical predecessor
                    critical_pred = None
                    for dep_name in current.dependencies:
                        dep_node = self.nodes[dep_name]
                        if dep_node.completion_time == current.scheduled_time:
                            critical_pred = dep_node
                            break
                    current = critical_pred
                
                if len(path) > len(critical_nodes):
                    critical_nodes = path[::-1]
                    critical_length = self.max_time
        
        return {
            'schedule_length': self.max_time,
            'critical_path': critical_nodes,
            'critical_path_length': critical_length,
            'resource_utilization': resource_utilization,
            'average_parallelism': avg_parallelism,
            'maximum_parallelism': max_parallelism,
            'schedule': dict(self.schedule),
            'node_schedule': {name: {'start': node.scheduled_time, 'end': node.completion_time} 
                            for name, node in self.nodes.items()},
            'resource_usage': {name: resource.usage_schedule for name, resource in self.resources.items()}
        }
    
    def ilp_scheduling(self) -> Dict[str, Any]:
        """ILP-based optimal scheduling (simplified heuristic version)"""
        # This is a simplified version - real ILP would use integer programming solver
        # We'll use a more sophisticated heuristic approach
        
        self.compute_priorities()
        
        # Create time-expanded scheduling
        max_possible_time = sum(node.latency for node in self.nodes.values())
        
        # Use dynamic programming approach for small problems
        if len(self.nodes) <= 20:
            return self.dp_optimal_scheduling(max_possible_time)
        else:
            # Fall back to list scheduling for larger problems
            return self.list_scheduling()
    
    def dp_optimal_scheduling(self, max_time: int) -> Dict[str, Any]:
        """Dynamic programming approach for optimal scheduling"""
        # Simplified DP - for demonstration purposes
        # Real implementation would be much more complex
        return self.list_scheduling()  # Fall back for now

# Example usage and testing
if __name__ == "__main__":
    # Test with example IR
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
    
    scheduler = ParallelismAwareScheduler()
    scheduler.parse_ir(ir_code)
    
    print("=== List Scheduling Results ===")
    results = scheduler.list_scheduling()
    
    print(f"Schedule Length: {results['schedule_length']} cycles")
    print(f"Critical Path: {' -> '.join(results['critical_path'])}")
    print(f"Average Parallelism: {results['average_parallelism']:.2f}")
    print(f"Maximum Parallelism: {results['maximum_parallelism']}")
    
    print(f"\nResource Utilization:")
    for resource, util in results['resource_utilization'].items():
        print(f"  {resource}: {util:.2%}")
    
    print(f"\nSchedule by Time:")
    for time in sorted(results['schedule'].keys()):
        nodes = results['schedule'][time]
        print(f"  Time {time}: {', '.join(nodes)}")
    
    # Test greedy balancing
    scheduler2 = ParallelismAwareScheduler()
    scheduler2.parse_ir(ir_code)
    
    print(f"\n=== Greedy Balancing Results ===")
    results2 = scheduler2.greedy_balancing()
    
    print(f"Schedule Length: {results2['schedule_length']} cycles")
    print(f"Average Parallelism: {results2['average_parallelism']:.2f}")
    
    print(f"\nComparison:")
    print(f"  List Scheduling: {results['schedule_length']} cycles")
    print(f"  Greedy Balancing: {results2['schedule_length']} cycles")
    speedup = results['schedule_length'] / results2['schedule_length'] if results2['schedule_length'] > 0 else 1
    print(f"  Speedup: {speedup:.2f}x")
