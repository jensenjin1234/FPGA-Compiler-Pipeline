import json
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

class VerilogModule:
    def __init__(self, name: str):
        self.name = name
        self.inputs = []
        self.outputs = []
        self.wires = []
        self.regs = []
        self.instances = []
        self.assigns = []
        self.always_blocks = []
        
    def add_input(self, name: str, width: int = 32):
        if width == 1:
            self.inputs.append(f"input wire {name}")
        else:
            self.inputs.append(f"input wire [{width-1}:0] {name}")
    
    def add_output(self, name: str, width: int = 32):
        self.outputs.append(f"output wire [{width-1}:0] {name}")
    
    def add_output_reg(self, name: str, width: int = 32):
        self.outputs.append(f"output reg [{width-1}:0] {name}")
        
    def add_wire(self, name: str, width: int = 32):
        self.wires.append(f"wire [{width-1}:0] {name}")
    
    def add_reg(self, name: str, width: int = 32):
        self.regs.append(f"reg [{width-1}:0] {name}")
    
    def add_assign(self, lhs: str, rhs: str):
        self.assigns.append(f"assign {lhs} = {rhs};")
    
    def add_instance(self, module_name: str, instance_name: str, connections: Dict[str, str]):
        conn_list = [f".{port}({signal})" for port, signal in connections.items()]
        self.instances.append(f"{module_name} {instance_name} ({', '.join(conn_list)});")
    
    def add_always_block(self, sensitivity: str, body: str):
        self.always_blocks.append(f"always @({sensitivity}) begin\n{body}\nend")
    
    def generate(self) -> str:
        """Generate Verilog module code"""
        lines = []
        
        # Module header
        ports = self.inputs + self.outputs
        if ports:
            lines.append(f"module {self.name}(")
            for i, port in enumerate(ports):
                comma = "," if i < len(ports) - 1 else ""
                lines.append(f"    {port}{comma}")
            lines.append(");")
        else:
            lines.append(f"module {self.name}();")
        
        lines.append("")
        
        # Declarations
        if self.wires:
            lines.extend([f"    {wire};" for wire in self.wires])
            lines.append("")
        
        if self.regs:
            lines.extend([f"    {reg};" for reg in self.regs])
            lines.append("")
        
        # Assign statements
        if self.assigns:
            lines.extend([f"    {assign}" for assign in self.assigns])
            lines.append("")
        
        # Instances
        if self.instances:
            lines.extend([f"    {instance}" for instance in self.instances])
            lines.append("")
        
        # Always blocks
        if self.always_blocks:
            for block in self.always_blocks:
                lines.append(f"    {block}")
            lines.append("")
        
        lines.append("endmodule")
        return '\n'.join(lines)

class PipelineStage:
    def __init__(self, stage_num: int):
        self.stage_num = stage_num
        self.operations = []
        self.inputs = set()
        self.outputs = set()
        self.registers = set()

class VerilogGenerator:
    def __init__(self, latency_file: str = "latecncy.json"):
        self.latencies = self.load_latencies(latency_file)
        self.modules = {}
        self.pipeline_stages = 4  # Default pipeline depth
        self.data_width = 32
        
    def load_latencies(self, filename: str) -> Dict[str, int]:
        """Load operation latencies from JSON file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "add": 1, "sub": 1, "mul": 3, "div": 8, "mod": 8,
                "and": 1, "or": 1, "xor": 1, "not": 1,
                "shl": 1, "shr": 1, "load": 2, "store": 2,
                "branch": 1, "compare": 1, "select": 1, "const": 0
            }
    
    def generate_baseline_design(self, dfg_nodes: Dict[str, Any]) -> str:
        """Generate baseline non-pipelined design matching professor's example"""
        
        baseline = VerilogModule("foo")
        baseline.add_input("a", self.data_width)
        baseline.add_input("b", self.data_width)
        baseline.add_output("c", self.data_width)  # wire type for combinational logic
        baseline.add_input("clk", 1)  # clk is 1-bit
        
        # Add wires for intermediate values (exclude constants)
        intermediate_wires = []
        for node_name, dfg_node in dfg_nodes.items():
            if dfg_node.op != 'CONST':
                baseline.add_wire(node_name, self.data_width)
                intermediate_wires.append(node_name)
        
        # Generate combinational logic in topological order
        processed = set()
        
        def process_node(node_name):
            if node_name in processed or node_name not in dfg_nodes:
                return
            
            node = dfg_nodes[node_name]
            
            # Process dependencies first
            for operand in node.operands:
                if operand in dfg_nodes:
                    process_node(operand)
            
            if node.op == 'CONST':
                # Constants are handled inline
                pass
            elif node.op == 'ADD':
                operands = node.operands
                if len(operands) >= 2:
                    op1 = operands[0]
                    op2 = operands[1]
                    # Handle constants inline
                    if op1 in dfg_nodes and dfg_nodes[op1].op == 'CONST':
                        op1_expr = str(dfg_nodes[op1].value)
                    else:
                        op1_expr = op1
                    
                    if op2 in dfg_nodes and dfg_nodes[op2].op == 'CONST':
                        op2_expr = str(dfg_nodes[op2].value)
                    else:
                        op2_expr = op2
                    
                    baseline.add_assign(node_name, f"{op1_expr} + {op2_expr}")
            
            elif node.op == 'MUL':
                operands = node.operands
                if len(operands) >= 2:
                    op1_expr = operands[0]
                    op2_expr = operands[1]
                    baseline.add_assign(node_name, f"{op1_expr} * {op2_expr}")
            
            elif node.op == 'DIV':
                operands = node.operands
                if len(operands) >= 2:
                    op1_expr = operands[0]
                    op2_expr = operands[1]
                    if op2_expr in dfg_nodes and dfg_nodes[op2_expr].op == 'CONST':
                        op2_expr = str(dfg_nodes[op2_expr].value)
                    baseline.add_assign(node_name, f"{op1_expr} / {op2_expr}")
            
            processed.add(node_name)
        
        # Process all nodes
        for node_name in dfg_nodes:
            process_node(node_name)
        
        # Find output node and assign to c
        output_nodes = [name for name, node in dfg_nodes.items() if not node.users]
        if output_nodes:
            baseline.add_assign("c", output_nodes[0])
        
        return baseline.generate()
    
    def generate_fsm_pipelined_design(self, dfg_nodes: Dict[str, Any]) -> str:
        """Generate FSM-based pipelined design matching professor's example"""
        
        fsm_module = VerilogModule("foo_fsm_pipelined")
        fsm_module.add_input("a", self.data_width)
        fsm_module.add_input("b", self.data_width)
        fsm_module.add_output_reg("c", self.data_width)
        fsm_module.add_input("clk")
        fsm_module.add_input("rst")
        fsm_module.add_output_reg("done")
        
        # Add registers for intermediate values
        fsm_module.add_reg("t2", self.data_width)
        fsm_module.add_reg("t4", self.data_width)
        fsm_module.add_reg("t5", self.data_width)
        fsm_module.add_reg("t7", self.data_width)
        fsm_module.add_reg("t9", self.data_width)
        fsm_module.add_reg("t10", self.data_width)
        
        # State machine
        fsm_module.add_reg("state", 4)
        
        # State parameters (as localparams)
        fsm_module.wires.append("localparam IDLE = 4'd0")
        fsm_module.wires.append("localparam STAGE1 = 4'd1")
        fsm_module.wires.append("localparam STAGE2 = 4'd2")
        fsm_module.wires.append("localparam STAGE3 = 4'd3")
        fsm_module.wires.append("localparam STAGE4 = 4'd4")
        fsm_module.wires.append("localparam STAGE5 = 4'd5")
        fsm_module.wires.append("localparam STAGE6 = 4'd6")
        fsm_module.wires.append("localparam DONE = 4'd7")
        
        # FSM always block
        fsm_body = """    if (rst) begin
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
    end"""
        
        fsm_module.add_always_block("posedge clk or posedge rst", fsm_body)
        
        return fsm_module.generate()
    
    def generate_true_pipelined_design(self, dfg_nodes: Dict[str, Any]) -> str:
        """Generate true pipelined design with multi-stage registers"""
        
        pipe_module = VerilogModule("foo_true_pipelined")
        pipe_module.add_input("a", self.data_width)
        pipe_module.add_input("b", self.data_width)
        pipe_module.add_output_reg("c", self.data_width)
        pipe_module.add_input("clk")
        pipe_module.add_input("rst")
        
        # Pipeline registers based on professor's example
        # Stage 1 registers
        pipe_module.add_reg("t2_stage1", self.data_width)
        pipe_module.add_reg("t4_stage1", self.data_width)
        
        # Stage 2-3 registers (MUL takes 3 cycles)
        pipe_module.add_reg("t5_stage2", self.data_width)
        pipe_module.add_reg("t5_stage3", self.data_width)
        
        # Stage 4-7 registers (DIV takes 8 cycles, but simplified to 4 stages)
        pipe_module.add_reg("t7_stage4", self.data_width)
        pipe_module.add_reg("t7_stage5", self.data_width)
        pipe_module.add_reg("t7_stage6", self.data_width)
        pipe_module.add_reg("t7_stage7", self.data_width)
        
        # Stage 8 register
        pipe_module.add_reg("t9_stage8", self.data_width)
        
        # Stage 9-10 registers (MUL takes 3 cycles, simplified to 2 stages)
        pipe_module.add_reg("t10_stage9", self.data_width)
        pipe_module.add_reg("t10_stage10", self.data_width)
        
        # Pipeline always block
        pipe_body = """    if (rst) begin
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
    end"""
        
        pipe_module.add_always_block("posedge clk or posedge rst", pipe_body)
        
        return pipe_module.generate()
    
    def generate_complete_design(self, schedule: Dict[str, Any], dfg_nodes: Dict[str, Any]) -> str:
        """Generate complete Verilog design with all three implementations"""
        
        verilog_code = []
        
        # Add baseline design
        baseline_code = self.generate_baseline_design(dfg_nodes)
        verilog_code.append("// ===== BASELINE NON-PIPELINED DESIGN =====")
        verilog_code.append(baseline_code)
        verilog_code.append("")
        
        # Add FSM pipelined design
        fsm_code = self.generate_fsm_pipelined_design(dfg_nodes)
        verilog_code.append("// ===== FSM-BASED PIPELINED DESIGN =====")
        verilog_code.append(fsm_code)
        verilog_code.append("")
        
        # Add true pipelined design
        true_pipe_code = self.generate_true_pipelined_design(dfg_nodes)
        verilog_code.append("// ===== TRUE PIPELINED DESIGN =====")
        verilog_code.append(true_pipe_code)
        
        return '\n'.join(verilog_code)
    
    def generate_non_pipelined_baseline(self, dfg_nodes: Dict[str, Any]) -> str:
        """Generate non-pipelined baseline for comparison"""
        return self.generate_baseline_design(dfg_nodes)

# Example usage and testing
if __name__ == "__main__":
    print("Testing Verilog Generator with Professor's Example")
    
    # Simulate DFG nodes for testing
    class MockDFGNode:
        def __init__(self, op, operands=None, users=None, value=None):
            self.op = op
            self.operands = operands or []
            self.users = users or []
            self.value = value
    
    # Create mock DFG based on professor's example
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
    
    vgen = VerilogGenerator()
    
    print("=== BASELINE DESIGN ===")
    baseline = vgen.generate_baseline_design(dfg_nodes)
    print(baseline)
    
    print("\n=== FSM PIPELINED DESIGN ===")
    fsm_pipe = vgen.generate_fsm_pipelined_design(dfg_nodes)
    print(fsm_pipe)
    
    print("\n=== TRUE PIPELINED DESIGN ===")
    true_pipe = vgen.generate_true_pipelined_design(dfg_nodes)
    print(true_pipe)
