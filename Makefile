# ECEN513 Final Exam - Hardware Compiler Pipeline Makefile
# Santa Clara University - Dr. Hossein Omidian

.PHONY: all run clean test help doc

# Default target
all: run

# Run the complete pipeline
run:
	@echo "Running ECEN513 Hardware Compiler Pipeline..."
	python main.py

# Run with automatic choice 1 (complete demo)
demo:
	@echo "Running complete demonstration..."
	echo "1" | python main.py

# Test individual components
test:
	@echo "Testing individual components..."
	echo "2" | python main.py

# Generate documentation
doc:
	@echo "Generating documentation..."
	echo "3" | python main.py

# Run everything (demo + test + docs)
full:
	@echo "Running full demonstration..."
	echo "4" | python main.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf output/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -f *.pyc src/*.pyc
	rm -f tb_top.vcd
	rm -f README.md

# Run individual modules for debugging
test-optimizer:
	@echo "Testing DFG Optimizer..."
	cd src && python dfg_optimizer.py

test-scheduler:
	@echo "Testing Scheduler..."
	cd src && python scheduler.py

test-unroller:
	@echo "Testing Loop Unroller..."
	cd src && python unroller.py

test-verilog:
	@echo "Testing Verilog Generator..."
	cd src && python verilog_gen.py

# Simulate with Vivado (if available)
simulate:
	@if command -v vivado >/dev/null 2>&1; then \
		echo "Running Vivado simulation..."; \
		vivado -mode batch -source simulate.tcl; \
	else \
		echo "Vivado not found. Install Xilinx Vivado for hardware simulation."; \
	fi

# Create necessary directories
setup:
	mkdir -p output
	mkdir -p ir
	mkdir -p tb

# Help
help:
	@echo "ECEN513 Hardware Compiler Pipeline - Available targets:"
	@echo ""
	@echo "  make run      - Run the complete pipeline (interactive)"
	@echo "  make demo     - Run automatic demo with professor's example"
	@echo "  make test     - Test individual components"
	@echo "  make doc      - Generate documentation"
	@echo "  make full     - Run everything (demo + test + docs)"
	@echo "  make clean    - Clean generated files"
	@echo "  make simulate - Run Vivado simulation (if available)"
	@echo "  make setup    - Create necessary directories"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Individual component testing:"
	@echo "  make test-optimizer - Test DFG optimizer only"
	@echo "  make test-scheduler - Test scheduler only"
	@echo "  make test-unroller  - Test loop unroller only"
	@echo "  make test-verilog   - Test Verilog generator only"
	@echo ""
	@echo "Usage example:"
	@echo "  make demo     # Quick demonstration"
	@echo "  make full     # Complete analysis" 