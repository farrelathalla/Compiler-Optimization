# Compiler Optimization
## 💻Multi-Paradigm Compiler Optimization Engine

A comprehensive compiler optimization system that implements five strategic algorithms to optimize C-like source code, providing detailed performance analysis and multiple optimization techniques.

## 🔧Features

- **Five Strategic Optimization Algorithms**:
  1. **Graph Analysis**: Control flow graphs, dominance analysis, loop detection
  2. **Dynamic Programming**: Optimal register allocation with spill minimization
  3. **Greedy Algorithms**: Priority-based instruction scheduling
  4. **Branch-and-Bound**: Optimal code sequence generation
  5. **Pattern Matching**: Dead code elimination and constant folding

- **Comprehensive Optimizations**:
  - Dead code elimination
  - Constant folding and propagation
  - Algebraic simplifications
  - Register allocation with minimal spilling
  - Instruction scheduling for pipeline optimization
  - Control flow optimization

## 📦Installation

### Requirements
- Python 3.8 or higher
- No external dependencies (uses only Python standard library)

### Setup
```bash
# Clone or download the optimizer
git clone <repository-url>
cd compiler-optimizer

# Make the script executable (Unix/Linux/Mac)
chmod +x compiler_optimizer.py

# Run the optimizer
python compiler_optimizer.py <input-file>
```

## 💡Usage

### Basic Usage
```bash
# Optimize a C file
python compiler_optimizer.py input.c -o optimized.c

# With verbose output
python compiler_optimizer.py input.c -o optimized.c -v

# Generate detailed report
python compiler_optimizer.py input.c -o optimized.c -r report.txt
```

### Command Line Options
- `input_file`: Source code file to optimize (.c or .txt)
- `-o, --output`: Output file for optimized code
- `-c, --config`: Configuration file (JSON format)
- `-v, --verbose`: Enable verbose output
- `-r, --report`: Generate detailed optimization report
- `--registers`: Number of available registers (default: 8)
- `--opt-level`: Optimization level 0-3 (default: 2)

### Example Files

#### Input Example (example.c):
```c
int factorial(int n) {
    int result = 1;
    int unused_var = 0;        // Dead code
    for(int i = 1; i <= n; i++) {
        result = result * i;
        unused_var = unused_var + 1;  // Dead code
    }
    return result + 0;         // Constant folding opportunity
}

int main() {
    int x = 5;
    int y = factorial(x) * 1;  // Constant folding
    return y;
}
```

#### Optimized Output:
```c
int factorial(int n) {
    int result = 1;
    for(int i = 1; i <= n; i++) {
        result = result * i;
    }
    return result;
}

int main() {
    int x = 5;
    int y = factorial(x);
    return y;
}
```

## 📝Configuration

Create a JSON configuration file to customize optimization behavior:

```json
{
    "optimization_level": 2,
    "target_architecture": "x86_64",
    "num_registers": 8,
    "enable_dead_code_elimination": true,
    "enable_constant_folding": true,
    "enable_loop_optimization": true,
    "max_branch_bound_depth": 10
}
```

## 🧪Running Tests

```bash
# Run the test suite
python compiler_optimizer.py --test

# Run example demonstration
python compiler_optimizer.py --example
```

## 🔍Optimization Techniques

### 1. Graph Analysis
- **Control Flow Graph (CFG)**: Builds a graph representation of program control flow
- **Dominance Analysis**: Computes dominance relationships between basic blocks
- **Loop Detection**: Identifies natural loops using back-edge analysis
- **Live Variable Analysis**: Determines variable lifetimes for optimization

### 2. Dynamic Programming Register Allocation
- **Interference Graph**: Constructs graph of variable conflicts
- **Optimal Allocation**: Uses DP to find minimum-cost register assignment
- **Spill Minimization**: Intelligently spills variables to memory when needed
- **Formula**: `DP[i][mask] = min(allocate_to_register, spill_to_memory)`

### 3. Greedy Instruction Scheduling
- **Dependency Graph**: Tracks data and control dependencies
- **Priority Function**: `priority = α×critical_path + β×(1/slack) + γ×resource_pressure`
- **Pipeline Optimization**: Reduces stalls and improves throughput
- **Ready Queue**: Maintains instructions ready for scheduling

### 4. Branch-and-Bound Code Generation
- **Search Space**: Explores different instruction orderings
- **Lower Bound**: Estimates minimum cost for partial solutions
- **Pruning**: Eliminates suboptimal branches early
- **Optimal Sequencing**: Finds best instruction order

### 5. Pattern Matching Optimization
- **Algebraic Simplifications**: x+0→x, x*1→x, x-x→0
- **Constant Folding**: Evaluates constant expressions at compile time
- **Dead Code Elimination**: Removes unreachable and unused code
- **Pattern Recognition**: Identifies and optimizes common patterns

## 📤Output Files

The optimizer generates several output files:

1. **Optimized Code** (.c): The optimized source code
2. **JSON Results** (.json): Detailed metrics and analysis
3. **Report File** (.txt): Human-readable optimization report

### Sample Report Output
```
=== COMPILER OPTIMIZATION REPORT ===

PERFORMANCE SUMMARY:
  Original instructions: 15
  Optimized instructions: 12
  Improvement: 20.0%
  Optimization time: 45.2 ms

ALGORITHMIC CONTRIBUTIONS:

1. Graph Analysis:
   - Basic blocks identified: 4
   - Natural loops detected: 1
   - Dominance analysis: Complete
   - Live variable analysis: Complete

2. Dynamic Programming (Register Allocation):
   - Variables allocated: 3
   - Registers available: 8
   - Spills required: 0
   - Optimal allocation: True

3. Greedy Algorithm (Instruction Scheduling):
   - Instructions scheduled: 15
   - Priority-based scheduling: Applied
   - Dependency graph: Constructed

4. Branch-and-Bound (Code Generation):
   - Sequences explored: 127
   - Optimal sequence found: True
   - Pruning applied: True

5. Pattern Matching:
   - Dead code eliminated: 3 statements
   - Constants folded: 2 expressions
   - Optimization patterns: 9 available
```

## ⚡Performance Metrics

The optimizer tracks and reports:
- **Instruction Count**: Before and after optimization
- **Dead Code**: Statements and variables eliminated
- **Constants Folded**: Expressions evaluated at compile time
- **Register Usage**: Allocation efficiency and spill count
- **Basic Blocks**: Control flow structure analysis
- **Loop Detection**: Natural loops identified
- **Optimization Time**: Processing duration

## ⚠️Limitations

- **Language Support**: Limited to C-like syntax subset
- **Data Types**: Basic types only (int, float, char)
- **Arrays**: Basic array support
- **Pointers**: Not currently supported
- **Recursion**: Limited optimization for recursive functions
- **Memory Model**: Simplified memory access model

## 🔒Algorithm Complexity

- **Lexical Analysis**: O(n) where n is source length
- **Parsing**: O(n) for most constructs
- **CFG Construction**: O(n) for n instructions
- **Dominance Analysis**: O(n²) worst case
- **Register Allocation (DP)**: O(n × 2^r) where r is registers
- **Instruction Scheduling**: O(n²) with dependency analysis
- **Branch-and-Bound**: Exponential worst case, pruned in practice

## 👤Author
### Farrel Athalla Putra
### NIM 13523118
### K2
