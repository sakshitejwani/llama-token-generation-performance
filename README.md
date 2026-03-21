# LLaMA Token Generation Performance: Bottleneck Identification & Optimization

**Team Members:**
- Kashish Jethmalani
- Sakshi Tejwani
- Harsita Baskaran

---

## Project Overview

This project implements comprehensive benchmarking and analysis of token-generation latency in LLaMA-2-7B, addressing all five goals from the course requirements:

1. **Benchmark Harness Design**  - Rigorous, repeatable, statistically sound methodology
2. **Latency Decomposition**  - Breakdown into 7 architectural components
3. **Scaling Analysis**  - How latency scales with sequence length
4. **Architectural Reasoning**  - Why we see these bottlenecks (KV-cache memory bandwidth)
5. **Optimization Proposal**  - Concrete KV-cache layout optimization with 15-25% improvement estimate

### Key Results

| Metric | Value |
|--------|-------|
| First-Token Latency (TTFT) | ~460 ms |
| Per-Token Latency (PTL) | ~50 ms |
| TTFT/PTL Speedup | ~9.2x (KV-cache effect) |
| Dominant Bottleneck | Attention (62% of time) |
| Sub-bottleneck | KV-cache memory reads |
| Proposed Optimization | Layout restructuring → 18-25% improvement |

---

## Directory Structure

```
PROJECT_6_LLAMA_LATENCY/
├── benchmarks/                          # Benchmarking code
│   ├── llama_latency_bench.py          # Main benchmark harness 
│   ├── instrumentation.py              # Instrumentation hooks 
│   └── results/                        # Benchmark output files
│       └── llama_latency_benchmark_*.json
│
├── analysis/                            # Analysis and visualization
│   ├── decomposition_analysis.py       # Latency breakdown analysis 
│   ├── visualization_generator.py      # Create 5 main charts 
│   ├── plots/                          # Generated visualization PNGs
│   │   ├── 01_latency_decomposition.png
│   │   ├── 02_scaling_sequence_length.png
│   │   ├── 03_kv_cache_effect.png
│   │   ├── 04_memory_bandwidth_analysis.png
│   │   └── 05_batch_efficiency.png
│   └── decomposition_analysis_report.md
│
├── optimization/                        # Optimization proposal
│   └── optimization_proposal.py        
│       → Generates: KV_CACHE_OPTIMIZATION_PROPOSAL.md
│
├── report/                              # Final deliverable
│   ├── PROJECT_6_REPORT.md            # Comprehensive final report
│   └── figures/                        # Report figures
│
├── requirements.txt                     # Python dependencies
├── README.md                           
└── SETUP.md                            # setup instructions
```

---

## Quick Start

### 1. Setup Environment

```bash
# Create Python virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace (for LLaMA model access)
huggingface-cli login
```

### 2. Run Benchmarks

```bash
cd benchmarks

python llama_latency_bench.py

# This will generate: results/llama_latency_benchmark_YYYYMMDD_HHMMSS.json
```

### 3. Analyze Results

```bash
cd analysis

# Decomposition analysis
python decomposition_analysis.py
# Output: decomposition_analysis_report.md

# Generate visualizations
python visualization_generator.py
# Output: 5 PNG charts in plots/
```

### 4. Generate Optimization Proposal

```bash
cd optimization

python optimization_proposal.py
# Output: KV_CACHE_OPTIMIZATION_PROPOSAL.md
```

## Project Implementation Summary

1. **Performance Analysis Skills**
   - Identifying bottlenecks through measurement
   - Connecting software performance to hardware limits
   - Using statistics rigorously

2. **System Understanding**
   - How transformers work internally
   - Memory hierarchy effects
   - Attention complexity

3. **Engineering Practice**
   - Reproducible benchmarking
   - Data-driven decision making
   - Optimization targeting

4. **Communication**
   - Visualizing complex data
   - Explaining technical concepts
   - Presenting concrete proposals

---

## Appendices

### A. Model Architecture (LLaMA-2-7B)
- Parameters: 4,096 hidden dimension
- Heads: 32 attention heads (128 dims each)
- Layers: 32 transformer blocks
- Vocab: 32,000 tokens
- Max sequence: 4,096 tokens

### B. Experimental Setup
- Device: CPU (Intel Core i9)
- OS: Windows 11
- Python: 3.10+
- PyTorch: 2.0.0
- Transformers: 4.35.0

### C. Abbreviations
- TTFT: Time To First Token
- PTL: Per-Token Latency
- KV: Key-Value (attention mechanism)
- MLP: Multi-Layer Perceptron
- QKV: Query-Key-Value projection
- FLOPs: Floating-point operations
- GB/s: Gigabytes per second

---

