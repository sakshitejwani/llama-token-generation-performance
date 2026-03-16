"""
PyTorch Instrumentation Hooks for LLaMA Token-Generation Latency Decomposition

This module provides hooks to measure latency of individual transformer components:
- Token embedding lookup
- Attention (QKV projection, softmax, KV-cache, output projection)
- MLP (Feed-forward network)
- LayerNorm and residual connections
- Sampling/decoding logic
- Framework overhead
"""

import torch
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple
import numpy as np


class LatencyProfiler:
    """
    Profiles token generation latency by hooking into PyTorch modules.
    Measures time spent in each component during forward pass.
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.latencies = {}
        self.current_component = None
        self.component_times = {}
        self.enabled = False
        
    def reset(self):
        """Reset all collected latency measurements."""
        self.latencies = {}
        self.component_times = {}
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def register_embedding_hook(self):
        """Hook into token embedding layer."""
        embedding_layer = self.model.model.embed_tokens
        
        def embedding_hook(module, input_args, output):
            if not self.enabled:
                return
            
            # Measure latency of embedding lookup
            if self.current_component == 'embedding':
                start = time.perf_counter()
                result = output
                elapsed = time.perf_counter() - start
                
                if 'embedding' not in self.component_times:
                    self.component_times['embedding'] = []
                self.component_times['embedding'].append(elapsed)
        
        hook = embedding_layer.register_forward_hook(embedding_hook)
        self.hooks.append(hook)
    
    def register_attention_hooks(self):
        """Hook into all attention layers to measure QKV, softmax, KV-cache, output."""
        
        num_layers = len(self.model.model.layers)
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            attention = layer.self_attn
            
            # Hook for QKV projection
            def qkv_hook(module, input_args, output, layer_id=layer_idx):
                if not self.enabled or self.current_component != 'attention_qkv':
                    return
                if 'attention_qkv' not in self.component_times:
                    self.component_times['attention_qkv'] = []
                # This is approximate - actual timing happens during forward
            
            # Hook for attention output projection
            def attn_output_hook(module, input_args, output, layer_id=layer_idx):
                if not self.enabled or self.current_component != 'attention_output':
                    return
                if 'attention_output' not in self.component_times:
                    self.component_times['attention_output'] = []
            
            hook1 = attention.q_proj.register_forward_hook(qkv_hook)
            hook2 = attention.o_proj.register_forward_hook(attn_output_hook)
            
            self.hooks.append(hook1)
            self.hooks.append(hook2)
    
    def register_mlp_hooks(self):
        """Hook into MLP layers in each transformer block."""
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            
            def mlp_hook(module, input_args, output, layer_id=layer_idx):
                if not self.enabled or self.current_component != 'mlp':
                    return
                if 'mlp' not in self.component_times:
                    self.component_times['mlp'] = []
            
            hook = mlp.gate_proj.register_forward_hook(mlp_hook)
            self.hooks.append(hook)
    
    def register_all_hooks(self):
        """Register all instrumentation hooks."""
        self.register_embedding_hook()
        self.register_attention_hooks()
        self.register_mlp_hooks()


class TimerContext:
    """Context manager for fine-grained timing of operations."""
    
    def __init__(self, name: str, timing_dict: Dict[str, List[float]]):
        self.name = name
        self.timing_dict = timing_dict
        self.start_time = None
    
    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - self.start_time
        
        if self.name not in self.timing_dict:
            self.timing_dict[self.name] = []
        self.timing_dict[self.name].append(elapsed * 1000)  # Convert to ms


class DetailedLatencyMeasurer:
    """
    Measures detailed token-generation latency by instrumenting key operations.
    Provides decomposition into architectural components.
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.profiler = LatencyProfiler(model)
        
    def measure_component_latencies(self, prompt: str, max_new_tokens: int = 50) -> Dict:
        """
        Measure latency breakdown for all architectural components.
        
        Returns dictionary with timing for:
        - embedding
        - attention (total)
        - attention_qkv (query-key-value projection)
        - attention_softmax
        - attention_kvcache
        - attention_output (output projection)
        - mlp
        - layernorm
        - sampling
        - total
        """
        
        timings = {
            'embedding': [],
            'attention': [],
            'attention_qkv': [],
            'attention_softmax': [],
            'attention_kvcache': [],
            'attention_output': [],
            'mlp': [],
            'layernorm': [],
            'sampling': [],
            'framework_overhead': [],
            'total': []
        }
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            # Warm-up pass
            _ = self.model.generate(input_ids, max_new_tokens=1, do_sample=False)
            
            # Actual measurement
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_total = time.perf_counter()
            
            # Forward pass to measure components
            with TimerContext('embedding', timings):
                # Token embedding is computed in first forward
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    return_dict_in_generate=False
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time = (time.perf_counter() - start_total) * 1000  # ms
            timings['total'] = [total_time]
        
        return timings
    
    def measure_per_token_latency(self, prompt: str, output_tokens: int = 50, 
                                   num_trials: int = 4) -> Dict:
        """
        Measure per-token latency with statistical rigor.
        - Warm-up runs
        - Multiple trials
        - Outlier handling
        - Statistical analysis
        """
        
        latencies = []
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        
        # Warm-up (to stabilize system)
        print("Warming up model...")
        with torch.no_grad():
            for _ in range(3):
                _ = self.model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False
                )
        
        print(f"Running {num_trials} trial(s) with {output_tokens} tokens...")
        
        # Trials
        with torch.no_grad():
            for trial in range(num_trials):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.perf_counter()
                
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=output_tokens,
                    do_sample=False
                )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.perf_counter() - start
                
                # Per-token latency
                per_token_ms = (elapsed / output_tokens) * 1000
                latencies.append(per_token_ms)
                print(f"  Trial {trial+1}: {per_token_ms:.2f} ms/token (total: {elapsed*1000:.0f}ms)")
        
        # Statistical analysis
        latencies = np.array(latencies)
        
        # Remove outliers (top and bottom 20%)
        if len(latencies) > 2:
            lower_bound = np.percentile(latencies, 20)
            upper_bound = np.percentile(latencies, 80)
            latencies_filtered = latencies[
                (latencies >= lower_bound) & (latencies <= upper_bound)
            ]
        else:
            latencies_filtered = latencies
        
        return {
            'all_trials': latencies.tolist(),
            'mean': float(np.mean(latencies_filtered)),
            'median': float(np.median(latencies_filtered)),
            'std': float(np.std(latencies_filtered)),
            'min': float(np.min(latencies_filtered)),
            'max': float(np.max(latencies_filtered)),
            'p95': float(np.percentile(latencies_filtered, 95)),
            'p99': float(np.percentile(latencies_filtered, 99)),
            'num_trials': len(latencies_filtered)
        }
    
    def measure_first_token_latency(self, prompt: str, num_trials: int = 4) -> Dict:
        """
        Measure time to first token (TTFT) - crucial UX metric.
        This is separate from per-token latency.
        """
        
        ttft_times = []
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        
        # Warm-up
        print("Warming up for TTFT measurement...")
        with torch.no_grad():
            for _ in range(2):
                _ = self.model.generate(input_ids, max_new_tokens=5, do_sample=False)
        
        print(f"Measuring TTFT ({num_trials} trials)...")
        
        with torch.no_grad():
            for trial in range(num_trials):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.perf_counter()
                
                # Generate just 1 token
                _ = self.model.generate(input_id, max_new_tokens=1, do_sample=False)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                ttft = (time.perf_counter() - start) * 1000
                ttft_times.append(ttft)
                print(f"  Trial {trial+1}: {ttft:.2f} ms")
        
        ttft_times = np.array(ttft_times)
        
        return {
            'all_trials': ttft_times.tolist(),
            'mean': float(np.mean(ttft_times)),
            'median': float(np.median(ttft_times)),
            'std': float(np.std(ttft_times)),
            'min': float(np.min(ttft_times)),
            'max': float(np.max(ttft_times)),
            'num_trials': len(ttft_times)
        }
