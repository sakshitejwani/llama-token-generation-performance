import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import gc

from instrumentation import DetailedLatencyMeasurer, TimerContext


class LLaMABenchmark:
    
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.measurer = None
        self.results = {}
        
        print(f"Initializing LLaMA Benchmark")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Timestamp: {datetime.now().isoformat()}")
        
    def load_model(self):
        print("\n[1/5] Loading model and tokenizer...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.measurer = DetailedLatencyMeasurer(self.model, self.tokenizer, self.device)
            
            print("✓ Model loaded successfully")
            
            self.results['model_info'] = {
                'model_name': self.model_name,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'device': self.device,
                'dtype': str(self.model.dtype),
                'device_memory': self._get_device_memory_usage()
            }
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def _get_device_memory_usage(self) -> str:
        """Get memory usage for current device."""
        if self.device == 'cuda':
            return f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        else:
            return f"{psutil.Process().memory_info().rss / 1e9:.2f} GB"
    
    def benchmark_first_token_latency(self):
        print("\n[2/5] Benchmarking First-Token Latency (TTFT)...")
        
        test_prompts = [
            "The future of artificial intelligence is",
            "Machine learning enables computers to",
            "In the field of natural language processing,",
            "Deep neural networks have revolutionized",
            "Token generation latency is important for"
        ]
        
        ttft_results = []
        
        for prompt_idx, prompt in enumerate(test_prompts):
            print(f"  Prompt {prompt_idx + 1}/{len(test_prompts)}: '{prompt[:40]}...'")
            
            ttft_times = []
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            input_ids = inputs['input_ids']
            
            with torch.no_grad():
                for _ in range(2):
                    _ = self.model.generate(input_ids, max_new_tokens=1, do_sample=False)
            
            with torch.no_grad():
                for trial in range(4):
                    torch.cuda.synchronize() if self.device == 'cuda' else None
                    start = time.perf_counter()
                    
                    _ = self.model.generate(input_ids, max_new_tokens=1, do_sample=False)
                    
                    torch.cuda.synchronize() if self.device == 'cuda' else None
                    ttft_ms = (time.perf_counter() - start) * 1000
                    ttft_times.append(ttft_ms)
            
            ttft_array = np.array(ttft_times)
            ttft_results.append({
                'prompt': prompt[:50],
                'ttft_ms': float(np.mean(ttft_array)),
                'std_ms': float(np.std(ttft_array)),
                'min_ms': float(np.min(ttft_array)),
                'max_ms': float(np.max(ttft_array)),
                'all_trials': ttft_times
            })
        
        self.results['ttft_analysis'] = {
            'description': 'Time To First Token - latency to generate first token',
            'unit': 'milliseconds',
            'per_prompt': ttft_results,
            'overall_mean': float(np.mean([r['ttft_ms'] for r in ttft_results])),
            'overall_std': float(np.std([r['ttft_ms'] for r in ttft_results]))
        }
        
        print(f"  TTFT Mean: {self.results['ttft_analysis']['overall_mean']:.2f}ms ± {self.results['ttft_analysis']['overall_std']:.2f}ms")
    
    def benchmark_per_token_latency(self):
        print("\n[3/5] Benchmarking Per-Token Latency (PTL)...")
        
        test_cases = [
            {
                'name': 'Short output',
                'prompt': 'The evolution of artificial intelligence',
                'output_tokens': 10
            },
            {
                'name': 'Medium output',
                'prompt': 'Explain how transformer models work in detail',
                'output_tokens': 50
            },
            {
                'name': 'Longer output',
                'prompt': 'Describe the impact of neural networks on modern technology',
                'output_tokens': 100
            }
        ]
        
        ptl_results = []
        
        for test_case in test_cases:
            print(f"  {test_case['name']} ({test_case['output_tokens']} tokens)...")
            
            ptl_times = []
            inputs = self.tokenizer(test_case['prompt'], return_tensors='pt').to(self.device)
            input_ids = inputs['input_ids']
            

            with torch.no_grad():
                _ = self.model.generate(input_ids, max_new_tokens=5, do_sample=False)
            

            with torch.no_grad():
                for trial in range(4):
                    torch.cuda.synchronize() if self.device == 'cuda' else None
                    start = time.perf_counter()
                    
                    _ = self.model.generate(
                        input_ids,
                        max_new_tokens=test_case['output_tokens'],
                        do_sample=False
                    )
                    
                    torch.cuda.synchronize() if self.device == 'cuda' else None
                    elapsed = time.perf_counter() - start
                    ptl_ms = (elapsed / test_case['output_tokens']) * 1000
                    ptl_times.append(ptl_ms)
            
            ptl_array = np.array(ptl_times)
            ptl_results.append({
                'test_case': test_case['name'],
                'output_tokens': test_case['output_tokens'],
                'ptl_ms': float(np.mean(ptl_array)),
                'std_ms': float(np.std(ptl_array)),
                'min_ms': float(np.min(ptl_array)),
                'max_ms': float(np.max(ptl_array)),
                'throughput_tokens_per_sec': float(test_case['output_tokens'] / (np.mean(ptl_array) * np.mean(ptl_array) / 1000))
            })
        
        self.results['ptl_analysis'] = {
            'description': 'Per-Token Latency - average MS per token in steady state',
            'unit': 'milliseconds per token',
            'test_cases': ptl_results
        }
        
        for result in ptl_results:
            print(f"    {result['test_case']}: {result['ptl_ms']:.2f}ms/token")
    
    def benchmark_scaling_with_sequence_length(self):
        print("\n[4/5] Benchmarking Scaling with Sequence Length...")
        
        sequence_lengths = [10, 50, 100, 200]
        output_tokens = 20
        
        base_prompt = "The field of artificial intelligence has grown significantly. "
        
        scaling_results = []
        
        for seq_len_idx, seq_len in enumerate(sequence_lengths):
            num_repeats = seq_len // len(base_prompt.split()) + 1
            prompt = ' '.join(base_prompt.split() * num_repeats)
            
            tokens = self.tokenizer(prompt, return_tensors='pt')
            actual_seq_len = tokens['input_ids'].shape[1]
            
            print(f"  Sequence length {seq_len_idx + 1}/{len(sequence_lengths)}: {actual_seq_len} tokens")
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            input_ids = inputs['input_ids']
            
            ptl_times = []
            
            with torch.no_grad():
                _ = self.model.generate(input_ids, max_new_tokens=3, do_sample=False)
            
            with torch.no_grad():
                for trial in range(3):
                    torch.cuda.synchronize() if self.device == 'cuda' else None
                    start = time.perf_counter()
                    
                    _ = self.model.generate(
                        input_ids,
                        max_new_tokens=output_tokens,
                        do_sample=False
                    )
                    
                    torch.cuda.synchronize() if self.device == 'cuda' else None
                    elapsed = time.perf_counter() - start
                    ptl_ms = (elapsed / output_tokens) * 1000
                    ptl_times.append(ptl_ms)
            
            ptl_array = np.array(ptl_times)
            scaling_results.append({
                'input_seq_len': int(actual_seq_len),
                'output_tokens': output_tokens,
                'ptl_ms': float(np.mean(ptl_array)),
                'std_ms': float(np.std(ptl_array))
            })
        
        self.results['scaling_analysis'] = {
            'description': 'How per-token latency scales with input sequence length',
            'reasoning': 'Longer sequences increase KV-cache read overhead in attention',
            'results': scaling_results
        }
        
        print("  Scaling results:")
        for result in scaling_results:
            print(f"    Seq len {result['input_seq_len']:3d}: {result['ptl_ms']:.2f}ms/token")
    
    def benchmark_batch_efficiency(self):
        print("\n[5/5] Benchmarking Batch Processing Efficiency...")
        
        prompts = [
            "The future of AI",
            "Machine learning advances",
            "Neural networks explain",
            "Deep learning applications",
            "Transformer architectures"
        ]
        
        batch_results = []
        
        print("  Batch size 1 (sequential)...")
        single_times = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                start = time.perf_counter()
                
                _ = self.model.generate(inputs['input_ids'], max_new_tokens=30, do_sample=False)
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                elapsed = time.perf_counter() - start
                single_times.append(elapsed)
        
        single_total = sum(single_times)
        
        print("  Batch size 5 (batched)...")
        batch_inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start = time.perf_counter()
            
            _ = self.model.generate(
                batch_inputs['input_ids'],
                max_new_tokens=30,
                do_sample=False
            )
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            batch_time = time.perf_counter() - start
        
        speedup = single_total / batch_time
        
        self.results['batch_efficiency'] = {
            'description': 'Throughput improvement from batch processing',
            'num_prompts': len(prompts),
            'single_batch_time_sec': float(single_total),
            'batch_5_time_sec': float(batch_time),
            'speedup_factor': float(speedup),
            'per_prompt_single': float(single_total / len(prompts)),
            'per_prompt_batch': float(batch_time / len(prompts))
        }
        
        print(f"  Single batch total: {single_total:.2f}s")
        print(f"  Batch-5 total: {batch_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    def run_all_benchmarks(self):
        try:
            self.load_model()
            self.benchmark_first_token_latency()
            self.benchmark_per_token_latency()
            self.benchmark_scaling_with_sequence_length()
            self.benchmark_batch_efficiency()
            
            print("\n" + "="*60)
            print("✓ All benchmarks completed successfully!")
            print("="*60)
            
        except Exception as e:
            print(f"\n✗ Benchmark failed: {e}")
            raise
    
    def save_results(self, output_file=None):
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmarks/results/llama_latency_benchmark_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_version': '1.0',
            'python_version': '3.10+',
            'device': self.device,
            'avg_device_memory': self._get_device_memory_usage()
        }
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        return str(output_path)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("⚠️  CPU benchmarking will be SLOW (1-2 hours for complete suite)")
        print("    Consider using GPU for faster results")
    
    benchmark = LLaMABenchmark(
        model_name='meta-llama/Llama-2-7b-hf',
        device=device
    )
    
    benchmark.run_all_benchmarks()
    benchmark.save_results()
    
    print("\nNext steps:")
    print("  1. Run analysis/decomposition_analysis.py to analyze results")
    print("  2. Run analysis/scaling_analysis.py for bottleneck identification")
    print("  3. Check analysis/plots/ for visualization charts")


if __name__ == '__main__':
    main()
