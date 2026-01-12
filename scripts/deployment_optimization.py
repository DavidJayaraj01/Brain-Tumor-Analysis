"""
Model Deployment Optimization Pipeline
======================================

Optimizes trained model for clinical deployment through:
1. ONNX export with graph optimization
2. INT8 quantization for size/speed
3. TensorRT acceleration
4. Model benchmarking

Author: Brain Tumor Detection Team
Date: January 2026
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
import argparse


class ModelOptimizer:
    """
    Deployment optimization manager
    """
    
    def __init__(self, model, input_shape=(1, 3, 224, 224)):
        """
        Args:
            model: PyTorch model
            input_shape: Input tensor shape for optimization
        """
        self.model = model.eval()
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
        
    def export_to_onnx(self, output_path, opset_version=14):
        """
        Export PyTorch model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        print(f"\n📦 Exporting model to ONNX (opset {opset_version})...")
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(self.device)
        
        # Export with dynamic axes for batch dimension
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['classification', 'segmentation', 'grading'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'classification': {0: 'batch_size'},
                'segmentation': {0: 'batch_size'},
                'grading': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Get model size
        model_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✓ ONNX export successful! Size: {model_size:.2f} MB")
        
        return output_path
    
    def quantize_onnx_int8(self, onnx_path, output_path, calibration_data=None):
        """
        Quantize ONNX model to INT8
        
        Args:
            onnx_path: Input ONNX model path
            output_path: Output quantized model path
            calibration_data: Calibration dataset for quantization
        """
        print("\n🔢 Quantizing model to INT8...")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Dynamic quantization (weights to INT8)
            quantize_dynamic(
                model_input=onnx_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
                optimize_model=True
            )
            
            # Compare sizes
            original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
            quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
            reduction = (1 - quantized_size / original_size) * 100
            
            print(f"✓ Quantization successful!")
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Quantized size: {quantized_size:.2f} MB")
            print(f"  Size reduction: {reduction:.1f}%")
            
        except ImportError:
            print("⚠️  ONNX quantization tools not available. Install: pip install onnxruntime-tools")
            return None
        
        return output_path
    
    def optimize_with_tensorrt(self, onnx_path, output_path):
        """
        Optimize ONNX model with TensorRT
        
        Requires NVIDIA GPU and TensorRT installation
        
        Args:
            onnx_path: Input ONNX model
            output_path: Output TensorRT engine
        """
        print("\n⚡ Optimizing with TensorRT...")
        
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder
            with trt.Builder(TRT_LOGGER) as builder, \
                 builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                 trt.OnnxParser(network, TRT_LOGGER) as parser:
                
                # Parse ONNX model
                print("  Parsing ONNX model...")
                with open(onnx_path, 'rb') as model_file:
                    if not parser.parse(model_file.read()):
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None
                
                # Build config
                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
                
                # Enable FP16 if available
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print("  FP16 mode enabled")
                
                # Build engine
                print("  Building TensorRT engine (may take several minutes)...")
                engine = builder.build_serialized_network(network, config)
                
                # Save engine
                with open(output_path, 'wb') as f:
                    f.write(engine)
                
                print(f"✓ TensorRT optimization successful!")
                print(f"  Engine saved to: {output_path}")
                
        except ImportError:
            print("⚠️  TensorRT not available. Install NVIDIA TensorRT for GPU acceleration.")
            return None
        except Exception as e:
            print(f"⚠️  TensorRT optimization failed: {e}")
            return None
        
        return output_path
    
    def benchmark_model(self, model_path, num_iterations=100, warmup=10):
        """
        Benchmark model inference speed
        
        Args:
            model_path: Path to model (ONNX or TensorRT)
            num_iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            
        Returns:
            Dict with benchmark results
        """
        print(f"\n⏱️  Benchmarking model ({num_iterations} iterations)...")
        
        # Load ONNX Runtime session
        if model_path.endswith('.onnx'):
            session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            input_name = session.get_inputs()[0].name
            
            # Dummy input
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(warmup):
                _ = session.run(None, {input_name: dummy_input})
            
            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.time()
                _ = session.run(None, {input_name: dummy_input})
                end = time.time()
                times.append(end - start)
        
        else:  # TensorRT engine
            print("⚠️  TensorRT benchmarking requires custom implementation")
            return None
        
        # Compute statistics
        times_ms = np.array(times) * 1000
        results = {
            'mean': np.mean(times_ms),
            'std': np.std(times_ms),
            'min': np.min(times_ms),
            'max': np.max(times_ms),
            'median': np.median(times_ms),
            'throughput': 1000 / np.mean(times_ms)  # images/second
        }
        
        print(f"✓ Benchmark results:")
        print(f"  Mean inference time: {results['mean']:.2f} ms")
        print(f"  Std deviation: {results['std']:.2f} ms")
        print(f"  Min/Max: {results['min']:.2f} / {results['max']:.2f} ms")
        print(f"  Throughput: {results['throughput']:.1f} images/second")
        
        return results
    
    def compare_accuracy(self, original_model, optimized_path, test_loader):
        """
        Compare accuracy between original and optimized models
        
        Args:
            original_model: Original PyTorch model
            optimized_path: Path to optimized ONNX model
            test_loader: Test data loader
            
        Returns:
            Dict with accuracy comparison
        """
        print("\n📊 Comparing accuracy...")
        
        # Load ONNX model
        session = ort.InferenceSession(
            optimized_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Test on sample
        original_preds = []
        onnx_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                
                # Original model
                orig_out = original_model(images)
                original_preds.append(orig_out['classification'].cpu())
                
                # ONNX model
                input_name = session.get_inputs()[0].name
                onnx_out = session.run(None, {input_name: images.cpu().numpy()})
                onnx_preds.append(torch.from_numpy(onnx_out[0]))
        
        # Compute difference
        original_preds = torch.cat(original_preds)
        onnx_preds = torch.cat(onnx_preds)
        
        max_diff = torch.abs(original_preds - onnx_preds).max().item()
        mean_diff = torch.abs(original_preds - onnx_preds).mean().item()
        
        print(f"  Max prediction difference: {max_diff:.6f}")
        print(f"  Mean prediction difference: {mean_diff:.6f}")
        
        if mean_diff < 0.01:
            print("  ✓ Accuracy preserved (difference < 1%)")
        else:
            print("  ⚠️  Significant accuracy degradation detected")
        
        return {'max_diff': max_diff, 'mean_diff': mean_diff}


def main():
    """
    Main optimization pipeline
    """
    parser = argparse.ArgumentParser(description='Optimize model for deployment')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='../results/optimized_models')
    parser.add_argument('--quantize', action='store_true', help='Enable INT8 quantization')
    parser.add_argument('--tensorrt', action='store_true', help='Enable TensorRT optimization')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n🔧 Loading model...")
    from models.hybrid_architecture import H2A_MTN
    
    model = H2A_MTN(num_classes=4, num_grades=2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from: {args.checkpoint}")
    
    # Initialize optimizer
    optimizer = ModelOptimizer(model)
    
    # 1. Export to ONNX
    onnx_path = output_dir / 'model.onnx'
    optimizer.export_to_onnx(str(onnx_path))
    
    # 2. Quantize (optional)
    if args.quantize:
        quantized_path = output_dir / 'model_int8.onnx'
        optimizer.quantize_onnx_int8(str(onnx_path), str(quantized_path))
    
    # 3. TensorRT (optional)
    if args.tensorrt:
        trt_path = output_dir / 'model.trt'
        optimizer.optimize_with_tensorrt(str(onnx_path), str(trt_path))
    
    # 4. Benchmark (optional)
    if args.benchmark:
        print("\n" + "=" * 80)
        print("BENCHMARKING")
        print("=" * 80)
        
        # Benchmark original ONNX
        optimizer.benchmark_model(str(onnx_path))
        
        # Benchmark quantized (if available)
        if args.quantize:
            print("\n--- Quantized Model ---")
            optimizer.benchmark_model(str(quantized_path))
    
    print("\n" + "=" * 80)
    print("✅ Optimization pipeline completed!")
    print("=" * 80)
    print(f"\nOptimized models saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage without CLI
    print("Model Deployment Optimization Pipeline")
    print("=" * 80)
    print("\nNote: This is a template. To use:")
    print("  python deployment_optimization.py --checkpoint path/to/model.pth")
    print("  Add --quantize for INT8 quantization")
    print("  Add --tensorrt for TensorRT optimization (requires NVIDIA GPU)")
    print("  Add --benchmark to measure inference speed")
    print("\n" + "=" * 80)
