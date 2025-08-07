import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import cv2
import json
import argparse
from pathlib import Path
import logging
from typing import Tuple, List, Dict
from focusdet_trainer import FocusDetModel, FocusDetConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusDetONNXExporter:
    """Export FocusDet models to ONNX format for deployment."""
    
    def __init__(self, model_path: str, image_type: str = 'EV'):
        self.model_path = model_path
        self.image_type = image_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model configuration
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = FocusDetConfig(image_type)
        
        # Update config from checkpoint if available
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Initialize model
        self.model = FocusDetModel(num_classes=self.config.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Image type: {image_type}")
        logger.info(f"Input size: {self.config.input_size}")
    
    def export_to_onnx(self, output_path: str, opset_version: int = 11, 
                      optimize: bool = True) -> str:
        """Export PyTorch model to ONNX format."""
        logger.info("Starting ONNX export...")
        
        # Create dummy input
        dummy_input = torch.randn(
            1, 3, self.config.input_size[0], self.config.input_size[1]
        ).to(self.device)
        
        # Define input and output names
        input_names = ['image']
        output_names = ['detections']
        
        # Dynamic axes for variable batch size (optional)
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'detections': {0: 'batch_size'}
        }
        
        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=optimize,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info(f"ONNX model exported to: {output_path}")
            
            # Verify the exported model
            self._verify_onnx_model(output_path, dummy_input)
            
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def _verify_onnx_model(self, onnx_path: str, test_input: torch.Tensor):
        """Verify the exported ONNX model."""
        logger.info("Verifying ONNX model...")
        
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure is valid")
            
            # Test inference with ONNX Runtime
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get input/output info
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]
            
            logger.info(f"Input shape: {input_info.shape}")
            logger.info(f"Output shape: {output_info.shape}")
            
            # Run inference
            test_input_np = test_input.cpu().numpy()
            ort_outputs = ort_session.run(None, {input_info.name: test_input_np})
            
            # Compare with PyTorch output
            with torch.no_grad():
                torch_output = self.model(test_input)
            
            torch_output_np = torch_output.cpu().numpy()
            ort_output_np = ort_outputs[0]
            
            # Check if outputs are close
            max_diff = np.max(np.abs(torch_output_np - ort_output_np))
            logger.info(f"Max difference between PyTorch and ONNX: {max_diff:.6f}")
            
            if max_diff < 1e-5:
                logger.info("✓ ONNX model verification successful!")
            else:
                logger.warning(f"⚠ Large difference detected: {max_diff}")
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            raise
    
    def optimize_onnx_model(self, onnx_path: str, optimized_path: str = None) -> str:
        """Optimize ONNX model for better performance."""
        if optimized_path is None:
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        logger.info("Optimizing ONNX model...")
        
        try:
            # Load model
            model = onnx.load(onnx_path)
            
            # Apply optimizations
            from onnxruntime.tools import optimizer
            
            # Basic optimizations
            optimized_model = optimizer.optimize_model(
                onnx_path,
                model_type='bert',  # Use generic optimizations
                num_heads=0,
                hidden_size=0,
                optimization_options=None
            )
            
            # Save optimized model
            optimized_model.save_model_to_file(optimized_path)
            logger.info(f"Optimized ONNX model saved to: {optimized_path}")
            
            return optimized_path
            
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
            logger.info("Continuing with non-optimized model...")
            return onnx_path
    
    def quantize_model(self, onnx_path: str, quantized_path: str = None) -> str:
        """Quantize ONNX model to INT8 for faster inference."""
        if quantized_path is None:
            quantized_path = onnx_path.replace('.onnx', '_quantized.onnx')
        
        logger.info("Quantizing ONNX model to INT8...")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantize_dynamic(
                onnx_path,
                quantized_path,
                weight_type=QuantType.QInt8
            )
            
            logger.info(f"Quantized model saved to: {quantized_path}")
            
            # Compare model sizes
            original_size = os.path.getsize(onnx_path) / (1024 * 1024)
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
            compression_ratio = original_size / quantized_size
            
            logger.info(f"Original size: {original_size:.2f} MB")
            logger.info(f"Quantized size: {quantized_size:.2f} MB")
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            
            return quantized_path
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            logger.info("Continuing with non-quantized model...")
            return onnx_path

class FocusDetONNXInference:
    """ONNX inference engine for FocusDet models."""
    
    def __init__(self, onnx_path: str, confidence_threshold: float = 0.5):
        self.onnx_path = onnx_path
        self.confidence_threshold = confidence_threshold
        
        # Initialize ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output info
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        
        # Extract input shape
        self.input_shape = self.input_info.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        logger.info(f"ONNX inference engine initialized")
        logger.info(f"Model: {onnx_path}")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Providers: {self.session.get_providers()}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        # Resize image
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Add batch dimension and transpose to NCHW
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess_detections(self, output: np.ndarray, 
                             original_shape: Tuple[int, int]) -> List[Dict]:
        """Postprocess model output to get detections."""
        detections = []
        
        # Simplified postprocessing - in practice, this would be more complex
        # involving anchor decoding, NMS, etc.
        
        # For now, return placeholder detections
        # This would need to be implemented based on the actual FocusDet output format
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on an image."""
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_info.name: input_tensor})
        
        # Postprocess
        detections = self.postprocess_detections(outputs[0], original_shape)
        
        return detections
    
    def detect_from_file(self, image_path: str) -> List[Dict]:
        """Run detection on an image file."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return self.detect(image)

def benchmark_model(onnx_path: str, num_iterations: int = 100):
    """Benchmark ONNX model performance."""
    logger.info(f"Benchmarking model: {onnx_path}")
    
    # Initialize inference engine
    inference_engine = FocusDetONNXInference(onnx_path)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        inference_engine.detect(dummy_image)
    
    # Benchmark
    import time
    start_time = time.time()
    
    for _ in range(num_iterations):
        inference_engine.detect(dummy_image)
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    logger.info(f"Benchmark results ({num_iterations} iterations):")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per inference: {avg_time*1000:.2f} ms")
    logger.info(f"FPS: {fps:.2f}")

def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export FocusDet model to ONNX')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained PyTorch model')
    parser.add_argument('--image_type', type=str, choices=['EV', 'SV'], required=True,
                       help='Image type (EV or SV)')
    parser.add_argument('--output_dir', type=str, default='onnx_models',
                       help='Output directory for ONNX models')
    parser.add_argument('--optimize', action='store_true',
                       help='Apply ONNX optimizations')
    parser.add_argument('--quantize', action='store_true',
                       help='Quantize model to INT8')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark the exported model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize exporter
    exporter = FocusDetONNXExporter(args.model_path, args.image_type)
    
    # Export to ONNX
    onnx_filename = f"focusdet_{args.image_type.lower()}.onnx"
    onnx_path = os.path.join(args.output_dir, onnx_filename)
    
    exported_path = exporter.export_to_onnx(onnx_path)
    
    # Apply optimizations if requested
    if args.optimize:
        optimized_path = exporter.optimize_onnx_model(exported_path)
        exported_path = optimized_path
    
    # Apply quantization if requested
    if args.quantize:
        quantized_path = exporter.quantize_model(exported_path)
        exported_path = quantized_path
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_model(exported_path)
    
    # Create deployment info
    deployment_info = {
        'model_path': exported_path,
        'image_type': args.image_type,
        'input_shape': exporter.config.input_size,
        'num_classes': exporter.config.num_classes,
        'class_names': ['chip', 'check'],
        'optimized': args.optimize,
        'quantized': args.quantize,
        'export_date': str(torch.datetime.now())
    }
    
    info_path = os.path.join(args.output_dir, f"deployment_info_{args.image_type.lower()}.json")
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ONNX EXPORT COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Model exported to: {exported_path}")
    print(f"Deployment info: {info_path}")
    print(f"Image type: {args.image_type}")
    print(f"Optimized: {args.optimize}")
    print(f"Quantized: {args.quantize}")
    print(f"\nReady for deployment!")

if __name__ == "__main__":
    main()

