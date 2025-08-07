#!/usr/bin/env python3
"""
Complete FocusDet Training Pipeline
==================================

This script runs the complete pipeline for training FocusDet models:
1. Convert XML annotations to COCO format
2. Train FocusDet models for EV and SV images
3. Export trained models to ONNX format

Usage:
    python run_focusdet_pipeline.py --all
    python run_focusdet_pipeline.py --convert_only
    python run_focusdet_pipeline.py --train_only --image_type EV
    python run_focusdet_pipeline.py --export_only --model_path path/to/model.pth
"""

import os
import sys
import subprocess
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('focusdet_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FocusDetPipeline:
    """Complete FocusDet training and export pipeline."""
    
    def __init__(self, source_data_path: str = r"D:\Photomask", 
                 output_base: str = "focusdet_output"):
        self.source_data_path = source_data_path
        self.output_base = output_base
        self.dataset_path = os.path.join(output_base, "dataset")
        self.models_path = os.path.join(output_base, "models")
        self.onnx_path = os.path.join(output_base, "onnx_models")
        
        # Create output directories
        for path in [self.output_base, self.dataset_path, self.models_path, self.onnx_path]:
            os.makedirs(path, exist_ok=True)
        
        logger.info(f"Pipeline initialized")
        logger.info(f"Source data: {self.source_data_path}")
        logger.info(f"Output base: {self.output_base}")
    
    def check_dependencies(self):
        """Check if all required dependencies are available."""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'torch', 'torchvision', 'opencv-python', 'onnx', 'onnxruntime'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.error("Please install them using: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("✓ All dependencies are available")
        return True
    
    def check_data_structure(self):
        """Check if the source data structure is correct."""
        logger.info("Checking data structure...")
        
        required_paths = [
            os.path.join(self.source_data_path, "DS0", "EV"),
            os.path.join(self.source_data_path, "DS0", "SV"),
            os.path.join(self.source_data_path, "DS2_Sort2", "EV"),
            os.path.join(self.source_data_path, "DS2_Sort2", "SV"),
            os.path.join(self.source_data_path, "MSA_Sort3", "EV"),
            os.path.join(self.source_data_path, "MSA_Sort3", "SV"),
        ]
        
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            logger.error(f"Missing required data directories:")
            for path in missing_paths:
                logger.error(f"  - {path}")
            return False
        
        # Check for XML files
        xml_count = 0
        for path in required_paths:
            xml_files = [f for f in os.listdir(path) if f.lower().endswith('.xml')]
            xml_count += len(xml_files)
            logger.info(f"  {path}: {len(xml_files)} XML files")
        
        if xml_count == 0:
            logger.error("No XML annotation files found!")
            return False
        
        logger.info(f"✓ Data structure is correct ({xml_count} total XML files)")
        return True
    
    def run_data_conversion(self):
        """Run XML to COCO conversion."""
        logger.info("="*60)
        logger.info("STEP 1: Converting XML annotations to COCO format")
        logger.info("="*60)
        
        try:
            # Import and run converter
            sys.path.append(os.getcwd())
            from xml_to_coco_converter import XMLToCOCOConverter
            
            converter = XMLToCOCOConverter(
                source_base_path=self.source_data_path,
                output_path=self.dataset_path
            )
            
            converter.convert_dataset()
            converter.print_statistics()
            
            logger.info("✓ Data conversion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data conversion failed: {e}")
            return False
    
    def run_training(self, image_type: str, epochs: int = 300, batch_size: int = 4):
        """Run FocusDet training for specified image type."""
        logger.info("="*60)
        logger.info(f"STEP 2: Training FocusDet model for {image_type} images")
        logger.info("="*60)
        
        try:
            # Import and run trainer
            from focusdet_trainer import FocusDetTrainer, FocusDetConfig
            
            # Create configuration
            config = FocusDetConfig(image_type=image_type)
            config.num_epochs = epochs
            config.batch_size = batch_size
            
            # Create trainer
            trainer = FocusDetTrainer(config, self.dataset_path)
            
            # Start training
            best_model_path = trainer.train()
            
            # Move model to models directory
            model_filename = f"focusdet_{image_type.lower()}_best.pth"
            final_model_path = os.path.join(self.models_path, model_filename)
            
            if os.path.exists(best_model_path):
                import shutil
                shutil.copy2(best_model_path, final_model_path)
                logger.info(f"✓ Model saved to: {final_model_path}")
            
            return final_model_path
            
        except Exception as e:
            logger.error(f"Training failed for {image_type}: {e}")
            return None
    
    def run_onnx_export(self, model_path: str, image_type: str, 
                       optimize: bool = True, quantize: bool = False):
        """Run ONNX export for trained model."""
        logger.info("="*60)
        logger.info(f"STEP 3: Exporting {image_type} model to ONNX format")
        logger.info("="*60)
        
        try:
            # Import and run exporter
            from onnx_exporter import FocusDetONNXExporter
            
            exporter = FocusDetONNXExporter(model_path, image_type)
            
            # Export to ONNX
            onnx_filename = f"focusdet_{image_type.lower()}.onnx"
            onnx_path = os.path.join(self.onnx_path, onnx_filename)
            
            exported_path = exporter.export_to_onnx(onnx_path)
            
            # Apply optimizations
            if optimize:
                optimized_path = exporter.optimize_onnx_model(exported_path)
                exported_path = optimized_path
            
            if quantize:
                quantized_path = exporter.quantize_model(exported_path)
                exported_path = quantized_path
            
            logger.info(f"✓ ONNX export completed: {exported_path}")
            return exported_path
            
        except Exception as e:
            logger.error(f"ONNX export failed for {image_type}: {e}")
            return None
    
    def run_complete_pipeline(self, epochs: int = 300, batch_size: int = 4,
                            optimize_onnx: bool = True, quantize_onnx: bool = False):
        """Run the complete pipeline for both EV and SV models."""
        logger.info("="*80)
        logger.info("STARTING COMPLETE FOCUSDET PIPELINE")
        logger.info("="*80)
        
        start_time = datetime.now()
        results = {
            'start_time': start_time.isoformat(),
            'steps_completed': [],
            'models_trained': {},
            'onnx_models': {},
            'errors': []
        }
        
        # Step 1: Data conversion
        if self.run_data_conversion():
            results['steps_completed'].append('data_conversion')
        else:
            results['errors'].append('Data conversion failed')
            return results
        
        # Step 2: Train models for both EV and SV
        for image_type in ['EV', 'SV']:
            logger.info(f"\nTraining {image_type} model...")
            
            model_path = self.run_training(image_type, epochs, batch_size)
            if model_path:
                results['models_trained'][image_type] = model_path
                results['steps_completed'].append(f'training_{image_type}')
                
                # Step 3: Export to ONNX
                onnx_path = self.run_onnx_export(
                    model_path, image_type, optimize_onnx, quantize_onnx
                )
                if onnx_path:
                    results['onnx_models'][image_type] = onnx_path
                    results['steps_completed'].append(f'onnx_export_{image_type}')
                else:
                    results['errors'].append(f'ONNX export failed for {image_type}')
            else:
                results['errors'].append(f'Training failed for {image_type}')
        
        # Finalize results
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['total_duration'] = str(end_time - start_time)
        
        # Save results
        results_path = os.path.join(self.output_base, 'pipeline_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self.print_pipeline_summary(results)
        
        return results
    
    def print_pipeline_summary(self, results: dict):
        """Print pipeline execution summary."""
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*80)
        
        logger.info(f"Start time: {results['start_time']}")
        logger.info(f"End time: {results['end_time']}")
        logger.info(f"Total duration: {results['total_duration']}")
        
        logger.info(f"\nSteps completed: {len(results['steps_completed'])}")
        for step in results['steps_completed']:
            logger.info(f"  ✓ {step}")
        
        if results['models_trained']:
            logger.info(f"\nModels trained: {len(results['models_trained'])}")
            for image_type, path in results['models_trained'].items():
                logger.info(f"  ✓ {image_type}: {path}")
        
        if results['onnx_models']:
            logger.info(f"\nONNX models exported: {len(results['onnx_models'])}")
            for image_type, path in results['onnx_models'].items():
                logger.info(f"  ✓ {image_type}: {path}")
        
        if results['errors']:
            logger.info(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors']:
                logger.error(f"  ✗ {error}")
        
        if len(results['onnx_models']) == 2:
            logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("Both EV and SV models are ready for deployment!")
        else:
            logger.warning("\n⚠ PIPELINE COMPLETED WITH ISSUES")
            logger.warning("Please check the errors above.")

def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='FocusDet Training Pipeline')
    
    # Pipeline mode options
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (convert + train + export)')
    parser.add_argument('--convert_only', action='store_true',
                       help='Only run data conversion')
    parser.add_argument('--train_only', action='store_true',
                       help='Only run training')
    parser.add_argument('--export_only', action='store_true',
                       help='Only run ONNX export')
    
    # Configuration options
    parser.add_argument('--source_data', type=str, default=r"D:\Photomask",
                       help='Path to source data directory')
    parser.add_argument('--output_dir', type=str, default='focusdet_output',
                       help='Output directory for all results')
    parser.add_argument('--image_type', type=str, choices=['EV', 'SV'],
                       help='Image type for training/export (required for train_only/export_only)')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model (required for export_only)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    
    # ONNX export options
    parser.add_argument('--no_optimize', action='store_true',
                       help='Skip ONNX optimization')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply INT8 quantization to ONNX models')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.all, args.convert_only, args.train_only, args.export_only]):
        parser.error("Must specify one of: --all, --convert_only, --train_only, --export_only")
    
    if args.train_only and not args.image_type:
        parser.error("--train_only requires --image_type")
    
    if args.export_only and (not args.model_path or not args.image_type):
        parser.error("--export_only requires --model_path and --image_type")
    
    # Initialize pipeline
    pipeline = FocusDetPipeline(args.source_data, args.output_dir)
    
    # Check dependencies
    if not pipeline.check_dependencies():
        sys.exit(1)
    
    # Check data structure (except for export_only)
    if not args.export_only and not pipeline.check_data_structure():
        sys.exit(1)
    
    # Execute requested pipeline
    try:
        if args.all:
            results = pipeline.run_complete_pipeline(
                epochs=args.epochs,
                batch_size=args.batch_size,
                optimize_onnx=not args.no_optimize,
                quantize_onnx=args.quantize
            )
            
        elif args.convert_only:
            success = pipeline.run_data_conversion()
            if not success:
                sys.exit(1)
                
        elif args.train_only:
            model_path = pipeline.run_training(args.image_type, args.epochs, args.batch_size)
            if not model_path:
                sys.exit(1)
                
        elif args.export_only:
            onnx_path = pipeline.run_onnx_export(
                args.model_path, args.image_type,
                optimize=not args.no_optimize,
                quantize=args.quantize
            )
            if not onnx_path:
                sys.exit(1)
        
        logger.info("\n🎉 Pipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

