import os
import sys
import torch
import torch.nn as nn
import yaml
import argparse
import shutil
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import warnings
warnings.filterwarnings('ignore')

class TinyDefectSVTrainer:
    """Advanced trainer for SV tiny defect detection optimized for 1024×500 images."""
    
    def __init__(self, dataset_path: str, output_dir: str = None):
        self.dataset_path = dataset_path
        self.model_type = "SV"
        
        # Create output directory with timestamp
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"sv_tiny_defect_results_{timestamp}"
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # SV-specific training parameters optimized for tiny defects
        self.training_params = {
            # Core training settings
            'epochs': 350,  # More epochs for smaller SV images
            'batch': 8,     # Larger batch for smaller images
            'imgsz': 1024,  # High resolution for tiny defects in 1024×500 images
            'lr0': 0.00008, # Slightly lower LR for SV stability
            'patience': 60,
            'save': True,
            'save_period': 25,
            'workers': 0,   # Single threaded for stability
            'seed': 42,
            'deterministic': True,
            
            # Model and optimization
            'optimizer': 'AdamW',
            'cos_lr': True,
            'close_mosaic': 60,  # Close mosaic later for SV
            'resume': False,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'freeze': None,
            'multi_scale': True,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            
            # Validation settings
            'val': True,
            'split': 'val',
            'save_json': True,
            'save_hybrid': False,
            'conf': 0.0005,  # Even lower confidence for tiny SV defects
            'iou': 0.65,     # Slightly higher IoU for SV
            'max_det': 1500, # Higher max detections for SV
            'half': False,
            'dnn': False,
            'plots': True,
            'show_labels': True,
            'show_conf': True,
            'show_boxes': True,
            
            # SV-specific augmentation parameters (optimized for 1024×500 images)
            'hsv_h': 0.008,     # Even more minimal hue change
            'hsv_s': 0.25,      # Moderate saturation change
            'hsv_v': 0.18,      # Moderate value change
            'degrees': 2.0,     # Smaller rotation for SV defects
            'translate': 0.025, # Minimal translation
            'scale': 0.03,      # Very minimal scaling for tiny defects
            'shear': 0.3,       # Small shear
            'perspective': 0.00005,  # Minimal perspective
            'flipud': 0.0,      # No vertical flip for defects
            'fliplr': 0.6,      # Higher horizontal flip for SV
            'mosaic': 0.25,     # Slightly more mosaic for SV
            'mixup': 0.08,      # Light mixup
            'copy_paste': 0.25, # More copy-paste for SV small objects
            
            # Loss function optimizations for tiny SV defects
            'box': 9.0,         # Even higher box loss weight
            'cls': 0.35,        # Lower classification loss
            'dfl': 2.5,         # Higher distribution focal loss
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'anchor_t': 3.5,    # Slightly lower anchor threshold
            'fl_gamma': 0.0,    # Focal loss gamma
        }
        
        print(f"SV Tiny Defect Trainer initialized")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {self.output_dir}")
        print(f"Optimized for: 1024×500 SV images with tiny defects")
    
    def setup_environment(self):
        """Setup training environment and check requirements."""
        print("\nSetting up training environment...")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"GPU Count: {gpu_count}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # SV images are smaller, so we can use larger batches
            if gpu_memory >= 8:
                self.training_params['batch'] = 12
            elif gpu_memory >= 6:
                self.training_params['batch'] = 8
            else:
                self.training_params['batch'] = 4
                print("Warning: Limited GPU memory. Reducing batch size.")
        else:
            print("Warning: No GPU detected. Training will be very slow.")
            self.training_params['batch'] = 2
            self.training_params['imgsz'] = 640
        
        # Check dataset
        dataset_yaml = os.path.join(self.dataset_path, 'dataset.yaml')
        if not os.path.exists(dataset_yaml):
            print(f"Error: Dataset configuration not found: {dataset_yaml}")
            return False
        
        # Validate dataset structure
        required_dirs = ['images/train', 'labels/train', 'images/val', 'labels/val']
        for dir_name in required_dirs:
            dir_path = os.path.join(self.dataset_path, dir_name)
            if os.path.exists(dir_path):
                file_count = len([f for f in os.listdir(dir_path) 
                                if os.path.isfile(os.path.join(dir_path, f))])
                print(f"Found {file_count} files in {dir_name}")
            else:
                print(f"Warning: {dir_name} not found")
        
        return True
    
    def create_custom_model_config(self):
        """Create custom YOLOv8 configuration optimized for tiny SV defects."""
        config = {
            # Model metadata
            'nc': 2,  # Number of classes (chip, check)
            'depth_multiple': 0.33,
            'width_multiple': 0.50,  # Balanced width for SV
            
            # Backbone optimized for tiny objects in smaller images
            'backbone': [
                [-1, 1, 'Conv', [64, 6, 2, 2]],   # 0-P1/2
                [-1, 1, 'Conv', [128, 3, 2]],     # 1-P2/4 (critical for tiny SV defects)
                [-1, 3, 'C2f', [128, True]],
                [-1, 1, 'Conv', [256, 3, 2]],     # 3-P3/8
                [-1, 6, 'C2f', [256, True]],
                [-1, 1, 'Conv', [512, 3, 2]],     # 5-P4/16
                [-1, 6, 'C2f', [512, True]],
                [-1, 1, 'Conv', [1024, 3, 2]],    # 7-P5/32
                [-1, 3, 'C2f', [1024, True]],
                [-1, 1, 'SPPF', [1024, 5]],       # 9
            ],
            
            # Head with enhanced P2 detection for tiny SV objects
            'head': [
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],      # cat backbone P4
                [-1, 3, 'C2f', [512]],            # 12
                
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],      # cat backbone P3
                [-1, 3, 'C2f', [256]],            # 15 (P3/8-medium)
                
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 2], 1, 'Concat', [1]],      # cat backbone P2
                [-1, 3, 'C2f', [128]],            # 18 (P2/4-small) - Enhanced for SV tiny defects
                
                [-1, 1, 'Conv', [256, 3, 2]],
                [[-1, 15], 1, 'Concat', [1]],     # cat head P3
                [-1, 3, 'C2f', [256]],            # 21 (P3/8-medium)
                
                [-1, 1, 'Conv', [512, 3, 2]],
                [[-1, 12], 1, 'Concat', [1]],     # cat head P4
                [-1, 3, 'C2f', [512]],            # 24 (P4/16-large)
                
                [-1, 1, 'Conv', [1024, 3, 2]],
                [[-1, 9], 1, 'Concat', [1]],      # cat head P5
                [-1, 3, 'C2f', [1024]],           # 27 (P5/32-xlarge)
                
                [[18, 21, 24, 27], 1, 'Detect', ['nc']]  # Detect(P2, P3, P4, P5)
            ]
        }
        
        return config
    
    def load_model(self, model_choice: str = "yolov8m"):
        """Load and configure model for SV tiny defect detection."""
        print(f"\nLoading {model_choice} model for SV training...")
        
        # Model selection optimized for SV tiny defects
        if model_choice == "yolov8n":
            self.model = YOLO('yolov8n.pt')
        elif model_choice == "yolov8s":
            self.model = YOLO('yolov8s.pt')
        elif model_choice == "yolov8m":
            self.model = YOLO('yolov8m.pt')  # Good balance for SV
        elif model_choice == "yolov8l":
            self.model = YOLO('yolov8l.pt')  # Better for tiny defects
        elif model_choice == "yolov8x":
            self.model = YOLO('yolov8x.pt')  # Best if GPU allows
        else:
            print(f"Unknown model choice: {model_choice}, using yolov8m")
            self.model = YOLO('yolov8m.pt')
        
        print(f"Model loaded: {model_choice}")
        return True
    
    def train_model(self, model_choice: str = "yolov8m"):
        """Train SV model with advanced tiny defect detection techniques."""
        if not self.load_model(model_choice):
            return False
        
        dataset_yaml = os.path.join(self.dataset_path, 'dataset.yaml')
        
        print(f"\nStarting SV tiny defect training...")
        print("=" * 60)
        print(f"Model: {model_choice}")
        print(f"Dataset: {dataset_yaml}")
        print(f"Image size: {self.training_params['imgsz']}")
        print(f"Epochs: {self.training_params['epochs']}")
        print(f"Batch size: {self.training_params['batch']}")
        print(f"Learning rate: {self.training_params['lr0']}")
        print(f"Optimized for: SV images (1024×500) with tiny defects")
        
        # Save training configuration
        config_path = os.path.join(self.output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'model_choice': model_choice,
                'dataset_path': self.dataset_path,
                'training_params': self.training_params,
                'optimization_target': 'tiny_defects_in_1024x500_SV_images'
            }, f, indent=2)
        
        try:
            # Set training parameters
            training_args = self.training_params.copy()
            training_args.update({
                'data': dataset_yaml,
                'project': self.output_dir,
                'name': 'train',
                'exist_ok': True,
                'pretrained': True,
                'verbose': True,
            })
            
            # Train the model
            print("\nStarting training...")
            results = self.model.train(**training_args)
            
            print(f"\nSV training completed successfully!")
            
            # Save best model with descriptive name
            best_model_path = os.path.join(self.output_dir, 'train', 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                final_model_path = os.path.join(self.output_dir, f'SV_tiny_defect_best_{model_choice}.pt')
                shutil.copy2(best_model_path, final_model_path)
                print(f"Best SV model saved: {final_model_path}")
                
                # Export to different formats for production
                try:
                    export_model = YOLO(best_model_path)
                    
                    # ONNX export for production inference
                    onnx_path = os.path.join(self.output_dir, f'SV_tiny_defect_best_{model_choice}.onnx')
                    export_model.export(format='onnx', imgsz=self.training_params['imgsz'])
                    print(f"ONNX model exported for production")
                    
                    # TensorRT export if available
                    try:
                        trt_path = os.path.join(self.output_dir, f'SV_tiny_defect_best_{model_choice}.engine')
                        export_model.export(format='engine', imgsz=self.training_params['imgsz'])
                        print(f"TensorRT model exported for optimized inference")
                    except Exception as e:
                        print(f"TensorRT export skipped: {e}")
                        
                except Exception as e:
                    print(f"Model export failed: {e}")
            
            # Save training summary
            self._save_training_summary(results, model_choice)
            
            return True
            
        except Exception as e:
            print(f"SV training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_training_summary(self, results, model_choice):
        """Save training summary and recommendations."""
        summary = {
            'model_type': 'SV',
            'model_choice': model_choice,
            'training_completed': datetime.now().isoformat(),
            'dataset_path': self.dataset_path,
            'output_dir': self.output_dir,
            'training_params': self.training_params,
            'optimization_focus': 'tiny_defects_in_1024x500_SV_images',
            
            'recommendations': {
                'inference_resolution': self.training_params['imgsz'],
                'confidence_threshold': 0.0005,
                'iou_threshold': 0.65,
                'use_sahi_inference': True,
                'sahi_slice_size': 512,
                'sahi_overlap_ratio': 0.3
            },
            
            'production_notes': [
                'Use SAHI slicing for inference on full 1024×500 images',
                'Apply same preprocessing as training (normalization, etc.)',
                'Consider Test Time Augmentation for critical detections',
                'Monitor for false positives in glass reflections/scratches',
                'Validate performance on actual production SV images',
                'SV defects may be even smaller than EV - use lower confidence thresholds'
            ]
        }
        
        summary_path = os.path.join(self.output_dir, 'SV_training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved: {summary_path}")
    
    def evaluate_model(self):
        """Evaluate trained SV model."""
        best_model_path = os.path.join(self.output_dir, 'train', 'weights', 'best.pt')
        
        if not os.path.exists(best_model_path):
            print("Trained model not found for evaluation")
            return False
        
        try:
            model = YOLO(best_model_path)
            dataset_yaml = os.path.join(self.dataset_path, 'dataset.yaml')
            
            print(f"\nEvaluating SV tiny defect model...")
            
            # Comprehensive evaluation with tiny object focus
            results = model.val(
                data=dataset_yaml,
                split='test',
                imgsz=self.training_params['imgsz'],
                batch=1,  # Single batch for accurate evaluation
                save_json=True,
                save_hybrid=False,
                conf=0.0005,  # Very low confidence for tiny SV objects
                iou=0.65,
                max_det=1500,
                half=False,
                device=None,
                dnn=False,
                plots=True,
                rect=False,
                save_txt=True,
                save_conf=True,
                save_crop=False,
                show_labels=True,
                show_conf=True,
                show_boxes=True,
                verbose=True,
                project=self.output_dir,
                name='evaluation',
                exist_ok=True
            )
            
            print(f"SV evaluation completed: {self.output_dir}/evaluation")
            
            # Save evaluation summary
            eval_summary = {
                'model_type': 'SV',
                'evaluation_completed': datetime.now().isoformat(),
                'evaluation_params': {
                    'confidence_threshold': 0.0005,
                    'iou_threshold': 0.65,
                    'image_size': self.training_params['imgsz']
                },
                'results_location': f"{self.output_dir}/evaluation"
            }
            
            eval_path = os.path.join(self.output_dir, 'SV_evaluation_summary.json')
            with open(eval_path, 'w') as f:
                json.dump(eval_summary, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"SV evaluation failed: {str(e)}")
            return False

def main():
    """Main function for SV tiny defect detection training."""
    parser = argparse.ArgumentParser(description='SV Tiny Defect Detection Training')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to the SV dataset directory (e.g., sahi_datasets/SV_sahi_dataset)')
    parser.add_argument('--model', type=str, default='yolov8m',
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='YOLOv8 model variant to use')
    parser.add_argument('--epochs', type=int, default=350,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=1024,
                       help='Training image size')
    parser.add_argument('--lr', type=float, default=0.00008,
                       help='Learning rate')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation after training')
    
    args = parser.parse_args()
    
    print("SV Tiny Defect Detection Training System")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Image Size: {args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Learning Rate: {args.lr}")
    print("Optimized for: 1024×500 SV images with tiny defects")
    
    # Initialize trainer
    trainer = TinyDefectSVTrainer(args.dataset, args.output)
    
    # Update parameters from command line
    trainer.training_params.update({
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'lr0': args.lr
    })
    
    # Setup environment
    if not trainer.setup_environment():
        print("Environment setup failed")
        sys.exit(1)
    
    # Train model
    if trainer.train_model(args.model):
        print(f"\nSV tiny defect model training completed!")
        
        if args.evaluate:
            trainer.evaluate_model()
        
        print(f"\nResults saved to: {trainer.output_dir}")
        print("\nNext steps:")
        print("1. Review training metrics in the results directory")
        print("2. Test the model on actual SV production images")
        print("3. Implement SAHI inference for full-resolution detection")
        print("4. Fine-tune confidence thresholds based on production requirements")
        print("5. Compare SV model performance with EV model")
    else:
        print("SV training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

