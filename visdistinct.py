import os
import shutil
import logging
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Set, Tuple
from datetime import datetime
import sys
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import psutil
import cpuinfo
import math

class DistanceImageSelector:
    @staticmethod
    def _get_cpu_info():
        """Get detailed CPU information and recommend optimal parameters"""
        try:
            cpu = cpuinfo.get_cpu_info()
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            architecture = cpu.get('arch', 'unknown')
            brand = cpu.get('brand_raw', 'unknown')
            
            is_ryzen = 'ryzen' in brand.lower()
            is_intel = 'intel' in brand.lower()
            is_threadripper = 'threadripper' in brand.lower()
            
            max_batch_by_memory = int((total_memory_gb * 16) / 2)
            
            if is_ryzen or is_threadripper:
                base_batch_size = 32
            elif is_intel:
                base_batch_size = 24
            else:
                base_batch_size = 16
                
            batch_size = min(base_batch_size * math.ceil(physical_cores / 6), max_batch_by_memory)
            
            if is_threadripper:
                num_workers = logical_cores - 2
            elif is_ryzen:
                num_workers = physical_cores - 1
            else:
                num_workers = max(1, min(physical_cores - 1, logical_cores // 2))
                
            return {
                'brand': brand,
                'architecture': architecture,
                'physical_cores': physical_cores,
                'logical_cores': logical_cores,
                'total_memory_gb': total_memory_gb,
                'recommended_batch_size': int(batch_size),
                'recommended_workers': num_workers
            }
        except Exception as e:
            return {
                'brand': 'unknown',
                'architecture': 'unknown',
                'physical_cores': 4,
                'logical_cores': 8,
                'total_memory_gb': 8,
                'recommended_batch_size': 16,
                'recommended_workers': 4
            }

    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            self.logger = logging.getLogger('DistanceImageSelector')
            self.logger.handlers.clear()
            
            self.logger.setLevel(logging.INFO)
            
            c_handler = logging.StreamHandler()
            f_handler = logging.FileHandler('image_selector.log')
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(formatter)
            f_handler.setFormatter(formatter)
            
            self.logger.addHandler(c_handler)
            self.logger.addHandler(f_handler)
        except Exception as e:
            raise Exception(f"Failed to setup logging: {str(e)}")

    def __init__(self, input_folder: str, output_folder: str = None, batch_size: int = None, num_workers: int = None):
        """Initialize the image selector"""
        try:
            self._setup_logging()
            
            self.input_folder = Path(input_folder)
            if output_folder is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_folder = self.input_folder / f"distinct_images_{timestamp}"
            self.output_folder = Path(output_folder)
            
            cpu_info = self._get_cpu_info()
            self.logger.info(f"Detected CPU: {cpu_info['brand']}")
            self.logger.info(f"Architecture: {cpu_info['architecture']}")
            self.logger.info(f"Physical cores: {cpu_info['physical_cores']}")
            self.logger.info(f"Logical cores: {cpu_info['logical_cores']}")
            self.logger.info(f"Total memory: {cpu_info['total_memory_gb']:.1f} GB")
            
            self.batch_size = batch_size if batch_size is not None else cpu_info['recommended_batch_size']
            self.num_workers = num_workers if num_workers is not None else cpu_info['recommended_workers']
            
            self.logger.info(f"Using batch size: {self.batch_size}")
            self.logger.info(f"Using workers: {self.num_workers}")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
            
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.logger.info("Successfully loaded CLIP model")
            
            self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            
        except Exception as e:
            raise Exception(f"Failed to initialize: {str(e)}")

    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if file is a valid image"""
        try:
            if file_path.suffix.lower() not in self.supported_formats:
                return False
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception as e:
            self.logger.warning(f"Invalid image file {file_path}: {e}")
            return False

    def _get_image_features(self, image_path: Path) -> torch.Tensor:
        """Extract CLIP features from image"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                # Always ensure 2D output (batch_size, features)
                features = features.squeeze()
                if features.ndim == 1:
                    features = features.unsqueeze(0)
                return features.cpu()
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            raise

    def _process_batch(self, image_paths: List[Path]) -> dict:
        """Process a batch of images"""
        try:
            batch_features = {}
            for path in image_paths:
                try:
                    features = self._get_image_features(path)
                    batch_features[path] = features
                except Exception as e:
                    self.logger.error(f"Error processing {path}: {e}")
            return batch_features
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            return {}

    def _extract_features(self, image_paths: List[Path]) -> dict:
        """Extract features from all images using batched processing"""
        try:
            features_dict = {}
            total_images = len(image_paths)
            
            batches = [image_paths[i:i + self.batch_size] 
                      for i in range(0, len(image_paths), self.batch_size)]
            
            start_time = time.time()
            processed_count = 0

            with tqdm(total=total_images, desc="Processing images") as pbar:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    try:
                        future_to_batch = {
                            executor.submit(self._process_batch, batch): batch 
                            for batch in batches
                        }
                        
                        for future in as_completed(future_to_batch):
                            try:
                                batch = future_to_batch[future]
                                batch_features = future.result()
                                features_dict.update(batch_features)
                                
                                processed_count += len(batch)
                                pbar.update(len(batch))
                                
                                elapsed_time = time.time() - start_time
                                images_per_second = processed_count / elapsed_time
                                remaining_images = total_images - processed_count
                                eta_seconds = remaining_images / images_per_second if images_per_second > 0 else 0
                                
                                self.logger.info(
                                    f"Processed {processed_count}/{total_images} images. "
                                    f"Speed: {images_per_second:.2f} img/s. "
                                    f"ETA: {eta_seconds/60:.1f} minutes"
                                )
                            except Exception as e:
                                self.logger.error(f"Error processing batch: {e}")
                                continue
                    except Exception as e:
                        self.logger.error(f"Error in thread pool execution: {e}")
                        raise
            
            return features_dict
        except Exception as e:
            self.logger.error(f"Error in feature extraction: {e}")
            return {}

    def select_by_distance(self, min_distance: float = 0.15, max_images: int = None) -> List[Path]:
        """Select images that are at least min_distance apart"""
        try:
            image_paths = [f for f in self.input_folder.iterdir() if self._is_valid_image(f)]
            
            if not image_paths:
                self.logger.error("No valid images found in input folder")
                return []

            features_dict = self._extract_features(image_paths)
            
            if not features_dict:
                self.logger.error("No features were successfully extracted")
                return []

            try:
                features_list = list(features_dict.values())
                features_matrix = torch.stack(features_list)
                
                # Debug info
                self.logger.info(f"Initial feature matrix shape: {features_matrix.shape}")
                
                # Ensure 2D matrix
                if features_matrix.ndim > 2:
                    features_matrix = features_matrix.view(features_matrix.size(0), -1)
                    self.logger.info(f"Reshaped feature matrix shape: {features_matrix.shape}")
                
                # Normalize features with better numerical stability
                norms = torch.norm(features_matrix, p=2, dim=1, keepdim=True)
                features_matrix = features_matrix / (norms + 1e-8)
                
                # Calculate pairwise distances with better numerical stability
                similarities = torch.mm(features_matrix, features_matrix.t())
                similarities = torch.clamp(similarities, min=-1.0, max=1.0)
                distances = 1.0 - similarities
                distances = torch.clamp(distances, min=0.0, max=1.0)
                
                self.logger.info(f"Distance matrix shape: {distances.shape}")
                self.logger.info(f"Distance range: [{distances.min().item():.3f}, {distances.max().item():.3f}]")
                
                # Initialize selection
                selected_indices = []
                selected_paths = []
                available_indices = set(range(len(image_paths)))
                paths_list = list(features_dict.keys())
                
                # Start with the most central image
                try:
                    center_distances = distances.mean(dim=1)
                    center_idx = int(center_distances.argmin().item())
                    selected_indices.append(center_idx)
                    selected_paths.append(paths_list[center_idx])
                    available_indices.remove(center_idx)
                    
                    self.logger.info(f"Selected initial center image: {paths_list[center_idx].name}")
                    
                    while available_indices:
                        # Convert available indices to tensor for indexing
                        available_idx_tensor = torch.tensor(list(available_indices), dtype=torch.long)
                        selected_idx_tensor = torch.tensor(selected_indices, dtype=torch.long)
                        
                        # Calculate distances to selected images
                        current_distances = distances[available_idx_tensor][:, selected_idx_tensor]
                        min_distances, _ = current_distances.min(dim=1)
                        
                        # Find valid candidates
                        valid_mask = min_distances >= min_distance
                        if not valid_mask.any():
                            self.logger.info(f"No more images found with minimum distance {min_distance}")
                            break
                        
                        # Among valid candidates, select the one with maximum minimum distance
                        valid_distances = min_distances[valid_mask]
                        best_local_idx = valid_distances.argmax()
                        global_idx = int(available_idx_tensor[valid_mask].index_select(0, best_local_idx.unsqueeze(0)).item())
                        
                        selected_indices.append(global_idx)
                        selected_paths.append(paths_list[global_idx])
                        available_indices.remove(global_idx)
                        
                        if len(selected_paths) % 10 == 0:
                            self.logger.info(f"Selected {len(selected_paths)} images...")
                        
                        if max_images and len(selected_paths) >= max_images:
                            self.logger.info(f"Reached maximum number of images: {max_images}")
                            break
                    
                    self.logger.info(f"Selected {len(selected_paths)} images with minimum distance {min_distance}")
                    return selected_paths
                    
                except Exception as e:
                    self.logger.error(f"Error in image selection: {e}")
                    self.logger.error(f"Current state - Selected: {len(selected_indices)}, Available: {len(available_indices)}")
                    raise
                
            except Exception as e:
                self.logger.error(f"Error in feature processing: {e}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error in selection process: {e}")
            return []

    def copy_distinct_images(self, min_distance: float = 0.15, max_images: int = None):
        """Copy distinct images to output folder"""
        try:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            selected_paths = self.select_by_distance(min_distance, max_images)
            
            if not selected_paths:
                self.logger.error("No images were selected")
                return
                
            self.logger.info(f"Copying {len(selected_paths)} images to {self.output_folder}")
            for path in selected_paths:
                try:
                    shutil.copy2(path, self.output_folder)
                    self.logger.info(f"Copied {path.name}")
                except Exception as e:
                    self.logger.error(f"Error copying {path}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in copy process: {e}")

def main():
    """Main function to run the script"""
    try:
        if len(sys.argv) < 2:
            print("Usage: python script.py input_folder [output_folder] [min_distance] [max_images]")
            sys.exit(1)
            
        input_folder = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None
        min_distance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
        max_images = int(sys.argv[4]) if len(sys.argv) > 4 else None
        
        selector = DistanceImageSelector(input_folder, output_folder)
        selector.copy_distinct_images(min_distance=min_distance, max_images=max_images)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
