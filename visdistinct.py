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

class DistinctImageSelector:
    def __init__(self, input_folder: str, output_folder: str = None, batch_size: int = 16, num_workers: int = None):
        """Initialize the image selector"""
        try:
            self.input_folder = Path(input_folder)
            if output_folder is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_folder = self.input_folder / f"distinct_images_{timestamp}"
            self.output_folder = Path(output_folder)
            
            # Setup configurations
            self.batch_size = batch_size
            self.num_workers = num_workers if num_workers is not None else max(1, mp.cpu_count() - 1)
            
            # Setup logging
            self._setup_logging()
            
            # Initialize CLIP
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"Using {self.num_workers} worker processes")
            self.logger.info(f"Batch size: {self.batch_size}")
            
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.logger.info("Successfully loaded CLIP model")
            
            # Supported image formats
            self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            
        except Exception as e:
            raise Exception(f"Failed to initialize: {str(e)}")

    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            self.logger = logging.getLogger('DistinctImageSelector')
            self.logger.setLevel(logging.INFO)
            
            # Create handlers
            c_handler = logging.StreamHandler()
            f_handler = logging.FileHandler('image_selector.log')
            
            # Create formatters
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(formatter)
            f_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(c_handler)
            self.logger.addHandler(f_handler)
        except Exception as e:
            raise Exception(f"Failed to setup logging: {str(e)}")

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
                features = features.squeeze()
                if features.ndim == 1:
                    features = features.unsqueeze(0)
                elif features.ndim > 2:
                    features = features.flatten()
                    features = features.unsqueeze(0)
            return features.cpu()
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            raise

    def _process_batch(self, image_paths: List[Path]) -> dict:
        """Process a batch of images"""
        batch_features = {}
        for path in image_paths:
            try:
                features = self._get_image_features(path)
                batch_features[path] = features
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}")
        return batch_features

    def select_distinct_images(self, n: int = 5, min_distance: float = 0.2) -> List[Path]:
        """Select n most visually distinct images"""
        try:
            # Get all valid image paths
            image_paths = [f for f in self.input_folder.iterdir() if self._is_valid_image(f)]
            
            if len(image_paths) < n:
                self.logger.warning(f"Only found {len(image_paths)} valid images, requested {n}")
                n = len(image_paths)
            
            if not image_paths:
                self.logger.error("No valid images found in input folder")
                return []

            # Create batches
            batches = [image_paths[i:i + self.batch_size] 
                      for i in range(0, len(image_paths), self.batch_size)]
            
            # Process batches
            features_dict = {}
            total_images = len(image_paths)
            start_time = time.time()
            processed_count = 0

            with tqdm(total=total_images, desc="Processing images") as pbar:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
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
                            eta_seconds = remaining_images / images_per_second
                            
                            self.logger.info(
                                f"Processed {processed_count}/{total_images} images. "
                                f"Speed: {images_per_second:.2f} img/s. "
                                f"ETA: {eta_seconds/60:.1f} minutes"
                            )
                        except Exception as e:
                            self.logger.error(f"Error processing batch: {e}")
                            continue

            if not features_dict:
                self.logger.error("No features were successfully extracted")
                return []

            # Process features
            try:
                features_list = list(features_dict.values())
                features_matrix = torch.stack(features_list)
                
                if features_matrix.ndim > 2:
                    original_shape = features_matrix.shape
                    features_matrix = features_matrix.reshape(original_shape[0], -1)
                
                self.logger.info(f"Feature matrix shape: {features_matrix.shape}")
                
                features_matrix = features_matrix / (features_matrix.norm(dim=1, keepdim=True) + 1e-8)
                distances = 1 - torch.mm(features_matrix, features_matrix.t())
                
                # Select distinct images
                selected_indices = []
                selected_paths = []
                
                center_idx = distances.mean(dim=1).argmin()
                selected_indices.append(center_idx)
                selected_paths.append(list(features_dict.keys())[center_idx])

                while len(selected_indices) < n:
                    distances_to_selected = distances[:, selected_indices].min(dim=1)[0]
                    best_idx = distances_to_selected.argmax()
                    
                    if distances_to_selected[best_idx] < min_distance:
                        self.logger.warning(f"Could not find more images with minimum distance {min_distance}")
                        break
                    
                    selected_indices.append(best_idx)
                    selected_paths.append(list(features_dict.keys())[best_idx])
                
                return selected_paths
                
            except Exception as e:
                self.logger.error(f"Error in feature processing: {e}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error in selection process: {e}")
            return []

    def copy_distinct_images(self, n: int = 5, min_distance: float = 0.2):
        """Copy n most distinct images to output folder"""
        try:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            selected_paths = self.select_distinct_images(n, min_distance)
            
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
            print("Usage: python script.py input_folder [output_folder] [num_images]")
            sys.exit(1)
            
        input_folder = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None
        num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
        selector = DistinctImageSelector(input_folder, output_folder)
        selector.copy_distinct_images(n=num_images)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()