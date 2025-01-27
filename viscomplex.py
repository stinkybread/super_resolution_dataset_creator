import torch
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from scipy.stats import entropy
import logging
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import psutil
import pickle

class SuperResImageSelector:
    def __init__(self, input_folder: str, output_folder: str = None):
        self.input_folder = Path(input_folder)
        if output_folder is None:
            self.output_folder = self.input_folder.parent / (self.input_folder.name + "_filtered")
        else:
            self.output_folder = Path(output_folder)
        
        # System resource detection
        self.total_ram = psutil.virtual_memory().total / (1024**3)
        self.available_ram = psutil.virtual_memory().available / (1024**3)
        self.cpu_count = psutil.cpu_count(logical=False)
        
        # Calculate optimal batch sizes
        self.processing_batch_size = min(32, max(8, self.cpu_count * 2))
        max_images_in_memory = int((self.available_ram * 0.7) / 0.01)
        self.distance_batch_size = min(5000, max_images_in_memory)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.temp_dir = Path("temp_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.temp_dir.mkdir(exist_ok=True)
        self._setup_logging()
        
        # Default brightness threshold
        self.max_brightness = 200

    def _setup_logging(self):
        self.logger = logging.getLogger('SuperResImageSelector')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.logger.info(f"System resources detected:")
        self.logger.info(f"Total RAM: {self.total_ram:.1f}GB")
        self.logger.info(f"Available RAM: {self.available_ram:.1f}GB")
        self.logger.info(f"CPU cores: {self.cpu_count}")
        self.logger.info(f"Processing batch size: {self.processing_batch_size}")
        self.logger.info(f"Distance calculation batch size: {self.distance_batch_size}")

    def _save_checkpoint(self, data, step_name):
        checkpoint_path = self.temp_dir / f"{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def _load_latest_checkpoint(self, step_name):
        checkpoints = list(self.temp_dir.glob(f"{step_name}_*.pkl"))
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        with open(latest_checkpoint, 'rb') as f:
            return pickle.load(f)

    def calculate_complexity(self, image_path: Path) -> dict:
        """Calculate image complexity metrics"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert to HSV to get brightness
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])  # V channel represents brightness
            
            # Skip overly bright images
            if brightness > self.max_brightness:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate entropy
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            img_entropy = entropy(hist_norm)
            
            # Calculate edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Calculate sharpness using Laplacian variance (32-bit for efficiency)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            sharpness = np.var(laplacian)
            
            # Calculate overall complexity score
            complexity_score = (
                0.4 * img_entropy +
                0.4 * edge_density +
                0.2 * (sharpness / 1000)  # Normalized and weighted
            )
            
            return {
                'entropy': img_entropy,
                'edge_density': edge_density,
                'sharpness': sharpness,
                'brightness': brightness,
                'complexity_score': complexity_score
            }
        except Exception as e:
            self.logger.error(f"Error calculating complexity for {image_path}: {e}")
            return None

    def get_clip_features(self, image_path: Path) -> torch.Tensor:
        """Extract CLIP features from image"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                # Convert to float32 and normalize
                features = features.float().cpu().flatten()
                features = features / (torch.norm(features) + 1e-8)
                return features
        except Exception as e:
            self.logger.error(f"Error getting CLIP features for {image_path}: {e}")
            return None

    def analyze_image(self, image_path: Path) -> dict:
        try:
            complexity_metrics = self.calculate_complexity(image_path)
            if complexity_metrics is None:
                return None
            
            clip_features = self.get_clip_features(image_path)
            if clip_features is None:
                return None
                
            return {
                'path': image_path,
                'clip_features': clip_features,
                **complexity_metrics
            }
        except Exception as e:
            self.logger.error(f"Error analyzing {image_path}: {e}")
            return None

    def select_images(self, min_distance: float = 0.15, complexity_threshold: float = 0.4):
        try:
            # Load or analyze images
            results = self._load_latest_checkpoint("analysis") or []
            
            if not results:
                image_paths = [p for p in self.input_folder.glob('*') 
                             if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
                
                self.logger.info("Analyzing images...")
                with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
                    futures = [executor.submit(self.analyze_image, path) 
                             for path in image_paths]
                    results = [f.result() for f in tqdm(futures) if f.result() is not None]
                
                self._save_checkpoint(results, "analysis")
            
            if not results:
                self.logger.error("No valid images found")
                return []
            
            # Sort by complexity and filter
            results.sort(key=lambda x: x['complexity_score'], reverse=True)
            complex_images = [r for r in results if r['complexity_score'] >= complexity_threshold]
            
            if not complex_images:
                self.logger.warning(f"No images meet complexity threshold {complexity_threshold}")
                complex_images = results[:1000]  # Take top 1000 by complexity
            
            self.logger.info(f"Selected {len(complex_images)} complex images")
            
            # Convert all features to numpy array (float32)
            features = np.vstack([img['clip_features'].numpy() for img in complex_images]).astype(np.float32)
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            # Calculate pairwise distances
            self.logger.info("Calculating pairwise distances...")
            similarities = np.dot(features, features.T)
            np.clip(similarities, -1.0, 1.0, out=similarities)  # Numerical stability
            distances = 1.0 - similarities
            distances = distances.astype(np.float32)  # Convert to float32
            
            # Select diverse images
            selected_indices = []
            available = set(range(len(complex_images)))
            
            # Start with most complex image (index 0 since already sorted)
            selected_indices.append(0)
            available.remove(0)
            
            self.logger.info("Selecting diverse images...")
            while available:
                curr_distances = distances[list(available)][:, selected_indices]
                min_dists = np.min(curr_distances, axis=1)
                valid = min_dists >= min_distance
                
                if not np.any(valid):
                    break
                    
                valid_indices = np.array(list(available))[valid]
                idx = valid_indices[0]  # Take first valid image (highest complexity)
                
                selected_indices.append(idx)
                available.remove(idx)
                
                if len(selected_indices) % 100 == 0:
                    self.logger.info(f"Selected {len(selected_indices)} images...")
            
            selected_paths = [complex_images[i]['path'] for i in selected_indices]
            self.logger.info(f"Total selected: {len(selected_paths)} images")
            return selected_paths
            
        except Exception as e:
            self.logger.error(f"Error in selection process: {e}")
            raise

    def copy_selected_images(self, min_distance: float = 0.15, complexity_threshold: float = 0.4):
        try:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            selected_paths = self.select_images(min_distance, complexity_threshold)
            
            if not selected_paths:
                self.logger.error("No images were selected")
                return
            
            for path in selected_paths:
                try:
                    shutil.copy2(path, self.output_folder)
                    self.logger.info(f"Copied {path.name}")
                except Exception as e:
                    self.logger.error(f"Error copying {path}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in copy process: {e}")

    def cleanup(self):
        """Clean up temporary files and directories"""
        try:
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            self.logger.error(f"Error cleaning up: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Select images suitable for super resolution')
    parser.add_argument('input_folder', help='Input folder containing images')
    parser.add_argument('--min_distance', type=float, default=0.15, 
                       help='Minimum CLIP distance between selected images')
    parser.add_argument('--complexity_threshold', type=float, default=0.4,
                       help='Minimum complexity score for selection')
    parser.add_argument('--max_brightness', type=float, default=200,
                       help='Maximum average brightness value (0-255)')
    
    args = parser.parse_args()
    
    selector = SuperResImageSelector(args.input_folder)
    selector.max_brightness = args.max_brightness
    try:
        selector.copy_selected_images(args.min_distance, args.complexity_threshold)
    finally:
        selector.cleanup()
