# --- START OF MODIFIED FILE srdc_v2.py ---

import cv2
import os
import shutil
import subprocess
from tqdm import tqdm
import glob
import concurrent.futures
import numpy as np
import json
import imagehash
from PIL import Image # PIL is used for imagehash
from collections import defaultdict
import re
from pathlib import Path # Added for pathlib operations

# Attempt to import SuperResImageSelector
try:
    from sisr_image_selector import SuperResImageSelector # Ensure sisr_image_selector.py is accessible
    SISR_AVAILABLE = True
except ImportError:
    print("Warning: sisr_image_selector.py not found or SuperResImageSelector class cannot be imported. SISR filtering will be unavailable.")
    SISR_AVAILABLE = False
    class SuperResImageSelector: # Dummy class if import fails
        def __init__(self, *args, **kwargs):
            print("ERROR: SuperResImageSelector is not available.")
        def select_images(self, *args, **kwargs):
            print("ERROR: SuperResImageSelector.select_images called but class is not available.")
            return []
        def cleanup(self):
            pass


# Check CUDA availability and print OpenCV version
print(f"OpenCV version: {cv2.__version__}")
print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")

use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

if use_gpu:
    print("CUDA-capable GPU detected. Using GPU acceleration.")
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
             cuda_device = cv2.cuda.getDevice()
             print(f"CUDA device index being used: {cuda_device}")
             print(f"CUDA device name: {cv2.cuda.getDeviceName(cuda_device)}")
    except cv2.error as e:
        print(f"Could not get CUDA device details: {e}")
        use_gpu = False # Fallback if details fail
else:
    print("No CUDA-capable GPU detected. Using CPU.")

def update_progress(progress_file, stage, video=None, completed=False, item_key='videos'):
    progress = {}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse progress file {progress_file}. Starting fresh.")
            progress = {}

    if video: # 'video' here can mean a video filename or a video subfolder name
        if stage not in progress:
            progress[stage] = {item_key: []}
        if item_key not in progress[stage] or not isinstance(progress[stage][item_key], list):
             progress[stage][item_key] = []
        if video not in progress[stage][item_key]:
            progress[stage][item_key].append(video)
    elif completed:
        if stage not in progress:
             progress[stage] = {}
        progress[stage]['completed'] = True

    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=4)

def get_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
             print(f"Warning: Could not parse progress file {progress_file}. Returning empty progress.")
             return {}
    return {}

def detect_interlacing(video_path, probe_duration=10, frame_limit=300):
    # ... (detect_interlacing function - no changes from previous version) ...
    print(f"Probing interlacing for {os.path.basename(video_path)}...")
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-filter:v", "idet", "-an", "-sn", "-dn",
        "-t", str(probe_duration),
        "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,1)',idet", # Process 1 frame per second
        "-map", "0:v:0?", "-f", "null", "-threads", "1", "-nostats", "-i", video_path,
    ]
    try:
        process = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, check=False)
        stderr_output = process.stderr
        tff_match = re.search(r"Single frame detection: TFF:\s*(\d+)", stderr_output)
        bff_match = re.search(r"BFF:\s*(\d+)", stderr_output)
        prog_match = re.search(r"Progressive:\s*(\d+)", stderr_output)

        if tff_match and bff_match and prog_match:
            tff_count = int(tff_match.group(1))
            bff_count = int(bff_match.group(1))
            prog_count = int(prog_match.group(1))
            total_detected = tff_count + bff_count + prog_count
            print(f"  Idet results: TFF={tff_count}, BFF={bff_count}, Progressive={prog_count}")
            if total_detected == 0:
                print("  Warning: idet filter detected 0 frames. Assuming progressive.")
                return False
            interlaced_ratio = (tff_count + bff_count) / total_detected
            if interlaced_ratio > 0.25: # Adjusted threshold, more likely interlaced
                print(f"  Detected as INTERLACED (ratio: {interlaced_ratio:.2f})")
                return True
            else:
                print(f"  Detected as PROGRESSIVE (ratio: {interlaced_ratio:.2f})")
                return False
        else: # Fallback for different FFmpeg versions or output formats
            rep_neither_match = re.search(r"Repeated Fields: Neither:\s*(\d+)", stderr_output)
            rep_top_match = re.search(r"Top:\s*(\d+)", stderr_output)
            rep_bottom_match = re.search(r"Bottom:\s*(\d+)", stderr_output)
            if rep_neither_match and rep_top_match and rep_bottom_match:
                rep_neither = int(rep_neither_match.group(1))
                rep_top = int(rep_top_match.group(1))
                rep_bottom = int(rep_bottom_match.group(1))
                rep_total = rep_neither + rep_top + rep_bottom
                print(f"  Idet results (Repeated Fields): Neither={rep_neither}, Top={rep_top}, Bottom={rep_bottom}")
                if rep_total == 0:
                    print("  Warning: idet filter detected 0 frames (repeated fields). Assuming progressive.")
                    return False
                interlaced_ratio = (rep_top + rep_bottom) / rep_total
                if interlaced_ratio > 0.1: # Lower threshold for repeated fields
                    print(f"  Detected as INTERLACED based on repeated fields (ratio: {interlaced_ratio:.2f})")
                    return True
                else:
                    print(f"  Detected as PROGRESSIVE based on repeated fields (ratio: {interlaced_ratio:.2f})")
                    return False
            print(f"  Warning: Could not parse idet summary from ffmpeg output for {os.path.basename(video_path)}. Assuming progressive.")
            # print(f"FFMPEG stderr for debug:\n{stderr_output}") # Uncomment for debugging idet
            return False
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure FFmpeg is installed and in your system's PATH.")
        return False
    except Exception as e:
        print(f"Error during interlacing detection for {os.path.basename(video_path)}: {e}")
        return False


def extract_frames_ffmpeg(video_path, output_folder, begin_time, end_time, scene_threshold,
                          width=None, height=None, scale=None,
                          video_type='unknown', deinterlace_mode='none',
                          deinterlace_filter='bwdif=mode=send_frame:parity=auto'):
    # ... (extract_frames_ffmpeg function - no changes from previous version) ...
    os.makedirs(output_folder, exist_ok=True)
    apply_deinterlace = False
    interlace_detected_by_auto = False # Renamed for clarity
    if deinterlace_mode == 'auto':
        is_interlaced = detect_interlacing(video_path)
        if is_interlaced:
            apply_deinterlace = True
            interlace_detected_by_auto = True
    elif deinterlace_mode == 'both': apply_deinterlace = True
    elif deinterlace_mode == 'lr' and video_type == 'lr': apply_deinterlace = True
    elif deinterlace_mode == 'hr' and video_type == 'hr': apply_deinterlace = True

    ffmpeg_command = ["ffmpeg", "-i", video_path]
    if begin_time != "00:00:00" or end_time != "00:00:00":
        ffmpeg_command.extend(["-ss", begin_time, "-to", end_time])
    vf_filters = []
    if apply_deinterlace:
        vf_filters.append(deinterlace_filter)
        print(f"Applying deinterlace filter '{deinterlace_filter}' to {os.path.basename(video_path)}")
    vf_filters.append(f"select='gt(scene,{scene_threshold})'")
    if width and height: vf_filters.append(f"scale={width}:{height}")
    elif scale: vf_filters.append(f"scale=iw*{scale}:ih*{scale}")
    vf_filters.append("showinfo") # Keep showinfo for potential debugging, output is not parsed here
    ffmpeg_command.extend(["-vf", ",".join(vf_filters), "-vsync", "vfr", "-q:v", "2", f"{output_folder}/frame_%06d.png"])
    video_basename = os.path.basename(video_path)
    print(f"\nExecuting FFmpeg for {video_basename}:\n  Command: {' '.join(ffmpeg_command)}")
    try:
        process = subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "deprecated pixel format used" in process.stderr: # Check stderr for warnings
             print(f"  Warning: Deprecated pixel format detected in {video_basename}.")
        # Report deinterlacing status
        deinterlace_msg = ""
        if apply_deinterlace:
            deinterlace_msg = " (Deinterlaced"
            if interlace_detected_by_auto:
                deinterlace_msg += " - Auto-Detected)"
            else:
                deinterlace_msg += ")"
        print(f"Frames extracted successfully from {video_basename}{deinterlace_msg}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {video_basename}: {e}\n  FFmpeg stderr:\n{e.stderr}")
    except FileNotFoundError:
         print("Error: ffmpeg command not found. Please ensure FFmpeg is installed and in your system's PATH.")


def preprocess_videos(lr_input_folder, hr_input_folder, extracted_images_folder,
                      begin_time, end_time, scene_threshold,
                      lr_width=None, lr_height=None, lr_scale=None,
                      hr_width=None, hr_height=None, hr_scale=None,
                      deinterlace_mode='none', deinterlace_filter='bwdif=mode=send_frame:parity=auto',
                      progress_file=None):
    # ... (preprocess_videos function - no changes from previous version) ...
    progress = get_progress(progress_file)
    stage_name = 'preprocess'
    if stage_name in progress and progress[stage_name].get('completed'):
        print("Preprocessing already completed. Skipping...")
        return os.path.join(extracted_images_folder, "LR"), os.path.join(extracted_images_folder, "HR")

    lr_base_path = os.path.join(extracted_images_folder, "LR")
    hr_base_path = os.path.join(extracted_images_folder, "HR")
    os.makedirs(lr_base_path, exist_ok=True); os.makedirs(hr_base_path, exist_ok=True)
    allowed_extensions = ('.avi', '.mp4', '.mkv', '.ts', '.mpg', '.mpeg', '.mov', '.flv', '.wmv') # Added more
    lr_video_files_all = sorted([f for f in os.listdir(lr_input_folder) if f.lower().endswith(allowed_extensions)])
    hr_video_files_all = sorted([f for f in os.listdir(hr_input_folder) if f.lower().endswith(allowed_extensions)])
    lr_video_map = {f.lower(): f for f in lr_video_files_all}
    hr_video_map = {f.lower(): f for f in hr_video_files_all}
    common_lower = set(lr_video_map.keys()) & set(hr_video_map.keys())
    if len(common_lower) != len(lr_video_files_all) or len(common_lower) != len(hr_video_files_all):
         print(f"Warning: LR and HR video lists do not perfectly match. Processing only common videos.\n  LR files ({len(lr_video_files_all)}): {', '.join(lr_video_files_all)}\n  HR files ({len(hr_video_files_all)}): {', '.join(hr_video_files_all)}\n  Common files ({len(common_lower)}): {', '.join(sorted(list(common_lower)))}")
    video_files_to_process = sorted([lr_video_map[f_lower] for f_lower in common_lower])
    if not video_files_to_process:
         print("Error: No matching video files found. Exiting preprocessing.")
         return lr_base_path, hr_base_path
    processed_in_progress = progress.get(stage_name, {}).get('videos', [])
    for video_file in tqdm(video_files_to_process, desc="Preprocessing videos"):
        if video_file in processed_in_progress:
            print(f"Skipping already processed video: {video_file}")
            continue
        video_name_no_ext = os.path.splitext(video_file)[0] # Use this for subfolder names
        hr_video_file = hr_video_map[video_file.lower()] # Get original HR filename
        print(f"\n--- Processing {video_file} ---")
        lr_video_path = os.path.join(lr_input_folder, video_file)
        lr_output_folder = os.path.join(lr_base_path, video_name_no_ext) # Use video_name_no_ext
        print(f"Extracting LR frames for {video_file}...")
        extract_frames_ffmpeg(lr_video_path, lr_output_folder, begin_time, end_time, scene_threshold, lr_width, lr_height, lr_scale, 'lr', deinterlace_mode, deinterlace_filter)
        hr_video_path = os.path.join(hr_input_folder, hr_video_file)
        hr_output_folder = os.path.join(hr_base_path, video_name_no_ext) # Use video_name_no_ext
        print(f"Extracting HR frames for {hr_video_file}...")
        extract_frames_ffmpeg(hr_video_path, hr_output_folder, begin_time, end_time, scene_threshold, hr_width, hr_height, hr_scale, 'hr', deinterlace_mode, deinterlace_filter)
        update_progress(progress_file, stage_name, video=video_file) # Track by original video filename
    update_progress(progress_file, stage_name, completed=True)
    print("--- Preprocessing Finished ---")
    return lr_base_path, hr_base_path


def autocrop_single_image(image_path, output_path, black_threshold=10, min_dimension=32):
    # ... (autocrop_single_image function - no changes from previous version) ...
    img = cv2.imread(image_path)
    if img is None:
        return False

    original_height, original_width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY)

    contours_tuple = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_tuple) == 2: contours = contours_tuple[0]
    else: contours = contours_tuple[1]

    if not contours: return False

    x_min, y_min, x_max, y_max = original_width, original_height, 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x); y_min = min(y_min, y)
        x_max = max(x_max, x + w); y_max = max(y_max, y + h)

    crop_w, crop_h = x_max - x_min, y_max - y_min
    if crop_w < min_dimension or crop_h < min_dimension: return False
    if x_min == 0 and y_min == 0 and x_max == original_width and y_max == original_height: return False

    try: cv2.imwrite(output_path, img[y_min:y_max, x_min:x_max]); return True
    except Exception: return False

def autocrop_frames_in_folders(base_image_path, black_threshold, min_dimension, progress_file, stage_name_suffix=""):
    # ... (autocrop_frames_in_folders function - no changes from previous version) ...
    stage_name = f'autocrop{stage_name_suffix}'
    progress = get_progress(progress_file)
    if stage_name in progress and progress[stage_name].get('completed'):
        print(f"Autocropping for {stage_name_suffix or 'images'} already completed. Skipping...")
        return

    print(f"--- Starting Autocropping for {os.path.basename(base_image_path)} ({stage_name_suffix}) ---")
    try: video_folders = [f for f in os.listdir(base_image_path) if os.path.isdir(os.path.join(base_image_path, f))]
    except FileNotFoundError: print(f"Error: Base path for autocropping not found: {base_image_path}"); update_progress(progress_file, stage_name, completed=True); return
    if not video_folders: print(f"No video folders in {base_image_path} for autocropping."); update_progress(progress_file, stage_name, completed=True); return

    processed_videos_in_progress = progress.get(stage_name, {}).get('videos', []) # Default item_key is 'videos'
    total_processed_all, total_cropped_all = 0, 0
    for video_folder_name in tqdm(video_folders, desc=f"Autocropping videos in {os.path.basename(base_image_path)}"):
        if video_folder_name in processed_videos_in_progress: print(f"Skipping already autocropped: {video_folder_name}"); continue
        current_video_frames_path = os.path.join(base_image_path, video_folder_name)
        try: image_files = [f for f in os.listdir(current_video_frames_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except FileNotFoundError: print(f"Warning: Image folder not found: {current_video_frames_path}. Skipping."); update_progress(progress_file, stage_name, video=video_folder_name); continue
        if not image_files: update_progress(progress_file, stage_name, video=video_folder_name); continue
        total_processed_all += len(image_files); cropped_this_video = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(autocrop_single_image, os.path.join(current_video_frames_path, img_f), os.path.join(current_video_frames_path, img_f), black_threshold, min_dimension) for img_f in image_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Cropping {video_folder_name}", leave=False):
                try:
                    if future.result(): cropped_this_video += 1
                except Exception as e: print(f"Error in autocropping worker for {video_folder_name}: {e}")
        if cropped_this_video > 0: print(f"  Cropped {cropped_this_video} / {len(image_files)} images in {video_folder_name}.")
        total_cropped_all += cropped_this_video; update_progress(progress_file, stage_name, video=video_folder_name) # Track by video_folder_name
    print(f"--- Autocropping Finished for {os.path.basename(base_image_path)} ({stage_name_suffix}) ---\n    Total images processed: {total_processed_all}\n    Total images actually cropped: {total_cropped_all}")
    update_progress(progress_file, stage_name, completed=True)


# --- NEW FUNCTION for Low Information Filtering ---
def check_image_variance(image_path, variance_threshold):
    """Checks if the variance of an image is below a threshold."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            # print(f"Warning: Could not read image {image_path} for variance check.")
            return False # Treat as not low variance if unreadable
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = np.var(gray_img)
        if var < variance_threshold:
            # print(f"  Image {os.path.basename(image_path)} has low variance: {var:.2f}")
            return True # Is low variance
        return False # Not low variance
    except Exception as e:
        # print(f"Warning: Error checking variance for {image_path}: {e}")
        return False

def filter_low_information_images(base_lr_path_str, base_hr_path_str, variance_threshold, check_hr_too, progress_file):
    stage_name = 'low_info_filter'
    progress = get_progress(progress_file)
    if stage_name in progress and progress[stage_name].get('completed'):
        print("Low information image filtering already completed. Skipping...")
        return

    print(f"--- Starting Low Information Image Filtering ---")
    print(f"  LR Base Path: {base_lr_path_str}")
    print(f"  HR Base Path: {base_hr_path_str}")
    print(f"  Variance Threshold: {variance_threshold}")
    print(f"  Check HR images too: {check_hr_too}")

    base_lr_path = Path(base_lr_path_str)
    base_hr_path = Path(base_hr_path_str)

    if not base_lr_path.is_dir():
        print(f"Error: LR base path for low info filtering not found: {base_lr_path}")
        update_progress(progress_file, stage_name, completed=True)
        return

    video_folders_lr = [f for f in os.listdir(base_lr_path) if os.path.isdir(base_lr_path / f)]
    if not video_folders_lr:
        print(f"No video subfolders found in {base_lr_path} for low info filtering.")
        update_progress(progress_file, stage_name, completed=True)
        return

    processed_video_folders = progress.get(stage_name, {}).get('video_folders', [])
    total_lr_removed = 0
    total_hr_removed = 0

    for video_folder_name in tqdm(video_folders_lr, desc="Filtering low info images by video folder"):
        if video_folder_name in processed_video_folders:
            print(f"Skipping already processed video folder for low info: {video_folder_name}")
            continue

        current_lr_video_path = base_lr_path / video_folder_name
        current_hr_video_path = base_hr_path / video_folder_name # Corresponding HR video folder

        try:
            lr_image_files = [f for f in os.listdir(current_lr_video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except FileNotFoundError:
            print(f"Warning: LR image folder not found: {current_lr_video_path}. Skipping.")
            update_progress(progress_file, stage_name, video=video_folder_name, item_key='video_folders')
            continue

        removed_in_folder_lr = 0
        removed_in_folder_hr = 0

        for lr_img_file in tqdm(lr_image_files, desc=f"Checking LR in {video_folder_name}", leave=False):
            lr_img_path = current_lr_video_path / lr_img_file
            remove_pair = False

            if check_image_variance(lr_img_path, variance_threshold):
                remove_pair = True
                # print(f"  Flagged LR {lr_img_path.name} for removal (low variance).")

            if not remove_pair and check_hr_too:
                hr_img_path = current_hr_video_path / lr_img_file # Assumes same filename
                if hr_img_path.exists():
                    if check_image_variance(hr_img_path, variance_threshold):
                        remove_pair = True
                        # print(f"  Flagged HR {hr_img_path.name} for removal (low variance), also removing LR.")
                # else:
                    # print(f"  HR counterpart {hr_img_path.name} not found for HR check.")


            if remove_pair:
                try:
                    if lr_img_path.exists():
                        lr_img_path.unlink()
                        removed_in_folder_lr += 1
                    hr_counterpart_path = current_hr_video_path / lr_img_file
                    if hr_counterpart_path.exists():
                        hr_counterpart_path.unlink()
                        removed_in_folder_hr += 1
                except OSError as e:
                    print(f"Error removing low info image {lr_img_path.name} or its HR counterpart: {e}")
        
        if removed_in_folder_lr > 0:
            print(f"  Removed {removed_in_folder_lr} LR images (and {removed_in_folder_hr} HR counterparts) from {video_folder_name} due to low information.")
        total_lr_removed += removed_in_folder_lr
        total_hr_removed += removed_in_folder_hr
        update_progress(progress_file, stage_name, video=video_folder_name, item_key='video_folders')

    print(f"--- Low Information Image Filtering Finished ---")
    print(f"    Total LR images removed: {total_lr_removed}")
    print(f"    Total HR images removed: {total_hr_removed}")
    update_progress(progress_file, stage_name, completed=True)


def process_image_pair(lr_img, hr_images, lr_folder, hr_folder, output_lr_folder, output_hr_folder, match_threshold, resize_height, resize_width, video_name):
    # ... (process_image_pair function - no changes from previous version) ...
    global use_gpu # Allow modification if GPU fails mid-process for a frame
    lr_img_path = os.path.join(lr_folder, lr_img)
    lr_frame = cv2.imread(lr_img_path)
    if lr_frame is None: print(f"Warning: Could not read LR image {lr_img_path}. Skipping."); return 0

    lr_gray, lr_frame_resized = None, None
    gpu_active_for_lr = False
    if use_gpu:
        try:
            lr_frame_gpu = cv2.cuda_GpuMat(); lr_frame_gpu.upload(lr_frame)
            lr_frame_gpu_resized = cv2.cuda.resize(lr_frame_gpu, (resize_width, resize_height))
            lr_gray_gpu = cv2.cuda.cvtColor(lr_frame_gpu_resized, cv2.COLOR_BGR2GRAY)
            gpu_active_for_lr = True
        except cv2.error as e: print(f"Warning: GPU error processing LR image {lr_img} ({e}). Falling back to CPU for this image."); use_gpu = False # Fallback for this pair
    if not gpu_active_for_lr: # CPU path for LR or fallback
        try:
             lr_frame_resized = cv2.resize(lr_frame, (resize_width, resize_height))
             lr_gray = cv2.cvtColor(lr_frame_resized, cv2.COLOR_BGR2GRAY)
        except cv2.error as e: print(f"Error: CPU resize/cvtColor failed for LR image {lr_img_path}: {e}. Skipping pair."); return 0

    best_match_hr_filename, best_score = None, float('inf')
    for hr_img_filename in hr_images:
        hr_img_path = os.path.join(hr_folder, hr_img_filename)
        hr_frame = cv2.imread(hr_img_path)
        if hr_frame is None: continue
        current_score, gpu_active_for_hr = float('inf'), False
        if use_gpu and gpu_active_for_lr: # Only try GPU for HR if LR GPU was successful
             try:
                 hr_frame_gpu = cv2.cuda_GpuMat(); hr_frame_gpu.upload(hr_frame)
                 hr_frame_gpu_resized = cv2.cuda.resize(hr_frame_gpu, (resize_width, resize_height))
                 hr_gray_gpu = cv2.cuda.cvtColor(hr_frame_gpu_resized, cv2.COLOR_BGR2GRAY)
                 matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_SQDIFF_NORMED) # CV_8UC1 for grayscale
                 result_gpu = matcher.match(hr_gray_gpu, lr_gray_gpu) # Target (HR) searched in Source (LR)
                 minVal, _, _, _ = cv2.cuda.minMaxLoc(result_gpu)
                 current_score = minVal; gpu_active_for_hr = True
             except cv2.error as e: print(f"Warning: GPU error processing HR image {hr_img_filename} ({e}). Fallback to CPU for this candidate.")
        if not gpu_active_for_hr: # CPU path for HR or fallback
             try:
                hr_frame_resized = cv2.resize(hr_frame, (resize_width, resize_height))
                hr_gray = cv2.cvtColor(hr_frame_resized, cv2.COLOR_BGR2GRAY)
                # Ensure lr_gray is available if LR processing fell back to CPU
                if lr_gray is None and lr_frame_resized is not None: # This lr_frame_resized is from CPU path
                     lr_gray = cv2.cvtColor(lr_frame_resized, cv2.COLOR_BGR2GRAY)
                if lr_gray is not None and hr_gray is not None:
                     # Template matching: hr_gray is template, lr_gray is source
                     result = cv2.matchTemplate(lr_gray, hr_gray, cv2.TM_SQDIFF_NORMED) # Search HR_template in LR_image
                     minVal, _, _, _ = cv2.minMaxLoc(result); current_score = minVal
                else: print(f"Error: Missing grayscale data for CPU match ({lr_img}, {hr_img_filename})."); current_score = float('inf')
             except cv2.error as e: print(f"Error: CPU resize/match failed for HR image {hr_img_path}: {e}. Skipping candidate."); current_score = float('inf')
        if current_score < best_score: best_score, best_match_hr_filename = current_score, hr_img_filename
    if best_match_hr_filename is not None and best_score < match_threshold:
        try:
            lr_src_path = os.path.join(lr_folder, lr_img)
            hr_src_path = os.path.join(hr_folder, best_match_hr_filename)
            base_name_no_ext = os.path.splitext(lr_img)[0]
            lr_dst_path = os.path.join(output_lr_folder, f"{video_name}_{base_name_no_ext}.png")
            hr_dst_path = os.path.join(output_hr_folder, f"{video_name}_{base_name_no_ext}.png")
            shutil.copy(lr_src_path, lr_dst_path); shutil.copy(hr_src_path, hr_dst_path)
            return 1
        except Exception as e: print(f"Error copying matched pair {lr_img}/{best_match_hr_filename}: {e}"); return 0
    return 0

def process_folder_pair(lr_base_path, hr_base_path, output_lr_base_path, output_hr_base_path, match_threshold, resize_height, resize_width, distance_modifier, progress_file):
    # ... (process_folder_pair function - no changes from previous version) ...
    progress = get_progress(progress_file)
    stage_name = 'match'
    if stage_name in progress and progress[stage_name].get('completed'): print("Matching already completed. Skipping..."); return
    try:
        # Ensure paths exist before listing
        if not os.path.isdir(lr_base_path): print(f"Error: LR base path for matching not found: {lr_base_path}"); return
        if not os.path.isdir(hr_base_path): print(f"Error: HR base path for matching not found: {hr_base_path}"); return
        
        lr_folders = {f for f in os.listdir(lr_base_path) if os.path.isdir(os.path.join(lr_base_path, f))}
        hr_folders = {f for f in os.listdir(hr_base_path) if os.path.isdir(os.path.join(hr_base_path, f))}
        common_folders = sorted(list(lr_folders & hr_folders))
    except FileNotFoundError as e: print(f"Error: Input folder not found for matching: {e}. Check EXTRACTED paths."); return # Should be caught by isdir
    if not common_folders: print("Error: No common video folders found for matching (e.g., in EXTRACTED/LR and EXTRACTED/HR)."); return
    
    print(f"Found {len(common_folders)} common video folders for matching.")
    processed_in_progress = progress.get(stage_name, {}).get('videos', []) # Default item_key is 'videos'

    for folder_name in common_folders: 
        if folder_name in processed_in_progress: print(f"Skipping already matched folder: {folder_name}"); continue
        
        lr_folder_path = os.path.join(lr_base_path, folder_name); hr_folder_path = os.path.join(hr_base_path, folder_name)
        
        # Check if these specific subfolders exist
        if not os.path.isdir(lr_folder_path): print(f"Warning: LR subfolder not found: {lr_folder_path}. Skipping for matching."); continue
        if not os.path.isdir(hr_folder_path): print(f"Warning: HR subfolder not found: {hr_folder_path}. Skipping for matching."); continue

        try:
            lr_images = sorted([f for f in os.listdir(lr_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            hr_images = sorted([f for f in os.listdir(hr_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        except FileNotFoundError: # Should not happen if isdir checks pass, but as a safeguard
            print(f"Warning: Image folders for '{folder_name}' became unavailable. Skipping."); continue
        
        if not lr_images: print(f"Warning: No LR images in {lr_folder_path} for '{folder_name}'. Skipping."); continue
        if not hr_images: print(f"Warning: No HR images in {hr_folder_path} for '{folder_name}'. Skipping."); continue
        
        current_pair_count = 0 
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_image_pair, lr_img, hr_images, lr_folder_path, hr_folder_path, output_lr_base_path, output_hr_base_path, match_threshold, resize_height, resize_width, folder_name) for lr_img in lr_images]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Matching images for {folder_name}"):
                try: current_pair_count += future.result()
                except Exception as e: print(f"Error in matching worker for {folder_name}: {e}")
        print(f"Total pairs created for {folder_name}: {current_pair_count}")
        update_progress(progress_file, stage_name, video=folder_name) # Track by folder_name
    update_progress(progress_file, stage_name, completed=True)
    print("--- Matching Finished ---")


def filter_similar_images(matched_lr_path, matched_hr_path, similarity_threshold=4, progress_file=None):
    # ... (filter_similar_images function - no changes from previous version) ...
    progress = get_progress(progress_file)
    stage_name = 'filter_phash' # Changed stage name for clarity
    if stage_name in progress and progress[stage_name].get('completed'): print("Similarity filtering (phash) already completed. Skipping..."); return 0
    print("Starting similarity detection and filtering (perceptual hash)...") # Clarified title
    try:
        if not os.path.isdir(matched_lr_path): print(f"Error: Matched LR path for filtering not found: {matched_lr_path}"); return 0
        if not os.path.isdir(matched_hr_path): print(f"Error: Matched HR path for filtering not found: {matched_hr_path}"); return 0
        lr_images_all = sorted([f for f in os.listdir(matched_lr_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except Exception as e: print(f"Error listing matched images for filtering: {e}"); return 0
    if not lr_images_all: print("No matched images found to filter (phash)."); update_progress(progress_file, stage_name, completed=True); return 0

    video_groups = defaultdict(list)
    for img_filename in lr_images_all: 
        # Regex to capture video name: assumes format VIDEO-NAME_frame_XXXXXX.png
        match = re.match(r"([^_]+)_frame_\d+\.png", img_filename) 
        if match: video_groups[match.group(1)].append(img_filename)
        else: print(f"Warning: Could not determine video name from '{img_filename}' for phash filter. Grouping as 'unknown_video_group'."); video_groups['unknown_video_group'].append(img_filename)

    total_removed_count = 0 
    processed_videos_in_progress = progress.get(stage_name, {}).get('videos', []) # Default item_key is 'videos'
    for video_name, images_in_group in tqdm(video_groups.items(), desc="Filtering similar images (phash) by video"): 
        if video_name == 'unknown_video_group': 
            print("Skipping 'unknown_video_group' for phash filtering as filenames are not standard.")
            continue
        if video_name in processed_videos_in_progress: print(f"Skipping already phash-filtered video group: {video_name}"); continue
        
        kept_hashes_and_filenames = {} 
        filenames_to_remove = set() 
        images_in_group.sort() 

        for img_filename in tqdm(images_in_group, desc=f"Hashing {video_name}", leave=False):
            lr_image_path = os.path.join(matched_lr_path, img_filename)
            hr_image_path = os.path.join(matched_hr_path, img_filename) 
            if not os.path.exists(lr_image_path) or not os.path.exists(hr_image_path):
                 print(f"Warning: Missing LR or HR file for {img_filename} during phash filtering. Skipping.")
                 continue
            try:
                with Image.open(lr_image_path) as img_pil: current_img_hash = imagehash.phash(img_pil) 
            except Exception as e: print(f"Warning: Could not hash {lr_image_path}: {e}. Skipping this image."); continue

            is_similar_found = False 
            hashes_to_check = list(kept_hashes_and_filenames.keys()) 
            for kept_hash in hashes_to_check:
                kept_filename = kept_hashes_and_filenames[kept_hash]
                if (current_img_hash - kept_hash) < similarity_threshold:
                    try:
                        current_lr_size = os.path.getsize(lr_image_path)
                        kept_lr_size = os.path.getsize(os.path.join(matched_lr_path, kept_filename))
                        if current_lr_size > kept_lr_size:
                            filenames_to_remove.add(kept_filename)
                            del kept_hashes_and_filenames[kept_hash] 
                            kept_hashes_and_filenames[current_img_hash] = img_filename 
                        else:
                            filenames_to_remove.add(img_filename)
                        is_similar_found = True; break
                    except Exception as e: 
                        print(f"Warning: Error comparing sizes for {img_filename}/{kept_filename}: {e}. Removing current.");
                        filenames_to_remove.add(img_filename); is_similar_found = True; break
            if not is_similar_found: kept_hashes_and_filenames[current_img_hash] = img_filename

        removed_count_this_group = 0 
        for img_filename_to_remove in filenames_to_remove:
            try:
                lr_file_to_remove = os.path.join(matched_lr_path, img_filename_to_remove)
                hr_file_to_remove = os.path.join(matched_hr_path, img_filename_to_remove)
                if os.path.exists(lr_file_to_remove): os.remove(lr_file_to_remove)
                if os.path.exists(hr_file_to_remove): os.remove(hr_file_to_remove)
                removed_count_this_group += 1
            except OSError as e: print(f"Error removing similar image file {img_filename_to_remove}: {e}")
        if removed_count_this_group > 0 : print(f"Removed {removed_count_this_group} similar image pairs (phash) from group {video_name}.")
        total_removed_count += removed_count_this_group
        update_progress(progress_file, stage_name, video=video_name) # Track by video_name

    print(f"\nSimilarity filtering (phash) completed. Total removed: {total_removed_count} similar image pairs.")
    update_progress(progress_file, stage_name, completed=True)
    print("--- Phash Filtering Finished ---")
    return total_removed_count


def align_images(output_lr_base_path, output_hr_base_path, aligned_output_base_path, img_align_scale, progress_file=None):
    # ... (align_images function - no changes from previous version) ...
    progress = get_progress(progress_file)
    stage_name = 'align'
    if stage_name in progress and progress[stage_name].get('completed'): print("Alignment already completed. Skipping..."); return
    if not shutil.which("ImgAlign"): print("Error: ImgAlign executable not found in PATH. Skipping alignment."); return
    if os.path.exists(aligned_output_base_path): print(f"Removing existing alignment output directory: {aligned_output_base_path}"); shutil.rmtree(aligned_output_base_path)
    os.makedirs(aligned_output_base_path); print(f"Created fresh alignment output directory: {aligned_output_base_path}")
    try:
        if not os.path.isdir(output_lr_base_path): print(f"Error: Matched LR path for alignment not found: {output_lr_base_path}"); return
        lr_images_for_alignment = sorted([f for f in os.listdir(output_lr_base_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) # Renamed
        if not lr_images_for_alignment: print("No matched images found to align."); update_progress(progress_file, stage_name, completed=True); return
    except Exception as e: print(f"Error listing images for alignment: {e}"); return
    print(f"Starting image alignment for {len(lr_images_for_alignment)} pairs...")

    video_groups = defaultdict(list)
    for img_filename in lr_images_for_alignment: # Renamed
        match = re.match(r"([^_]+)_frame_\d+\.png", img_filename) # Consistent regex
        if match: video_groups[match.group(1)].append(img_filename)
        else: print(f"Warning: Could not determine video name from '{img_filename}' for alignment. Grouping as 'unknown_alignment_group'."); video_groups['unknown_alignment_group'].append(img_filename)

    processed_videos_in_progress = progress.get(stage_name, {}).get('videos', []) # Default item_key 'videos'
    final_aligned_lr_path = os.path.join(aligned_output_base_path, "LR")
    final_aligned_hr_path = os.path.join(aligned_output_base_path, "HR")
    final_aligned_overlay_path = os.path.join(aligned_output_base_path, "Overlay")
    os.makedirs(final_aligned_lr_path, exist_ok=True); os.makedirs(final_aligned_hr_path, exist_ok=True); os.makedirs(final_aligned_overlay_path, exist_ok=True)

    for video_name, images_in_group in tqdm(video_groups.items(), desc="Aligning videos", unit="video"): # Renamed
        if video_name == 'unknown_alignment_group': print(f"Skipping alignment for 'unknown_alignment_group'."); continue
        if video_name in processed_videos_in_progress: print(f"Skipping already aligned video: {video_name}"); continue
        print(f"\nAligning images for video group: {video_name}...")
        
        temp_batch_base = os.path.join(aligned_output_base_path, f"__temp_align_{video_name}")
        temp_batch_lr_input = os.path.join(temp_batch_base, "lr_input_batch")
        temp_batch_hr_input = os.path.join(temp_batch_base, "hr_input_batch")
        temp_batch_aligned_output = os.path.join(temp_batch_base, "aligned_output_batch")
        # Clean up previous temp batch if it exists (e.g. from a failed run)
        if os.path.exists(temp_batch_base): shutil.rmtree(temp_batch_base)
        os.makedirs(temp_batch_lr_input, exist_ok=True); os.makedirs(temp_batch_hr_input, exist_ok=True); os.makedirs(temp_batch_aligned_output, exist_ok=True)

        print(f"  Copying {len(images_in_group)} pairs to temporary batch folders for {video_name}...")
        valid_images_for_this_batch = [] # Renamed
        for img_filename in tqdm(images_in_group, desc=f"Preparing batch {video_name}", leave=False):
            src_lr = os.path.join(output_lr_base_path, img_filename)
            src_hr = os.path.join(output_hr_base_path, img_filename) 
            if not os.path.exists(src_lr) or not os.path.exists(src_hr):
                print(f"Warning: Missing source LR ({src_lr}) or HR ({src_hr}) for {img_filename}. Skipping this pair for alignment.")
                continue
            try:
                shutil.copy(src_lr, os.path.join(temp_batch_lr_input, img_filename))
                shutil.copy(src_hr, os.path.join(temp_batch_hr_input, img_filename))
                valid_images_for_this_batch.append(img_filename)
            except Exception as e: print(f"Error copying {img_filename} for alignment batch: {e}. Skipping pair.")

        if not valid_images_for_this_batch or not os.listdir(temp_batch_lr_input):
             print(f"  No valid image pairs found or copied for {video_name}. Skipping alignment for this video.")
             if os.path.exists(temp_batch_base): shutil.rmtree(temp_batch_base)
             update_progress(progress_file, stage_name, video=video_name); continue

        print(f"  Running ImgAlign for {video_name} batch ({len(valid_images_for_this_batch)} pairs)...")
        img_align_cmd = ["ImgAlign", "-s", str(img_align_scale), "-m", "0", "-g", temp_batch_hr_input, "-l", temp_batch_lr_input, "-c", "-i", "-1", "-j", "-ai", "-o", temp_batch_aligned_output]
        print(f"    ImgAlign Command: {' '.join(img_align_cmd)}")
        try:
            process = subprocess.Popen(img_align_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            stdout, stderr = process.communicate() 
            if process.returncode != 0:
                print(f"  Error running ImgAlign for {video_name}. Return Code: {process.returncode}\n    ImgAlign Stderr: {stderr.strip()}"); # Added strip()
                if os.path.exists(temp_batch_base): shutil.rmtree(temp_batch_base)
                update_progress(progress_file, stage_name, video=video_name); continue 
            else: print(f"  ImgAlign completed successfully for {video_name}.")
        except FileNotFoundError: print("  Error: ImgAlign command not found. Ensure it's in PATH."); shutil.rmtree(temp_batch_base); return 
        except Exception as e: print(f"  Exception during ImgAlign execution for {video_name}: {e}"); shutil.rmtree(temp_batch_base); update_progress(progress_file, stage_name, video=video_name); continue

        print(f"  Moving aligned files for {video_name}...")
        imgalign_actual_output_subfolder = os.path.join(temp_batch_aligned_output, 'Output')
        artifacts_filepath = os.path.join(imgalign_actual_output_subfolder, "Artifacts.txt") 
        filenames_to_skip_due_to_artifacts = set() 
        if os.path.exists(artifacts_filepath):
             print(f"  Found Artifacts.txt for {video_name}. Will skip listed files.")
             try:
                 with open(artifacts_filepath, 'r') as f: filenames_to_skip_due_to_artifacts = {line.strip() for line in f if line.strip()}
                 print(f"    Artifacts listed: {', '.join(filenames_to_skip_due_to_artifacts) if filenames_to_skip_due_to_artifacts else 'None'}")
             except Exception as e: print(f"  Warning: Could not read Artifacts.txt: {e}")

        moved_pair_count, skipped_artifact_set_count = 0, 0 
        for folder_type in ['HR', 'LR', 'Overlay']: 
            src_folder_path = os.path.join(imgalign_actual_output_subfolder, folder_type) 
            dst_folder_path_map = {'HR': final_aligned_hr_path, 'LR': final_aligned_lr_path, 'Overlay': final_aligned_overlay_path} 
            dst_target_folder = dst_folder_path_map[folder_type] 
            if os.path.exists(src_folder_path):
                for file_in_src in os.listdir(src_folder_path): 
                    base_filename_no_ext, _ = os.path.splitext(file_in_src) 
                    if base_filename_no_ext in filenames_to_skip_due_to_artifacts:
                        if folder_type == 'LR': skipped_artifact_set_count += 1 
                        continue 
                    try: shutil.move(os.path.join(src_folder_path, file_in_src), os.path.join(dst_target_folder, file_in_src))
                    except Exception as e: print(f"  Error moving aligned file {file_in_src}: {e}")
                    if folder_type == 'LR': moved_pair_count += 1
        print(f"  Moved {moved_pair_count} aligned pairs for {video_name}. Skipped {skipped_artifact_set_count} artifact-marked sets.")
        print(f"  Cleaning up temporary batch folder for {video_name}."); shutil.rmtree(temp_batch_base)
        update_progress(progress_file, stage_name, video=video_name)
    update_progress(progress_file, stage_name, completed=True)
    print("\n--- Image Alignment Process Completed ---")


def filter_aligned_with_sisr(aligned_lr_input_folder_str, aligned_hr_input_folder_str,
                             final_sisr_output_lr_folder_str, final_sisr_output_hr_folder_str,
                             sisr_min_clip_distance, sisr_complexity_threshold, sisr_max_brightness,
                             progress_file):
    # ... (filter_aligned_with_sisr function - no changes from previous version) ...
    global SISR_AVAILABLE
    if not SISR_AVAILABLE:
        print("SISR filtering stage skipped as SuperResImageSelector is not available.")
        return

    stage_name = 'sisr_filter'
    progress = get_progress(progress_file)
    if stage_name in progress and progress[stage_name].get('completed'):
        print("SISR filtering already completed. Skipping...")
        return

    print(f"--- Starting SISR Filtering on Aligned Images ---")
    print(f"  LR Input (Aligned): {aligned_lr_input_folder_str}")
    print(f"  HR Input (Aligned): {aligned_hr_input_folder_str}")
    print(f"  LR Output (SISR-Filtered): {final_sisr_output_lr_folder_str}")
    print(f"  HR Output (SISR-Filtered): {final_sisr_output_hr_folder_str}")
    print(f"  Config: Min CLIP Distance={sisr_min_clip_distance}, Complexity Threshold={sisr_complexity_threshold}, Max Brightness={sisr_max_brightness}")

    aligned_lr_input_path = Path(aligned_lr_input_folder_str)
    aligned_hr_input_path = Path(aligned_hr_input_folder_str)
    final_sisr_output_lr_path = Path(final_sisr_output_lr_folder_str)
    final_sisr_output_hr_path = Path(final_sisr_output_hr_folder_str)

    os.makedirs(final_sisr_output_lr_path, exist_ok=True)
    os.makedirs(final_sisr_output_hr_path, exist_ok=True)

    if not aligned_lr_input_path.is_dir() or not os.listdir(aligned_lr_input_path):
        print(f"Error: Aligned LR input folder '{aligned_lr_input_path}' is empty or not found. Cannot perform SISR filtering.")
        update_progress(progress_file, stage_name, completed=True) 
        return
    
    selected_lr_paths_from_sisr = []
    selector_instance = None # To ensure cleanup is called
    try:
        print("Initializing SuperResImageSelector...")
        # Pass None for output_folder to sisr_image_selector so it doesn't create its own default output
        # It will still create its 'temp_...' dir for checkpoints.
        selector_instance = SuperResImageSelector(input_folder=str(aligned_lr_input_path), output_folder=None) 
        selector_instance.max_brightness = sisr_max_brightness

        print("Running SISR selection logic on aligned LR images...")
        selected_lr_paths_from_sisr = selector_instance.select_images(
            min_distance=sisr_min_clip_distance,
            complexity_threshold=sisr_complexity_threshold
        )        
    except Exception as e:
        print(f"Error during SISR selection process: {e}")
        import traceback
        traceback.print_exc()
        return # Don't mark as completed on error
    finally:
        if selector_instance:
            print("Cleaning up SISR temporary files...")
            selector_instance.cleanup() 

    if not selected_lr_paths_from_sisr:
        print("SISR selector returned no images. No files will be copied for this stage.")
        update_progress(progress_file, stage_name, completed=True)
        return

    print(f"SISR selector identified {len(selected_lr_paths_from_sisr)} LR images. Copying selected LR/HR pairs...")
    copied_pair_count = 0
    for lr_selected_path_obj in tqdm(selected_lr_paths_from_sisr, desc="Copying SISR selected pairs"):
        lr_filename = lr_selected_path_obj.name
        hr_corresponding_path = aligned_hr_input_path / lr_filename

        if not hr_corresponding_path.exists():
            print(f"Warning: Corresponding aligned HR file not found for {lr_selected_path_obj} at {hr_corresponding_path}. Skipping this pair.")
            continue
        try:
            shutil.copy2(lr_selected_path_obj, final_sisr_output_lr_path / lr_filename)
            shutil.copy2(hr_corresponding_path, final_sisr_output_hr_path / lr_filename)
            copied_pair_count += 1
        except Exception as e:
            print(f"Error copying SISR-selected pair {lr_filename}: {e}")

    print(f"Successfully copied {copied_pair_count} SISR-filtered image pairs.")
    update_progress(progress_file, stage_name, completed=True)
    print(f"--- SISR Filtering Finished ---")


if __name__ == "__main__":
    # --- Configuration Options ---

    # Input/Output Paths
    lr_input_video_folder = "E:\\BTAS\\BTAS_DVD_LR"  # Path to the folder containing Low-Resolution videos
    hr_input_video_folder = "E:\\BTAS\\BTAS_BD_HR" # Path to the folder containing High-Resolution videos
    output_base_folder = "E:\\BTAS\\Output_SRDC_v2_Test" # Base directory where all processed files will be saved

    # Frame Extraction & Preprocessing
    begin_time = "00:00:00"  # Start time for frame extraction "HH:MM:SS" (00:00:00 for beginning of video)
    end_time = "00:00:00"    # End time for frame extraction "HH:MM:SS" (00:00:00 for end of video, processes whole video if both are 00:00:00)
    scene_threshold = 0.23   # Scene change detection sensitivity for FFmpeg (0.0 to 1.0).
                            # Lower values are more sensitive, extracting more frames near scene changes.
    lr_width, lr_height = None, None # Target width and height for LR frames. Set to None to keep original dimensions.
                                     # If both are set, frames will be resized to these exact dimensions.
    lr_scale = None          # Alternatively, set a scale factor for LR frames (e.g., 0.5 for half size).
                             # This overrides lr_width/lr_height if set. None to disable scaling.
    hr_width, hr_height = None, None # Target width and height for HR frames. Set to None to keep original.
    hr_scale = None          # Alternatively, set a scale factor for HR frames. Overrides hr_width/hr_height.

    # Deinterlacing Options
    # Determines if and how deinterlacing is applied during frame extraction.
    # Options:
    #   'none': No deinterlacing.
    #   'lr': Deinterlace only LR videos.
    #   'hr': Deinterlace only HR videos.
    #   'both': Deinterlace both LR and HR videos.
    #   'auto': Attempt to auto-detect interlacing for each video and deinterlace if detected.
    DEINTERLACE_MODE = 'none' # Options: 'none', 'lr', 'hr', 'both', 'auto'
    # FFmpeg filter string for deinterlacing.
    # 'bwdif' (Bob Weaver Deinterlacing Filter) is generally good quality.
    # 'yadif' (Yet Another Deinterlacing Filter) is faster but may have lower quality.
    # Add options like ':mode=send_frame:parity=auto' as needed by the filter.
    DEINTERLACE_FILTER = 'bwdif=mode=send_frame:parity=auto' # Example: High quality bwdif options

    # Autocropping Configuration
    CROP_BLACK_BORDERS = True  # Set to True to enable automatic black border cropping after frame extraction.
    CROP_BLACK_THRESHOLD = 15   # Pixel intensity threshold (0-255). Pixels with grayscale intensity
                                # below this value are considered part of a black border.
    CROP_MIN_CONTENT_DIMENSION = 300 # Minimum width or height (in pixels) of the detected content area
                                    # after potential cropping. If the content is smaller than this,
                                    # the crop is skipped for that image to prevent overly aggressive cropping.

    # --- NEW: Low Information Filter Configuration (Pre-Match) ---
    ENABLE_LOW_INFO_FILTER = True # Set to True to enable filtering of low variance images before matching
    LOW_INFO_VARIANCE_THRESHOLD = 100 # Grayscale pixel intensity variance threshold. Images below this are removed.
                                     # Pure black/white is 0. Tune this value based on your content. Start with 50-200.
    LOW_INFO_CHECK_HR_TOO = False   # If True, also check the HR image for low variance. If either LR or HR (if checked)
                                   # is below threshold, the pair is removed.

    # Frame Matching (Template Matching)
    match_threshold = 0.3   # Threshold for template matching (cv2.TM_SQDIFF_NORMED).
                            # Lower values mean a stricter (better) match. Range is typically 0.0 (perfect) to 1.0.
    resize_height = 540     # Internal height to which frames are resized *before* template matching for consistency.
                            # Adjust based on typical input video aspect ratios for optimal comparison. (e.g., 480, 540, 720)
    resize_width = int(resize_height * (4/3)) # Internal width for resizing before template matching.
                                              # Common aspect ratios: (16/9) or (4/3). Adjust as needed.
    distance_modifier = 0.75 # (Currently unused in process_image_pair, can be removed or repurposed)

    # Similarity Filtering (Perceptual Hash - Post-Match, Pre-Align)
    # After matching, this step removes image pairs where the LR images are too similar to each other
    # (e.g., consecutive frames with very little change).
    similarity_threshold = 4 # Perceptual hash (phash) difference threshold.
                             # Lower values mean images must be *more* similar to be considered duplicates and removed.
                             # A common starting point is 4-6. Higher values remove fewer images.

    # Image Alignment (Requires ImgAlign tool: https://github.com/NicholasGU/ImgAlign)
    # This step attempts to spatially align the matched LR and HR image pairs.
    img_align_scale = 2     # The integer scale factor between LR and HR images (e.g., 2 if HR is 2x LR resolution).
                            # ImgAlign uses this to help with the alignment process.
                            # If LR and HR are the same resolution after extraction/scaling for ImgAlign, set to 1.
                            # This depends on the resolutions of images *fed into ImgAlign*.

    # SISR Filtering (Post-Align)
    ENABLE_SISR_FILTERING = True # Set to True to run this stage using SuperResImageSelector
    SISR_MIN_CLIP_DISTANCE = 0.10  # Minimum CLIP distance for SISR selector (0.0-1.0, higher is more distinct)
    SISR_COMPLEXITY_THRESHOLD = 0.3 # Minimum complexity score for SISR selector (adjust based on results)
    SISR_MAX_BRIGHTNESS = 220     # Max average brightness (0-255) for SISR selector

    # --- Derived Paths & Progress File ---
    # Numbered prefixes help see the pipeline flow in the filesystem
    extracted_images_folder = os.path.join(output_base_folder, "1_EXTRACTED")
    # Low info filter modifies EXTRACTED in-place, so no new folder number.
    matched_base_folder = os.path.join(output_base_folder, "2_MATCHED")
    # pHash filter modifies MATCHED in-place.
    aligned_output_base_path = os.path.join(output_base_folder, "3_ALIGNED")
    sisr_filtered_base_folder = os.path.join(output_base_folder, "4_SISR_FILTERED")
    progress_file = os.path.join(output_base_folder, "progress.json")

    # --- Dynamic Output Paths (within main stages for clarity) ---
    output_lr_base_path = os.path.join(matched_base_folder, "LR") # For stage 2 output
    output_hr_base_path = os.path.join(matched_base_folder, "HR") # For stage 2 output

    # --- Create Base Output Directories ---
    os.makedirs(extracted_images_folder, exist_ok=True) # For LR and HR subfolders
    os.makedirs(matched_base_folder, exist_ok=True)   # For LR and HR subfolders
    # align_images creates its own base folder (aligned_output_base_path)
    if ENABLE_SISR_FILTERING:
        os.makedirs(sisr_filtered_base_folder, exist_ok=True) # For LR and HR subfolders

    print("--- Starting Image Pairing Pipeline ---")
    print(f"Progress file: {progress_file}")
    # Print key configurations
    print(f"Deinterlace Mode: {DEINTERLACE_MODE}")
    if DEINTERLACE_MODE != 'none': print(f"  Deinterlace Filter: {DEINTERLACE_FILTER}")
    print(f"Autocrop Black Borders: {CROP_BLACK_BORDERS}")
    if CROP_BLACK_BORDERS: print(f"  Crop Black Threshold: {CROP_BLACK_THRESHOLD}, Min Content Dimension: {CROP_MIN_CONTENT_DIMENSION}")
    print(f"Low Information Filter: {ENABLE_LOW_INFO_FILTER}")
    if ENABLE_LOW_INFO_FILTER: print(f"  Low Info Variance Threshold: {LOW_INFO_VARIANCE_THRESHOLD}, Check HR Too: {LOW_INFO_CHECK_HR_TOO}")
    print(f"SISR Filtering: {ENABLE_SISR_FILTERING}")
    if ENABLE_SISR_FILTERING and not SISR_AVAILABLE:
        print("  WARNING: SISR_AVAILABLE is False. SISR filtering stage will be skipped despite being enabled.")
    elif ENABLE_SISR_FILTERING:
        print(f"  SISR Config: Min CLIP Dist={SISR_MIN_CLIP_DISTANCE}, Complexity Thresh={SISR_COMPLEXITY_THRESHOLD}, Max Brightness={SISR_MAX_BRIGHTNESS}")


    # --- Pipeline Stages ---
    print("\n=== Stage 1: Preprocessing Videos (Frame Extraction) ===")
    lr_extracted_path, hr_extracted_path = preprocess_videos(
        lr_input_folder=lr_input_video_folder, hr_input_folder=hr_input_video_folder,
        extracted_images_folder=extracted_images_folder, begin_time=begin_time, end_time=end_time,
        scene_threshold=scene_threshold, lr_width=lr_width, lr_height=lr_height, lr_scale=lr_scale,
        hr_width=hr_width, hr_height=hr_height, hr_scale=hr_scale,
        deinterlace_mode=DEINTERLACE_MODE, deinterlace_filter=DEINTERLACE_FILTER,
        progress_file=progress_file
    )

    if CROP_BLACK_BORDERS:
        print("\n=== Stage 1.5: Autocropping Black Borders ===")
        if lr_extracted_path and os.path.isdir(lr_extracted_path):
            autocrop_frames_in_folders(lr_extracted_path, CROP_BLACK_THRESHOLD, CROP_MIN_CONTENT_DIMENSION, progress_file, "_LR")
        else: print(f"Skipping LR autocropping: Path '{lr_extracted_path}' not valid or empty.")
        if hr_extracted_path and os.path.isdir(hr_extracted_path):
            autocrop_frames_in_folders(hr_extracted_path, CROP_BLACK_THRESHOLD, CROP_MIN_CONTENT_DIMENSION, progress_file, "_HR")
        else: print(f"Skipping HR autocropping: Path '{hr_extracted_path}' not valid or empty.")
    else: print("\n=== Skipping Stage 1.5: Autocropping Black Borders (disabled) ===")

    # --- NEW STAGE ---
    if ENABLE_LOW_INFO_FILTER:
        print("\n=== Stage 1.7: Filtering Low Information Images (Pre-Match) ===")
        # This filter operates on the output of Stage 1 (and 1.5 if enabled), i.e., lr_extracted_path / hr_extracted_path
        filter_low_information_images(
            base_lr_path_str=lr_extracted_path,
            base_hr_path_str=hr_extracted_path,
            variance_threshold=LOW_INFO_VARIANCE_THRESHOLD,
            check_hr_too=LOW_INFO_CHECK_HR_TOO,
            progress_file=progress_file
        )
    else:
        print("\n=== Skipping Stage 1.7: Filtering Low Information Images (disabled) ===")


    print("\n=== Stage 2: Matching Frame Pairs (Template Matching) ===")
    # Ensure the output paths for matching are created before calling
    os.makedirs(output_lr_base_path, exist_ok=True)
    os.makedirs(output_hr_base_path, exist_ok=True)
    process_folder_pair(
        lr_base_path=lr_extracted_path, hr_base_path=hr_extracted_path, # Input from extraction (potentially filtered by low_info)
        output_lr_base_path=output_lr_base_path, output_hr_base_path=output_hr_base_path, # Output to MATCHED/LR and MATCHED/HR
        match_threshold=match_threshold, resize_height=resize_height, resize_width=resize_width,
        distance_modifier=distance_modifier, progress_file=progress_file
    )

    print("\n=== Stage 3: Filtering Similar Images (Perceptual Hash) ===")
    filter_similar_images(
        matched_lr_path=output_lr_base_path, matched_hr_path=output_hr_base_path, # Input from Stage 2
        similarity_threshold=similarity_threshold, progress_file=progress_file
    )

    print("\n=== Stage 4: Aligning Images (using ImgAlign) ===")
    align_images(
        output_lr_base_path=output_lr_base_path, output_hr_base_path=output_hr_base_path, # Input from Stage 3 (phash filtered)
        aligned_output_base_path=aligned_output_base_path, # Output to ALIGNED base
        img_align_scale=img_align_scale,
        progress_file=progress_file
    )

    if ENABLE_SISR_FILTERING and SISR_AVAILABLE:
        print("\n=== Stage 5: Filtering Aligned Images with SISR Selector ===")
        sisr_input_lr_aligned = os.path.join(aligned_output_base_path, "LR") # Input LR from Stage 4
        sisr_input_hr_aligned = os.path.join(aligned_output_base_path, "HR") # Input HR from Stage 4
        
        # Ensure the base output folder for SISR exists before defining LR/HR subpaths
        os.makedirs(sisr_filtered_base_folder, exist_ok=True)
        sisr_final_output_lr = os.path.join(sisr_filtered_base_folder, "LR")
        sisr_final_output_hr = os.path.join(sisr_filtered_base_folder, "HR")
        os.makedirs(sisr_final_output_lr, exist_ok=True) # Ensure specific LR/HR output dirs exist
        os.makedirs(sisr_final_output_hr, exist_ok=True)


        filter_aligned_with_sisr(
            aligned_lr_input_folder_str=sisr_input_lr_aligned,
            aligned_hr_input_folder_str=sisr_input_hr_aligned,
            final_sisr_output_lr_folder_str=sisr_final_output_lr,
            final_sisr_output_hr_folder_str=sisr_final_output_hr,
            sisr_min_clip_distance=SISR_MIN_CLIP_DISTANCE,
            sisr_complexity_threshold=SISR_COMPLEXITY_THRESHOLD,
            sisr_max_brightness=SISR_MAX_BRIGHTNESS,
            progress_file=progress_file
        )
    elif ENABLE_SISR_FILTERING and not SISR_AVAILABLE:
        print("\n=== Skipping Stage 5: SISR Filtering (SuperResImageSelector not available) ===")
    else:
        print("\n=== Skipping Stage 5: SISR Filtering (disabled by configuration) ===")


    print("\n--- Image Processing and Alignment Pipeline Completed ---")

# --- END OF MODIFIED FILE ---