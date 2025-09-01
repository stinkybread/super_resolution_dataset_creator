# srdc_pipeline.py - Complete SRDC Processing Pipeline

import cv2
import os
import shutil
import subprocess
from tqdm import tqdm
import glob
import concurrent.futures
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict
import time
import threading
import queue
import sys
import warnings

# Attempt to import ImageHash related libraries
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    print("Warning: imagehash or Pillow not found. pHash similarity filtering will be unavailable.")
    IMAGEHASH_AVAILABLE = False

# Configuration Import
try:
    import config
except ImportError:
    print("FATAL ERROR: config.py not found. Please ensure it exists in the same directory.")
    exit(1)

# Global variables for FFmpeg paths
FFMPEG_PATH = config.FFMPEG_PATH
FFPROBE_PATH = config.FFPROBE_PATH

def _find_executable(name, configured_path):
    """Find executable in configured path or system PATH."""
    if configured_path and Path(configured_path).is_file():
        print(f"Using configured {name}: {configured_path}")
        return str(configured_path)

    found_path = shutil.which(name)
    if found_path:
        print(f"Found {name} in PATH: {found_path}")
        return found_path

    print(f"ERROR: {name} not found. Set path in config.py or install.")
    return None

# OpenCV and CUDA detection
print(f"OpenCV version: {cv2.__version__}")
CUDA_AVAILABLE_OPENCV = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"OpenCV CUDA available: {CUDA_AVAILABLE_OPENCV}")
use_gpu_opencv = CUDA_AVAILABLE_OPENCV

if use_gpu_opencv:
    print("OpenCV: CUDA GPU detected. Using GPU for OpenCV tasks where possible.")
    try:
        print(f"OpenCV: CUDA device: {cv2.cuda.getDeviceName(cv2.cuda.getDevice())}")
    except cv2.error as e:
        print(f"OpenCV: CUDA device details error: {e}")
        use_gpu_opencv = False
else:
    print("OpenCV: No CUDA GPU for OpenCV. Using CPU.")

def update_progress(progress_file_path: Path, stage: str, item: str = None, completed: bool = False, item_key: str = 'items'):
    """Update progress tracking file."""
    progress_data = {}
    if progress_file_path.exists():
        try:
            with progress_file_path.open('r') as f:
                progress_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Progress file {progress_file_path} corrupt. Starting fresh.")

    progress_data.setdefault(stage, {item_key: [], 'completed': False})
    if not isinstance(progress_data[stage].get(item_key), list):
        progress_data[stage][item_key] = []

    if item and item not in progress_data[stage][item_key]:
        progress_data[stage][item_key].append(item)

    if completed:
        progress_data[stage]['completed'] = True
    elif item:
        progress_data[stage]['completed'] = False

    try:
        with progress_file_path.open('w') as f:
            json.dump(progress_data, f, indent=4)
    except IOError as e:
        print(f"Error writing progress file {progress_file_path}: {e}")

def get_progress(progress_file_path: Path) -> dict:
    """Get current progress from file."""
    if progress_file_path.exists():
        try:
            with progress_file_path.open('r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Progress file {progress_file_path} corrupt. Returning empty.")
            return {}
    return {}

def get_video_duration(video_path: Path) -> float | None:
    """Get video duration using ffprobe."""
    global FFPROBE_PATH
    if not FFPROBE_PATH:
        print("Error: ffprobe path not set for duration check.")
        return None

    if not video_path.exists():
        print(f"Error: Video file not found for duration: {video_path}")
        return None

    cmd = [
        FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print(f"ffprobe timeout getting duration for {video_path.name}.")
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"ffprobe error getting duration for {video_path.name}: {e}")
    return None

def is_hdr_video(video_path: Path) -> bool:
    """Enhanced HDR detection using ffprobe with JSON output."""
    global FFPROBE_PATH
    if not FFPROBE_PATH:
        return False
        
    cmd = [
        FFPROBE_PATH, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=color_transfer,color_primaries,side_data_list",
        "-of", "json",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        data = json.loads(result.stdout)
        stream = data.get('streams', [{}])[0]
        
        # Check for HDR indicators
        if stream.get('color_transfer') in ['smpte2084', 'arib-std-b67']:
            return True
        if stream.get('color_primaries') == 'bt2020':
            return True
        if any(sd.get('side_data_type') == 'Dolby Vision' 
               for sd in stream.get('side_data_list', [])):
            return True
            
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError):
        return False
    return False

def detect_interlacing(video_path: Path, probe_duration: int = 10) -> bool:
    """Detect if video is interlaced using ffmpeg idet filter."""
    global FFMPEG_PATH
    print(f"Probing interlacing for {video_path.name}...")

    cmd = [
        FFMPEG_PATH, "-hide_banner", "-filter:v", "idet", "-an", "-sn", "-dn",
        "-t", str(probe_duration), "-map", "0:v:0?", "-f", "null", "-threads", "1",
        "-nostats", "-i", str(video_path)
    ]

    try:
        process = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                               text=True, check=False, timeout=60)
        stderr = process.stderr

        tff = re.search(r"TFF:\s*(\d+)", stderr)
        bff = re.search(r"BFF:\s*(\d+)", stderr)
        prog = re.search(r"Progressive:\s*(\d+)", stderr)

        if tff and bff and prog:
            t, b, p = int(tff.group(1)), int(bff.group(1)), int(prog.group(1))
            total = t + b + p
            print(f"  Idet: TFF={t}, BFF={b}, Prog={p}")

            if total == 0:
                print("  Warning: idet 0 frames. Assuming progressive.")
                return False

            interlaced_ratio = (t + b) / total
            is_interlaced = interlaced_ratio > 0.25
            print(f"  Detected as {'INTERLACED' if is_interlaced else 'PROGRESSIVE'} (ratio: {interlaced_ratio:.2f})")
            return is_interlaced

        print(f"  Warning: Could not parse idet summary for {video_path.name}. Assuming progressive.")
        return False
    except Exception as e:
        print(f"Error during interlacing detection for {video_path.name}: {e}")
        return False

def extract_frames_ffmpeg(video_path: Path, output_folder: Path, begin_time: str, end_time: str,
                         scene_threshold: float, video_type: str, deinterlace_mode: str,
                         deinterlace_filter: str):
    """Extract frames using FFmpeg with optional deinterlacing, chroma upsampling, and HDR tone mapping."""
    global FFMPEG_PATH
    output_folder.mkdir(parents=True, exist_ok=True)

    apply_deinterlace = False
    if deinterlace_mode == 'auto':
        apply_deinterlace = detect_interlacing(video_path)
    elif deinterlace_mode == 'both':
        apply_deinterlace = True
    elif deinterlace_mode == 'lr' and video_type == 'lr':
        apply_deinterlace = True
    elif deinterlace_mode == 'hr' and video_type == 'hr':
        apply_deinterlace = True

    # Determine if HDR tone mapping should be applied
    apply_hdr_tone_mapping = False
    if config.ENABLE_HDR_TONE_MAPPING:
        hdr_mode = getattr(config, 'HDR_TONE_MAPPING_MODE', 'auto')
        if hdr_mode == 'both':
            apply_hdr_tone_mapping = True
        elif hdr_mode == 'hr_only' and video_type == 'hr':
            apply_hdr_tone_mapping = True
        elif hdr_mode == 'lr_only' and video_type == 'lr':
            apply_hdr_tone_mapping = True
        elif hdr_mode == 'auto':
            # Enhanced HDR detection
            apply_hdr_tone_mapping = is_hdr_video(video_path)
            print(f"  HDR auto-detected: {apply_hdr_tone_mapping} for {video_path.name}")

    cmd = [FFMPEG_PATH, "-i", str(video_path)]

    if begin_time != "00:00:00" or end_time != "00:00:00":
        cmd.extend(["-ss", begin_time])
        if end_time != "00:00:00":
            cmd.extend(["-to", end_time])

    vf = []
    if apply_deinterlace:
        vf.append(deinterlace_filter)
        print(f"  Applying deinterlace '{deinterlace_filter}' to {video_path.name}")
    if config.ENABLE_CHROMA_UPSAMPLING:
        vf.append(config.CHROMA_UPSAMPLING_FILTER)
        print(f"  Applying high-quality chroma upsampling to {video_path.name}")
    if apply_hdr_tone_mapping:
        vf.append(config.HDR_TONE_MAPPING_FILTER)
        print(f"  Applying HDR tone mapping to {video_type.upper()} video: {video_path.name}")

    vf.append(f"select='gt(scene,{scene_threshold})',showinfo")

    cmd.extend([
        "-vf", ",".join(vf), "-vsync", "vfr", "-q:v", "2",
        str(output_folder / "frame_%06d.png"),
        "-hide_banner", "-loglevel", "info", "-nostats"
    ])
    print(f"\nFFmpeg for {video_path.name}:\n  {' '.join(cmd)}")
    timestamps, frame_files, duration = [], [], get_video_duration(video_path)

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        _, stderr = process.communicate(timeout=7200)

        if process.returncode != 0:
            print(f"Error extracting from {video_path.name} (code {process.returncode}):\n{stderr[-1000:]}")
            return False, [], [], duration

        log_message = f"Frames extracted from {video_path.name}"
        if apply_deinterlace: log_message += " (Deinterlaced)"
        if config.ENABLE_CHROMA_UPSAMPLING: log_message += " (Chroma Upsampled)"
        if apply_hdr_tone_mapping: log_message += f" (HDR Tonemapped - {video_type.upper()})"
        print(log_message)

        pts_re = re.compile(r'\[Parsed_showinfo.*pts_time:(\d+\.?\d*)')
        timestamps = [float(m.group(1)) for m in pts_re.finditer(stderr)]
        frame_files = sorted([f.name for f in output_folder.glob("frame_*.png")])

        n_frames, n_ts = len(frame_files), len(timestamps)
        if n_frames == 0:
            print(f"Warning: No frames saved for {video_path.name}.")
            return False, [], [], duration

        if n_ts > n_frames:
            timestamps = timestamps[:n_frames]
        elif n_ts < n_frames:
            print(f"Warning: Timestamps ({n_ts}) < Frames ({n_frames}) for {video_path.name}. Padding timestamps.")
            if timestamps:
                timestamps.extend([timestamps[-1]] * (n_frames - n_ts))
            else:
                print(f"ERROR: No timestamps parsed for {video_path.name} but frames exist.")
                return False, [], [], duration
        return True, frame_files, timestamps, duration
    except Exception as e:
        print(f"Error extracting {video_path.name}: {e}")
        return False, [], [], duration

def preprocess_videos(lr_input_folder: Path, hr_input_folder: Path, extracted_images_folder: Path, progress_file: Path):
    """Extract and timestamp frames from LR and HR videos, matching by filename stem."""
    prog = get_progress(progress_file)
    stage = 'preprocess'
    if prog.get(stage, {}).get('completed', False):
        print("Preprocessing already completed. Skipping...")
        return extracted_images_folder / "LR", extracted_images_folder / "HR"

    lr_base = extracted_images_folder / "LR"
    hr_base = extracted_images_folder / "HR"
    lr_base.mkdir(parents=True, exist_ok=True)
    hr_base.mkdir(parents=True, exist_ok=True)

    exts = ('.avi', '.mp4', '.mkv', '.ts', '.mpg', '.mpeg', '.mov', '.flv', '.wmv', '.webm')
    lr_files = [p for p in lr_input_folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    hr_files = [p for p in hr_input_folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    lr_videos = {p.stem.lower(): p for p in lr_files}
    hr_videos = {p.stem.lower(): p for p in hr_files}

    if len(lr_files) != len(lr_videos):
        print(f"Warning: Found {len(lr_files)} LR videos but only {len(lr_videos)} unique filenames (stems). Some videos might be ignored.")
    if len(hr_files) != len(hr_videos):
        print(f"Warning: Found {len(hr_files)} HR videos but only {len(hr_videos)} unique filenames (stems). Some videos might be ignored.")

    common_stems = sorted(list(set(lr_videos.keys()) & set(hr_videos.keys())))
    videos_map = {lr_videos[stem]: hr_videos[stem] for stem in common_stems}
    print(f"Found {len(videos_map)} video pairs based on matching filenames.")

    if len(videos_map) != len(lr_videos) or len(videos_map) != len(hr_videos):
        print(f"Warning: Mismatch in video counts. LR: {len(lr_videos)}, HR: {len(hr_videos)}, Common: {len(videos_map)}. Some videos may not have a matching pair.")

    if not videos_map:
        print("Error: No matching video pairs found. Please check your input folders.")
        update_progress(progress_file, stage, completed=True)
        return lr_base, hr_base

    processed_items = prog.get(stage, {}).get('items', [])
    for lr_path, hr_path in tqdm(videos_map.items(), desc="Preprocessing videos"):
        if lr_path.name in processed_items:
            print(f"Skipping already processed video: {lr_path.name}")
            continue

        name_stem = lr_path.stem
        print(f"\n--- Processing {lr_path.name} (LR) & {hr_path.name} (HR) ---")
        lr_out = lr_base / name_stem
        print(f"Extracting LR for {lr_path.name}...")
        lr_ok, lr_f, lr_ts, lr_dur = extract_frames_ffmpeg(lr_path, lr_out, config.BEGIN_TIME, config.END_TIME, config.SCENE_THRESHOLD, 'lr', config.DEINTERLACE_MODE, config.DEINTERLACE_FILTER)

        if lr_ok and lr_f and lr_ts:
            with (lr_out / config.METADATA_FILENAME).open('w') as f:
                json.dump({"duration": lr_dur, "timestamps": dict(zip(lr_f, lr_ts))}, f, indent=4)
        elif not lr_ok:
            print(f"Skipping HR for {name_stem} due to LR failure.")
            update_progress(progress_file, stage, item=lr_path.name)
            continue

        hr_out = hr_base / name_stem
        print(f"Extracting HR for {hr_path.name}...")
        hr_ok, hr_f, hr_ts, hr_dur = extract_frames_ffmpeg(hr_path, hr_out, config.BEGIN_TIME, config.END_TIME, config.SCENE_THRESHOLD, 'hr', config.DEINTERLACE_MODE, config.DEINTERLACE_FILTER)

        if hr_ok and hr_f and hr_ts:
            with (hr_out / config.METADATA_FILENAME).open('w') as f:
                json.dump({"duration": hr_dur, "timestamps": dict(zip(hr_f, hr_ts))}, f, indent=4)
        update_progress(progress_file, stage, item=lr_path.name)

    update_progress(progress_file, stage, completed=True)
    print("--- Preprocessing Finished ---")
    return lr_base, hr_base

def _find_template_in_container(container_gray, content_gray):
    """
    Helper function to find a smaller 'content' image within a larger 'container' image.
    Returns the match score and the top-left location of the match.
    Optimized for performance by downscaling very large container images.
    """
    h_cont, w_cont = container_gray.shape
    h_contn, w_contn = content_gray.shape

    if w_cont < w_contn or h_cont < h_contn:
        return 0.0, None

    # If the container is large (e.g., >1920px wide), scale it down for faster matching.
    max_width = 1920.0
    if w_cont > max_width:
        scale = max_width / w_cont
        container_scaled = cv2.resize(container_gray, (int(w_cont * scale), int(h_cont * scale)), cv2.INTER_AREA)
        content_scaled = cv2.resize(content_gray, (int(w_contn * scale), int(h_contn * scale)), cv2.INTER_AREA)
    else:
        scale = 1.0
        container_scaled = container_gray
        content_scaled = content_gray

    result = cv2.matchTemplate(container_scaled, content_scaled, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc_scaled = cv2.minMaxLoc(result)

    # Convert location back to original container coordinates if we scaled down.
    original_loc = (int(max_loc_scaled[0] / scale), int(max_loc_scaled[1] / scale)) if scale < 1.0 else max_loc_scaled
    return max_val, original_loc

def fix_mismatched_content_pair(lr_path: Path, hr_path: Path):
    """
    Compares an LR/HR pair and crops the 'container' image to match the 'content'.
    Handles both pan-and-scan (e.g., 4:3 content in a 16:9 frame) and open-matte
    (e.g., 16:9 content in a 4:3 frame) by running a "contest".
    """
    try:
        lr_img = cv2.imread(str(lr_path))
        hr_img = cv2.imread(str(hr_path))
        if lr_img is None or hr_img is None: return False

        h_lr, w_lr = lr_img.shape[:2]
        h_hr, w_hr = hr_img.shape[:2]

        # If aspect ratios are very similar, no fix is likely needed.
        if abs((w_lr / h_lr) - (w_hr / h_hr)) < 0.05:
            return False

        lr_gray = cv2.cvtColor(lr_img, cv2.COLOR_BGR2GRAY)
        hr_gray = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)

        # Contest 1: Find LR (content) inside HR (container).
        score1, loc1 = _find_template_in_container(hr_gray, lr_gray)

        # Contest 2: Find HR (content) inside LR (container).
        score2, loc2 = _find_template_in_container(lr_gray, hr_gray)

        # --- Decide the winner and perform the crop ---
        if score1 > score2 and score1 > config.CONTENT_FIX_THRESHOLD:
            # Winner: LR is content, HR is container. Crop HR.
            top_left = loc1
            bottom_right = (top_left[0] + w_lr, top_left[1] + h_lr)
            cropped_img = hr_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            if cropped_img.size > 0:
                cv2.imwrite(str(hr_path), cropped_img)
                return True
        elif score2 > score1 and score2 > config.CONTENT_FIX_THRESHOLD:
            # Winner: HR is content, LR is container. Crop LR.
            top_left = loc2
            bottom_right = (top_left[0] + w_hr, top_left[1] + h_hr)
            cropped_img = lr_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            if cropped_img.size > 0:
                cv2.imwrite(str(lr_path), cropped_img)
                return True

        return False
    except Exception:
        # Silently fail on any error to not stop the whole batch process.
        return False

def fix_mismatched_content_in_folders(lr_base: Path, hr_base: Path, prog_file: Path):
    """Iterates through all extracted frames and applies the universal content fix."""
    stage = 'content_fix'
    prog_data = get_progress(prog_file)
    if prog_data.get(stage, {}).get('completed', False):
        print("Mismatched content fix already completed. Skipping.")
        return

    print(f"\n--- Attempting to Fix Mismatched Content (Threshold: {config.CONTENT_FIX_THRESHOLD}) ---")
    vid_folders = [f for f in lr_base.iterdir() if f.is_dir()]
    processed_folders = prog_data.get(stage, {}).get('items', [])
    total_fixed = 0

    for vid_folder in tqdm(vid_folders, desc="Fixing Content Mismatches"):
        if vid_folder.name in processed_folders:
            continue

        hr_vid_folder = hr_base / vid_folder.name
        if not hr_vid_folder.is_dir(): continue

        image_pairs = [(p, hr_vid_folder / p.name) for p in vid_folder.glob('*.png') if (hr_vid_folder / p.name).exists()]
        if not image_pairs:
            update_progress(prog_file, stage, item=vid_folder.name)
            continue

        fixed_count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fix_mismatched_content_pair, lr, hr): lr for lr, hr in image_pairs}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Analyzing {vid_folder.name}", leave=False):
                try:
                    if fut.result():
                        fixed_count += 1
                except Exception:
                    pass

        if fixed_count > 0:
            print(f"  Applied content-crop fix to {fixed_count} pairs in {vid_folder.name}.")

        total_fixed += fixed_count
        update_progress(prog_file, stage, item=vid_folder.name)

    print(f"--- Mismatched Content Fix Finished. Total pairs fixed: {total_fixed} ---")
    update_progress(prog_file, stage, completed=True)

def autocrop_single_image(img_path: Path, out_path: Path, crop_black: bool, black_thresh: int, crop_white: bool, white_thresh: int, min_dim: int):
    """Autocrop black and/or white borders from a single image."""
    img = cv2.imread(str(img_path))
    if img is None: return False
    h_orig, w_orig = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if crop_black:
        _, black_mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY)
    else:
        black_mask = np.full(gray.shape, 255, dtype=np.uint8)

    if crop_white:
        _, white_mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY_INV)
        white_mask = cv2.bitwise_not(white_mask)
    else:
        white_mask = np.full(gray.shape, 255, dtype=np.uint8)

    final_content_mask = cv2.bitwise_and(black_mask, white_mask)
    cntrs, _ = cv2.findContours(final_content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cntrs: return False
    x_min, y_min, x_max, y_max = w_orig, h_orig, 0, 0
    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    if (x_max - x_min) < min_dim or (y_max - y_min) < min_dim: return False
    if x_min == 0 and y_min == 0 and x_max == w_orig and y_max == h_orig: return False

    try:
        cv2.imwrite(str(out_path), img[y_min:y_max, x_min:x_max])
        return True
    except Exception:
        return False

def autocrop_frames_in_folders(base_path: Path, prog_file: Path, suffix: str = ""):
    """Autocrop black and/or white borders from frames in all video folders."""
    stage = f'autocrop{suffix}'
    prog_data = get_progress(prog_file)
    if prog_data.get(stage, {}).get('completed', False):
        print(f"Autocrop {base_path.name}{suffix} done. Skip.")
        return

    print(f"\n--- Autocropping {base_path.name}{suffix} (Black: {config.CROP_BLACK_BORDERS}, White: {config.CROP_WHITE_BORDERS}) ---")
    vid_folders = [f for f in base_path.iterdir() if f.is_dir()]
    if not vid_folders:
        print(f"No subfolders in {base_path}.")
        update_progress(prog_file, stage, completed=True)
        return

    processed = prog_data.get(stage, {}).get('items', [])
    total_proc, total_crop = 0, 0

    for vid_folder in tqdm(vid_folders, desc=f"Autocropping in {base_path.name}"):
        if vid_folder.name in processed: continue
        imgs = list(vid_folder.glob('*.png'))
        if not imgs:
            update_progress(prog_file, stage, item=vid_folder.name)
            continue

        total_proc += len(imgs)
        cropped_this = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(autocrop_single_image, p, p, config.CROP_BLACK_BORDERS, config.CROP_BLACK_THRESHOLD, config.CROP_WHITE_BORDERS, config.CROP_WHITE_THRESHOLD, config.CROP_MIN_CONTENT_DIMENSION): p for p in imgs}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Cropping {vid_folder.name}", leave=False):
                try:
                    if fut.result(): cropped_this += 1
                except Exception as e:
                    print(f"Autocrop error for {futures[fut].name}: {e}")
        
        if cropped_this > 0: print(f"  Cropped {cropped_this}/{len(imgs)} in {vid_folder.name}.")
        total_crop += cropped_this
        update_progress(prog_file, stage, item=vid_folder.name)

    print(f"--- Autocrop {base_path.name}{suffix} Finished. Processed: {total_proc}, Cropped: {total_crop} ---")
    update_progress(prog_file, stage, completed=True)

def check_image_variance(img_path: Path, thresh: float):
    """Check if image has low information content based on variance."""
    try:
        img = cv2.imread(str(img_path))
        return np.var(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < thresh if img is not None else False
    except Exception:
        return False

def filter_low_information_images(lr_base: Path, hr_base: Path, prog_file: Path):
    """Filter out low information images based on variance threshold."""
    stage = 'low_info_filter'
    prog_data = get_progress(prog_file)
    if prog_data.get(stage, {}).get('completed', False):
        print("Low info filter done. Skip.")
        return

    print(f"\n--- Low Info Filter (Thresh: {config.LOW_INFO_VARIANCE_THRESHOLD}, HR Check: {config.LOW_INFO_CHECK_HR_TOO}) ---")
    vid_folders = [f for f in lr_base.iterdir() if f.is_dir()]
    if not vid_folders:
        update_progress(prog_file, stage, completed=True)
        return

    processed = prog_data.get(stage, {}).get('items', [])
    tot_lr_rem, tot_hr_rem = 0, 0
    for lr_vid_f in tqdm(vid_folders, desc="Filtering low info"):
        if lr_vid_f.name in processed: continue
        hr_vid_f = hr_base / lr_vid_f.name
        if not hr_vid_f.is_dir():
            update_progress(prog_file, stage, item=lr_vid_f.name)
            continue

        lr_rem, hr_rem = 0, 0
        for lr_p in tqdm(list(lr_vid_f.glob('*.png')), desc=f"Checking {lr_vid_f.name}", leave=False):
            rm = check_image_variance(lr_p, config.LOW_INFO_VARIANCE_THRESHOLD)
            if not rm and config.LOW_INFO_CHECK_HR_TOO and (hr_p := hr_vid_f / lr_p.name).exists():
                rm = check_image_variance(hr_p, config.LOW_INFO_VARIANCE_THRESHOLD)
            
            if rm:
                try: lr_p.unlink(); lr_rem += 1
                except OSError as e: print(f"Err removing LR {lr_p.name}: {e}")
                if (hr_c := hr_vid_f / lr_p.name).exists():
                    try: hr_c.unlink(); hr_rem += 1
                    except OSError as e: print(f"Err removing HR {hr_c.name}: {e}")
        
        if lr_rem > 0: print(f"  Removed {lr_rem} LR (and {hr_rem} HR) from {lr_vid_f.name}.")
        tot_lr_rem, tot_hr_rem = tot_lr_rem + lr_rem, tot_hr_rem + hr_rem
        update_progress(prog_file, stage, item=lr_vid_f.name)

    print(f"--- Low Info Filter Finished. LR Rem: {tot_lr_rem}, HR Rem: {tot_hr_rem} ---")
    update_progress(prog_file, stage, completed=True)

def process_image_pair(lr_img_name: str, lr_folder_path: Path, lr_time: float, hr_candidate_img_names: list[str], hr_folder_path: Path, hr_timestamps_data: dict):
    """Process a single LR image against HR candidates for matching."""
    try:
        lr_frame = cv2.imread(str(lr_folder_path / lr_img_name))
        assert lr_frame is not None
        lr_gray_resized = cv2.cvtColor(cv2.resize(lr_frame, (config.MATCH_RESIZE_WIDTH, config.MATCH_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    except (cv2.error, AssertionError):
        return None

    best_hr_name, best_score, matched_hr_ts = None, float('-inf'), None
    for hr_name in hr_candidate_img_names:
        try:
            hr_frame = cv2.imread(str(hr_folder_path / hr_name))
            assert hr_frame is not None
            hr_gray_resized = cv2.cvtColor(cv2.resize(hr_frame, (config.MATCH_RESIZE_WIDTH, config.MATCH_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            _, cur_score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(hr_gray_resized, lr_gray_resized, cv2.TM_CCOEFF_NORMED))
            if cur_score > best_score:
                best_score, best_hr_name, matched_hr_ts = cur_score, hr_name, hr_timestamps_data.get(hr_name)
        except (cv2.error, AssertionError):
            continue
    
    return (best_hr_name, best_score, lr_time, matched_hr_ts) if best_hr_name and best_score >= config.MATCH_THRESHOLD and matched_hr_ts is not None else None

def filter_temporal_inconsistency(matches: list, video_name: str) -> list:
    """Filter matches to maintain temporal consistency."""
    if not matches: return []
    matches.sort(key=lambda x: x[2])
    filtered, last_hr_ts = [], -1.0
    for lr_f, hr_f, lr_ts, hr_ts in matches:
        if hr_ts >= last_hr_ts:
            filtered.append((lr_f, hr_f, lr_ts, hr_ts))
            last_hr_ts = hr_ts
    if removed := len(matches) - len(filtered): print(f"  Temporal filter ({video_name}): Removed {removed} inconsistent HR matches.")
    return filtered

def process_folder_pair(lr_extracted_base: Path, hr_extracted_base: Path, out_lr_matched: Path, out_hr_matched: Path, prog_file: Path):
    """Process LR/HR folder pairs for frame matching."""
    stage = 'match'
    prog_data = get_progress(prog_file)
    if prog_data.get(stage, {}).get('completed', False):
        print("Matching done. Skip.")
        return

    common_names = sorted(list({f.name for f in lr_extracted_base.iterdir() if f.is_dir()} & {f.name for f in hr_extracted_base.iterdir() if f.is_dir()}))
    if not common_names:
        print("Error: No common video folders for matching.")
        update_progress(prog_file, stage, completed=True)
        return
    print(f"Found {len(common_names)} common video folders for matching.")
    processed = prog_data.get(stage, {}).get('items', [])

    for vid_name in common_names:
        if vid_name in processed: continue
        lr_fld, hr_fld = lr_extracted_base / vid_name, hr_extracted_base / vid_name
        if not (lr_meta_f := lr_fld / config.METADATA_FILENAME).exists() or not (hr_meta_f := hr_fld / config.METADATA_FILENAME).exists():
            print(f"Metadata missing for {vid_name}. Skip.")
            update_progress(prog_file, stage, item=vid_name)
            continue
        try:
            with lr_meta_f.open('r') as f: lr_meta = json.load(f)
            with hr_meta_f.open('r') as f: hr_meta = json.load(f)
            lr_ts_data, lr_dur = lr_meta["timestamps"], lr_meta.get("duration")
            hr_ts_data = hr_meta["timestamps"]
            if not lr_dur or lr_dur <= 0:
                lr_dur = config.FALLBACK_MATCH_CANDIDATE_WINDOW_SECONDS / config.INITIAL_MATCH_CANDIDATE_WINDOW_PERCENTAGE if config.INITIAL_MATCH_CANDIDATE_WINDOW_PERCENTAGE > 0 else 1800
        except Exception as e:
            print(f"Error loading metadata for {vid_name}: {e}. Skip.")
            update_progress(prog_file, stage, item=vid_name)
            continue

        lr_imgs = sorted([name for name in lr_ts_data if (lr_fld / name).exists()])
        hr_imgs = sorted([name for name in hr_ts_data if (hr_fld / name).exists()])
        if not lr_imgs or not hr_imgs:
            print(f"No valid images/timestamps for {vid_name}. Skip.")
            update_progress(prog_file, stage, item=vid_name)
            continue

        raw_matches, last_lr_ts, last_hr_ts, first_match_done = [], None, None, False
        initial_win_secs = lr_dur * config.INITIAL_MATCH_CANDIDATE_WINDOW_PERCENTAGE
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for lr_name in lr_imgs:
                lr_time = lr_ts_data.get(lr_name)
                if lr_time is None: continue
                
                if not first_match_done:
                    min_hr_t, max_hr_t = lr_time - initial_win_secs, lr_time + initial_win_secs
                elif last_lr_ts is not None and last_hr_ts is not None:
                    expected_hr_t = last_hr_ts + (lr_time - last_lr_ts)
                    min_hr_t, max_hr_t = expected_hr_t - config.SUBSEQUENT_MATCH_CANDIDATE_WINDOW_SECONDS, expected_hr_t + config.SUBSEQUENT_MATCH_CANDIDATE_WINDOW_SECONDS
                else:
                    min_hr_t, max_hr_t = lr_time - config.FALLBACK_MATCH_CANDIDATE_WINDOW_SECONDS, lr_time + config.FALLBACK_MATCH_CANDIDATE_WINDOW_SECONDS
                
                hr_candidates = [hr_n for hr_n in hr_imgs if min_hr_t <= hr_ts_data.get(hr_n, float('-inf')) <= max_hr_t]
                if hr_candidates:
                    futures[executor.submit(process_image_pair, lr_name, lr_fld, lr_time, hr_candidates, hr_fld, hr_ts_data)] = (lr_name, lr_time)

            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Matching {vid_name}"):
                lr_name_done, lr_time_done = futures[fut]
                try:
                    if res := fut.result():
                        raw_matches.append((lr_name_done, res[0], res[2], res[3]))
                        if not first_match_done or res[2] > last_lr_ts:
                            last_lr_ts, last_hr_ts, first_match_done = res[2], res[3], True
                except Exception as e:
                    print(f"Match worker error for {lr_name_done} ({vid_name}): {e}")

        consistent_matches = filter_temporal_inconsistency(raw_matches, vid_name)
        copied = 0
        for lr_f, hr_f, _, _ in consistent_matches:
            try:
                stem = Path(lr_f).stem
                shutil.copy2(lr_fld / lr_f, out_lr_matched / f"{vid_name}_{stem}.png")
                shutil.copy2(hr_fld / hr_f, out_hr_matched / f"{vid_name}_{stem}.png")
                copied += 1
            except Exception as e:
                print(f"Error copying {lr_f}/{hr_f} for {vid_name}: {e}")
        
        print(f"Consistent pairs for {vid_name}: {len(consistent_matches)}, Copied: {copied}")
        update_progress(prog_file, stage, item=vid_name)
    
    update_progress(prog_file, stage, completed=True)
    print("--- Matching Finished ---")

def filter_similar_images_phash(lr_matched: Path, hr_matched: Path, prog_file: Path):
    """Filter similar images using perceptual hashing."""
    if not IMAGEHASH_AVAILABLE or config.PHASH_SIMILARITY_THRESHOLD < 0:
        print("Skipping pHash Filter (disabled or unavailable).")
        return
    
    stage = 'filter_phash'
    prog_data = get_progress(prog_file)
    if prog_data.get(stage, {}).get('completed', False):
        print("pHash filter done. Skip.")
        return

    print(f"\n--- pHash Similarity Filter (Threshold: {config.PHASH_SIMILARITY_THRESHOLD}) ---")
    try:
        lr_imgs_all = sorted([f for f in lr_matched.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    except Exception as e:
        print(f"Error listing for pHash: {e}")
        return
    if not lr_imgs_all:
        update_progress(prog_file, stage, completed=True)
        return

    groups = defaultdict(list)
    re_fname = re.compile(r"^(.*?)_frame_\d+\.png$")
    for p in lr_imgs_all:
        groups[m.group(1) if (m := re_fname.match(p.name)) else 'unknown_video_group'].append(p)
    
    tot_rem = 0
    processed = prog_data.get(stage, {}).get('items', [])
    for vid_name, paths in tqdm(groups.items(), desc="pHash by video"):
        if vid_name == 'unknown_video_group' or vid_name in processed: continue
        
        kept_hashes, to_rem_stems = {}, set()
        for lr_p in tqdm(sorted(paths), desc=f"Hashing {vid_name}", leave=False):
            if not (hr_p := hr_matched / lr_p.name).exists(): continue
            try:
                with Image.open(lr_p) as pil_img: cur_hash = imagehash.phash(pil_img)
                size = lr_p.stat().st_size
            except Exception: continue
            
            is_sim = False
            for khash, (kpath, ksize) in list(kept_hashes.items()):
                if (cur_hash - khash) < config.PHASH_SIMILARITY_THRESHOLD:
                    if size >= ksize:
                        to_rem_stems.add(kpath.stem)
                        del kept_hashes[khash]
                        kept_hashes[cur_hash] = (lr_p, size)
                    else:
                        to_rem_stems.add(lr_p.stem)
                    is_sim = True
                    break
            if not is_sim: kept_hashes[cur_hash] = (lr_p, size)
        
        rem_grp = 0
        for stem_rem in to_rem_stems:
            try:
                if (f_lr := lr_matched / f"{stem_rem}.png").exists(): f_lr.unlink()
                if (f_hr := hr_matched / f"{stem_rem}.png").exists(): f_hr.unlink()
                rem_grp += 1
            except OSError: pass
        
        if rem_grp > 0: print(f"pHash ({vid_name}): Removed {rem_grp} pairs.")
        tot_rem += rem_grp
        update_progress(prog_file, stage, item=vid_name)
    
    print(f"pHash filter done. Total removed: {tot_rem}.")
    update_progress(prog_file, stage, completed=True)

def read_output_stream(pipe, output_queue, stop_event):
    """Thread function to continuously read output from pipe."""
    try:
        while not stop_event.is_set():
            line = pipe.readline()
            if not line:  # EOF
                break
            output_queue.put(('line', line.rstrip()))
    except Exception as e:
        output_queue.put(('error', str(e)))
    finally:
        output_queue.put(('eof', None))

def run_imgalign_with_robust_streaming(cmd: list, vid_name: str, batch_suffix: str, timeout_seconds: int) -> bool:
    """Run ImageAlign with robust real-time output streaming using threading."""
    print(f"    ImgAlign cmd: {' '.join(cmd)}")
    
    try:
        # Suppress warnings from ImgAlign if requested
        env = os.environ.copy()
        if not config.IMG_ALIGN_SHOW_WARNINGS:
            env['PYTHONWARNINGS'] = 'ignore'
        
        # Start process with unbuffered output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True, 
            encoding='utf-8', 
            errors='replace',
            bufsize=0,  # Unbuffered
            universal_newlines=True,
            env=env
        )
        
        # Set up threading for output reading
        output_queue = queue.Queue()
        stop_event = threading.Event()
        
        # Start output reading thread
        reader_thread = threading.Thread(
            target=read_output_stream,
            args=(process.stdout, output_queue, stop_event),
            daemon=True
        )
        reader_thread.start()
        
        start_time = time.time()
        output_lines = []
        last_progress_time = start_time
        eof_received = False
        processed_images = 0
        
        print(f"    ImgAlign ({vid_name}{batch_suffix}) started...")
        
        # Main loop to handle output and check timeout
        while True:
            current_time = time.time()
            
            # Check if process has finished
            if process.poll() is not None and eof_received:
                break
                
            # Check timeout
            if current_time - start_time > timeout_seconds:
                print(f"    ImgAlign ({vid_name}{batch_suffix}) timed out after {timeout_seconds/3600:.1f} hours")
                stop_event.set()
                try:
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)
                except:
                    pass
                return False
            
            # Read available output with timeout
            try:
                msg_type, content = output_queue.get(timeout=1.0)
                
                if msg_type == 'line' and content:
                    output_lines.append(content)
                    
                    # Count processed images (look for .png filenames)
                    if content.endswith('.png') and not any(warn in content.lower() for warn in ['warning', 'futurewarning', 'error']):
                        processed_images += 1
                    
                    # Show real-time progress
                    if config.IMG_ALIGN_ENABLE_PROGRESS_OUTPUT:
                        # Filter out warnings if requested
                        if config.IMG_ALIGN_SHOW_WARNINGS or not any(warn in content.lower() for warn in ['warning', 'futurewarning']):
                            print(f"    ImgAlign: {content}")
                    else:
                        # Print periodic updates with image count
                        if current_time - last_progress_time >= 30:  # Every 30 seconds
                            print(f"    ImgAlign progress: {processed_images} images processed, {len(output_lines)} total lines...")
                            last_progress_time = current_time
                    
                elif msg_type == 'error':
                    print(f"    ImgAlign output reader error: {content}")
                    
                elif msg_type == 'eof':
                    eof_received = True
                    
            except queue.Empty:
                # No output available, continue checking
                continue
        
        # Clean up
        stop_event.set()
        
        # Wait for process to finish
        return_code = process.wait()
        
        # Wait for reader thread to finish
        reader_thread.join(timeout=5)
        
        if return_code != 0:
            print(f"    ImgAlign ({vid_name}{batch_suffix}) failed with return code {return_code}")
            if output_lines:
                print("    Last 10 output lines:")
                for line in output_lines[-10:]:
                    if not any(warn in line.lower() for warn in ['warning', 'futurewarning']) or config.IMG_ALIGN_SHOW_WARNINGS:
                        print(f"      {line}")
            return False
        
        print(f"    ImgAlign ({vid_name}{batch_suffix}) completed successfully")
        print(f"    Processed {processed_images} images, {len(output_lines)} total output lines")
        return True
        
    except Exception as e:
        print(f"    ImgAlign ({vid_name}{batch_suffix}) error: {e}")
        return False

def run_imgalign_simple_timeout(cmd: list, vid_name: str, batch_suffix: str, timeout_seconds: int) -> bool:
    """Simple alternative that just runs with timeout, no streaming."""
    print(f"    ImgAlign cmd: {' '.join(cmd)}")
    print(f"    ImgAlign ({vid_name}{batch_suffix}) started (simple mode)...")
    
    try:
        start_time = time.time()
        
        # Suppress warnings if requested
        env = os.environ.copy()
        if not config.IMG_ALIGN_SHOW_WARNINGS:
            env['PYTHONWARNINGS'] = 'ignore'
        
        # Run with simple timeout
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout_seconds,
            env=env
        )
        
        elapsed = time.time() - start_time
        print(f"    ImgAlign ({vid_name}{batch_suffix}) finished in {elapsed/60:.1f} minutes")
        
        if process.returncode != 0:
            print(f"    ImgAlign failed with return code {process.returncode}")
            if process.stderr and config.IMG_ALIGN_SHOW_WARNINGS:
                print("    Error output:")
                for line in process.stderr.split('\n')[-10:]:  # Last 10 lines
                    if line.strip():
                        print(f"      {line}")
            return False
        
        print(f"    ImgAlign ({vid_name}{batch_suffix}) completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"    ImgAlign ({vid_name}{batch_suffix}) timed out after {timeout_seconds/3600:.1f} hours")
        return False
    except Exception as e:
        print(f"    ImgAlign ({vid_name}{batch_suffix}) error: {e}")
        return False

def process_imgalign_results(tmp_out: Path, final_lr: Path, final_hr: Path, final_ov: Path, 
                           vid_name: str, batch_suffix: str) -> tuple[int, int]:
    """Process ImageAlign results and move files to final locations."""
    out_sub = tmp_out / "Output"
    artifacts = set()
    
    # Read artifacts file if it exists
    artifacts_file = out_sub / "Artifacts.txt"
    if artifacts_file.exists():
        try: 
            with open(artifacts_file, 'r') as f:
                artifacts = {Path(line.strip()).stem for line in f if line.strip()}
        except Exception as e: 
            print(f"    Warning: Could not read Artifacts.txt: {e}")

    moved, skipped = 0, 0
    lr_output_folder = out_sub / "LR"
    
    if not lr_output_folder.is_dir():
        print(f"    Warning: No 'LR' output folder found for {vid_name}{batch_suffix}")
        return moved, skipped
    
    # Process each output file
    for lr_file in lr_output_folder.iterdir():
        if not lr_file.is_file(): 
            continue
            
        # Skip artifacts
        if lr_file.stem in artifacts:
            skipped += 1
            continue
            
        try:
            # Move LR file
            shutil.move(str(lr_file), str(final_lr / lr_file.name))
            
            # Move corresponding HR file if it exists
            hr_file = out_sub / "HR" / lr_file.name
            if hr_file.exists(): 
                shutil.move(str(hr_file), str(final_hr / lr_file.name))
            
            # Move corresponding overlay file if it exists
            ov_file = out_sub / "Overlay" / lr_file.name
            if ov_file.exists(): 
                shutil.move(str(ov_file), str(final_ov / lr_file.name))
                
            moved += 1
            
        except Exception as e: 
            print(f"    Error moving {lr_file.name}: {e}")
    
    return moved, skipped

def align_images_with_robust_streaming(lr_matched: Path, hr_matched: Path, aligned_base: Path, prog_file: Path):
    """Align image pairs using ImgAlign tool with robust streaming output."""
    stage = 'align'
    prog_data = get_progress(prog_file)
    if prog_data.get(stage, {}).get('completed', False):
        print("Alignment done. Skip.")
        return

    imgalign_exe = shutil.which("ImgAlign")
    if not imgalign_exe:
        print("Error: ImgAlign N/A. Skip align.")
        update_progress(prog_file, stage, completed=True)
        return

    if aligned_base.exists(): 
        shutil.rmtree(aligned_base)
    aligned_base.mkdir(parents=True)

    imgs_to_align = sorted([f for f in lr_matched.iterdir() 
                           if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not imgs_to_align:
        print("No images to align.")
        update_progress(prog_file, stage, completed=True)
        return

    print(f"\n--- Aligning {len(imgs_to_align)} pairs ---")
    
    # Group images by video
    groups = defaultdict(list)
    re_fname = re.compile(r"^(.*?)_frame_\d+\.png$")
    for p in imgs_to_align:
        match = re_fname.match(p.name)
        group_name = match.group(1) if match else 'unknown_align_group'
        groups[group_name].append(p)

    processed = prog_data.get(stage, {}).get('items', [])
    final_lr, final_hr, final_ov = aligned_base / "LR", aligned_base / "HR", aligned_base / "Overlay"
    final_lr.mkdir(); final_hr.mkdir(); final_ov.mkdir()

    timeout_seconds = int(config.IMG_ALIGN_TIMEOUT_HOURS * 3600)
    max_batch_size = config.IMG_ALIGN_MAX_BATCH_SIZE

    for vid_name, paths in tqdm(groups.items(), desc="Aligning videos"):
        if vid_name == 'unknown_align_group' or vid_name in processed: 
            continue

        print(f"\nAligning group: {vid_name} ({len(paths)} images)...")
        
        # Split into batches if needed
        if max_batch_size > 0 and len(paths) > max_batch_size:
            batches = [paths[i:i + max_batch_size] for i in range(0, len(paths), max_batch_size)]
            print(f"  Splitting into {len(batches)} batches of max {max_batch_size} images each")
        else:
            batches = [paths]

        batch_success_count = 0
        for batch_idx, batch_paths in enumerate(batches):
            batch_suffix = f"_batch_{batch_idx + 1}" if len(batches) > 1 else ""
            print(f"\n  Processing{batch_suffix}: {len(batch_paths)} images...")
            
            # Create temporary directories
            tmp_base = aligned_base / f"__temp_{vid_name}{batch_suffix}"
            if tmp_base.exists():
                shutil.rmtree(tmp_base)
            
            tmp_lr = tmp_base / "lr_in"
            tmp_hr = tmp_base / "hr_in" 
            tmp_out = tmp_base / "aligned_out"
            
            tmp_lr.mkdir(parents=True)
            tmp_hr.mkdir()
            tmp_out.mkdir()

            # Copy files for this batch
            valid_pairs = 0
            for lr_p_src in batch_paths:
                hr_p_src = hr_matched / lr_p_src.name
                if hr_p_src.exists():
                    try:
                        shutil.copy2(lr_p_src, tmp_lr / lr_p_src.name)
                        shutil.copy2(hr_p_src, tmp_hr / lr_p_src.name)
                        valid_pairs += 1
                    except Exception as e:
                        print(f"    Error copying {lr_p_src.name}: {e}")

            if valid_pairs == 0:
                print(f"    No valid pairs for batch{batch_suffix}. Skipping.")
                shutil.rmtree(tmp_base)
                continue

            print(f"    Prepared {valid_pairs} valid pairs for alignment")

            # Build ImgAlign command
            cmd = [
                imgalign_exe,
                "-s", str(config.IMG_ALIGN_SCALE),
                "-m", "0",  # Motion estimation method
                "-g", str(tmp_hr),  # HR (ground truth) folder
                "-l", str(tmp_lr),  # LR folder
                "-c",  # Create HR copy
                "-j",  # ???
                "-ai",  # ???
                "-o", str(tmp_out)  # Output folder
            ]
            
            # Add interpolation parameter based on config
            if config.IMG_ALIGN_ENABLE_LR_COLOR_COPY:
                cmd.extend(["-i", "1"])  # Copy LR color to HR
                print(f"    Using LR color copying mode (-i 1)")
            # If disabled, don't add -i parameter at all (ImgAlign default)
            
            # Try robust streaming first, fall back to simple if it fails
            success = False
            
            try:
                print(f"    Attempting robust streaming mode...")
                success = run_imgalign_with_robust_streaming(cmd, vid_name, batch_suffix, timeout_seconds)
            except Exception as e:
                print(f"    Robust streaming failed: {e}")
                print(f"    Falling back to simple timeout mode...")
                success = run_imgalign_simple_timeout(cmd, vid_name, batch_suffix, timeout_seconds)
            
            if not success:
                print(f"    Batch{batch_suffix} failed. Skipping...")
                shutil.rmtree(tmp_base)
                continue

            # Process results - move aligned images to final location
            moved, skipped = process_imgalign_results(tmp_out, final_lr, final_hr, final_ov, vid_name, batch_suffix)
            print(f"    Batch{batch_suffix}: Moved {moved} aligned pairs, skipped {skipped} artifacts")
            
            if moved > 0:
                batch_success_count += 1
            
            # Clean up temporary files
            shutil.rmtree(tmp_base)

        print(f"  Video {vid_name}: {batch_success_count}/{len(batches)} batches successful")
        update_progress(prog_file, stage, item=vid_name)

    update_progress(prog_file, stage, completed=True)
    print("\n--- Image Alignment Done ---")

if __name__ == "__main__":
    t_start = time.time()
    FFMPEG_PATH = _find_executable("ffmpeg", config.FFMPEG_PATH)
    FFPROBE_PATH = _find_executable("ffprobe", config.FFPROBE_PATH)
    if not FFMPEG_PATH or not FFPROBE_PATH:
        print("FATAL: ffmpeg/ffprobe N/A.")
        exit(1)

    lr_in, hr_in, out_base = Path(config.LR_INPUT_VIDEO_FOLDER), Path(config.HR_INPUT_VIDEO_FOLDER), Path(config.OUTPUT_BASE_FOLDER)
    out_base.mkdir(parents=True, exist_ok=True)
    extracted_fld, matched_fld, aligned_fld = out_base / config.EXTRACTED_SUBFOLDER_NAME, out_base / config.MATCHED_SUBFOLDER_NAME, out_base / config.ALIGNED_SUBFOLDER_NAME
    prog_f = out_base / config.PROGRESS_FILENAME
    lr_matched_out, hr_matched_out = matched_fld / "LR", matched_fld / "HR"
    extracted_fld.mkdir(exist_ok=True); matched_fld.mkdir(exist_ok=True); lr_matched_out.mkdir(exist_ok=True); hr_matched_out.mkdir(exist_ok=True)

    print("--- Pipeline Configuration ---")
    for k, v in [("LR Input", lr_in), ("HR Input", hr_in), ("Output Base", out_base), ("Deinterlace Mode", config.DEINTERLACE_MODE), ("Chroma Upsampling", config.ENABLE_CHROMA_UPSAMPLING), ("HDR Tone Mapping", config.ENABLE_HDR_TONE_MAPPING), ("HDR Mode", getattr(config, 'HDR_TONE_MAPPING_MODE', 'auto')), ("Content Fix", config.ATTEMPT_CONTENT_FIX), ("Autocrop Black", f"{config.CROP_BLACK_BORDERS}, Thresh: {config.CROP_BLACK_THRESHOLD if config.CROP_BLACK_BORDERS else 'N/A'}"), ("Autocrop White", f"{config.CROP_WHITE_BORDERS}, Thresh: {config.CROP_WHITE_THRESHOLD if config.CROP_WHITE_BORDERS else 'N/A'}"), ("Low Info Filter", f"{config.ENABLE_LOW_INFO_FILTER}, Thresh: {config.LOW_INFO_VARIANCE_THRESHOLD if config.ENABLE_LOW_INFO_FILTER else 'N/A'}"), ("Match Thresh", config.MATCH_THRESHOLD), ("pHash Thresh", config.PHASH_SIMILARITY_THRESHOLD), ("ImgAlign Scale", config.IMG_ALIGN_SCALE), ("ImgAlign Timeout", f"{config.IMG_ALIGN_TIMEOUT_HOURS}h"), ("ImgAlign Batch Size", config.IMG_ALIGN_MAX_BATCH_SIZE), ("ImgAlign LR Color", config.IMG_ALIGN_ENABLE_LR_COLOR_COPY)]:
        print(f"{k}: {v}")

    print("\n=== Stage 1: Preprocessing (Frame Extraction) ===")
    lr_ext_base, hr_ext_base = preprocess_videos(lr_in, hr_in, extracted_fld, prog_f)

    if config.ATTEMPT_CONTENT_FIX:
        fix_mismatched_content_in_folders(lr_ext_base, hr_ext_base, prog_f)
    else:
        print("\nSkipping Mismatched Content Fix (disabled).")
    
    if config.CROP_BLACK_BORDERS or config.CROP_WHITE_BORDERS:
        autocrop_frames_in_folders(lr_ext_base, prog_f, "_LR")
        autocrop_frames_in_folders(hr_ext_base, prog_f, "_HR")
    else:
        print("\nSkipping Autocrop (disabled).")

    if config.ENABLE_LOW_INFO_FILTER:
        filter_low_information_images(lr_ext_base, hr_ext_base, prog_f)
    else:
        print("\nSkipping Low Info Filter (disabled).")

    print("\n=== Stage 2: Temporal Matching ===")
    process_folder_pair(lr_ext_base, hr_ext_base, lr_matched_out, hr_matched_out, prog_f)
    
    print("\n=== Stage 3: Deduplication ===")
    if config.PHASH_SIMILARITY_THRESHOLD >= 0:
        filter_similar_images_phash(lr_matched_out, hr_matched_out, prog_f)
    else:
        print("Skipping pHash Deduplication (disabled).")
    
    print("\n=== Stage 4: Final Alignment ===")
    align_images_with_robust_streaming(lr_matched_out, hr_matched_out, aligned_fld, prog_f)

    total_time = time.time() - t_start
    print(f"\n--- Pipeline Finished --- Total Time: {total_time/60:.2f} minutes ---")
    print("\n=== Final Output Summary ===")
    if lr_matched_out.exists(): print(f"Matched pairs (before alignment): {len(list(lr_matched_out.glob('*.png')))}")
    if (aligned_lr := aligned_fld / "LR").exists(): print(f"Aligned pairs: {len(list(aligned_lr.glob('*.png')))}")
    print(f"\nOutput directory: {out_base}")
    print("Pipeline completed successfully!")
