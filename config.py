"""
This configuration file contains all the user-adjustable settings for the SRDC pipeline.
Tweak these parameters to suit your specific video sources and dataset requirements.
"""

import os

# --- Core Paths ---
# These paths MUST be configured correctly for the pipeline to run.
LR_INPUT_VIDEO_FOLDER = "E:\\Lupin\\LR"  # Path to the folder containing Low-Resolution videos.
HR_INPUT_VIDEO_FOLDER = "E:\\Lupin\\HR" # Path to the folder containing High-Resolution videos.
OUTPUT_BASE_FOLDER = "E:\\Lupin\\Output" # Base directory where all processed folders and files will be saved.

# --- Stage 1: Frame Extraction ---
BEGIN_TIME = "00:00:00"  # Start time for frame extraction, in "HH:MM:SS" format.
END_TIME = "00:00:00"    # End time for frame extraction. "00:00:00" means process until the end.
SCENE_THRESHOLD = 0.20   # Sensitivity for scene-change detection (0.0 to 1.0). Lower values detect more changes.

# --- Deinterlacing ---
# Automatically handles interlaced video sources, common in older TV broadcasts or DVDs.
DEINTERLACE_MODE = 'auto' # 'none', 'auto' (recommended), 'lr', 'hr', 'both'.
DEINTERLACE_FILTER = 'bwdif=mode=send_frame:parity=auto' # The FFmpeg filter to use for deinterlacing.

# --- Color Space & Fidelity ---
# Fixes common color issues when extracting frames from video files.
ENABLE_CHROMA_UPSAMPLING = True # RECOMMENDED. Fixes blocky color edges on most videos by using a high-quality scaler.
CHROMA_UPSAMPLING_FILTER = 'scale=sws_flags=lanczos:in_color_matrix=bt709,format=yuv444p'

# --- HDR Tone Mapping (Enhanced) ---
ENABLE_HDR_TONE_MAPPING = False # Enable ONLY for High Dynamic Range (HDR) sources (e.g., 4K Blu-rays) to prevent washed-out colors.
HDR_TONE_MAPPING_FILTER = 'zscale=t=linear:npl=100,tonemap=tonemap=hable,zscale=p=bt709:t=bt709:m=bt709,format=bgr24'

# Choose which video types get HDR tone mapping: 'hr_only', 'lr_only', 'both', 'auto'
HDR_TONE_MAPPING_MODE = 'auto'  # 'auto' = only apply to videos detected as HDR

# --- Stage 2: Mismatched Content Correction ---
# Crucial for matching sources with different aspect ratios (e.g., Widescreen vs. Fullscreen).
ATTEMPT_CONTENT_FIX = True # Automatically detects and crops the "container" frame (either LR or HR) to match the "content" frame.
CONTENT_FIX_THRESHOLD = 0.75 # Confidence score (0.0 to 1.0) needed to confirm a content mismatch before cropping.

# --- Stage 3: Border Cropping ---
CROP_BLACK_BORDERS = True
CROP_WHITE_BORDERS = False
CROP_BLACK_THRESHOLD = 15  # Pixels below this grayscale value are considered part of a black border.
CROP_WHITE_THRESHOLD = 240 # Pixels above this grayscale value are considered part of a white border.
CROP_MIN_CONTENT_DIMENSION = 300 # After cropping, the image must be at least this tall/wide to be kept.

# --- Stage 4: Low Information Filtering ---
ENABLE_LOW_INFO_FILTER = False # If enabled, removes overly simple frames (e.g., solid colors, fades).
LOW_INFO_VARIANCE_THRESHOLD = 100 # Threshold for pixel variance. Lower values are more aggressive.
LOW_INFO_CHECK_HR_TOO = False # Also check the HR frame for low information.

# --- Stage 5: Temporal Frame Matching ---
MATCH_THRESHOLD = 0.65 # Similarity score (0.0 to 1.0) needed to consider two frames a temporal match.
MATCH_RESIZE_HEIGHT = 540 # For speed, frames are resized to this height before the initial temporal match.
MATCH_RESIZE_WIDTH = int(MATCH_RESIZE_HEIGHT * (1.77/1)) # Aspect ratio for resize, e.g., 16/9 or 4/3 - currently a balance in between

# Defines the time window (in seconds) to search for a matching HR frame.
INITIAL_MATCH_CANDIDATE_WINDOW_PERCENTAGE = 0.06 # For the first match, search +/- this % of the total video duration.
SUBSEQUENT_MATCH_CANDIDATE_WINDOW_SECONDS = 5   # For later matches, use a smaller, fixed window to account for drift.
FALLBACK_MATCH_CANDIDATE_WINDOW_SECONDS = 10.0 # Fallback window if video duration is unknown.

# --- Stage 6: Deduplication ---
PHASH_SIMILARITY_THRESHOLD = 4 # Removes visually similar pairs. Lower is stricter. Set to -1 to disable.

# --- Stage 7: Final Alignment ---
IMG_ALIGN_SCALE = 2 # The integer scale factor between your LR and HR sources (e.g., 1080p HR is 2x a 540p LR).

# --- ImageAlign Advanced Options ---
IMG_ALIGN_ENABLE_LR_COLOR_COPY = False # If True, uses -i 1 to copy LR color to generated HR pairs. If False, omits -i parameter entirely.

# --- ImageAlign Timeout & Batching (Enhanced) ---
IMG_ALIGN_TIMEOUT_HOURS = 12.0  # Timeout in hours (increased for reliability)
IMG_ALIGN_MAX_BATCH_SIZE = 50   # Maximum images per batch (0 = no limit, smaller = more reliable)
IMG_ALIGN_ENABLE_PROGRESS_OUTPUT = True  # Show real-time progress from ImageAlign

# --- ImageAlign Warning & Output Control ---
IMG_ALIGN_SHOW_WARNINGS = False  # Set to False to suppress FutureWarnings and other warnings from ImgAlign
IMG_ALIGN_RETRY_FAILED_BATCHES = True   # Retry failed batches once with simple timeout method
IMG_ALIGN_FALLBACK_TO_SIMPLE = True    # Use simple timeout if streaming fails

# --- Internal Naming Conventions (Do Not Edit) ---
EXTRACTED_SUBFOLDER_NAME = "1_EXTRACTED"
MATCHED_SUBFOLDER_NAME = "2_MATCHED"
ALIGNED_SUBFOLDER_NAME = "3_ALIGNED"
PROGRESS_FILENAME = "progress.json"
METADATA_FILENAME = "metadata.json"

# --- Tool Paths (Optional) ---
# If FFmpeg/FFprobe are not in your system's PATH, specify their full paths here.
FFMPEG_PATH = None
FFPROBE_PATH = None
