# config.py

import os

# --- Input/Output Paths ---
LR_INPUT_VIDEO_FOLDER = "E:\\Movie7\\\LR"  # Path to the folder containing Low-Resolution videos
HR_INPUT_VIDEO_FOLDER = "E:\\Movie7\\HR" # Path to the folder containing High-Resolution videos
OUTPUT_BASE_FOLDER = "E:\\Movie7\\Output" # Base directory where all processed files will be saved

# --- Frame Extraction & Preprocessing ---
BEGIN_TIME = "00:00:00"  # Start time for frame extraction "HH:MM:SS"
END_TIME = "00:00:00"    # End time for frame extraction "HH:MM:SS"
SCENE_THRESHOLD = 0.15   # Scene change detection sensitivity for FFmpeg (0.0 to 1.0).

LR_WIDTH, LR_HEIGHT = 720, 540 # Target width/height for LR frames. None to keep original.
LR_SCALE = None          # Scale factor for LR frames. Overrides width/height.
HR_WIDTH, HR_HEIGHT = None, None # Target width/height for HR frames. None to keep original.
HR_SCALE = None          # Scale factor for HR frames. Overrides width/height.

# --- Deinterlacing Options ---
DEINTERLACE_MODE = 'none' # Options: 'none', 'lr', 'hr', 'both', 'auto'
DEINTERLACE_FILTER = 'bwdif=mode=send_frame:parity=auto'

# --- Autocropping Configuration ---
CROP_BLACK_BORDERS = True
CROP_BLACK_THRESHOLD = 15
CROP_MIN_CONTENT_DIMENSION = 300

# --- Low Information Filter Configuration (Pre-Match) ---
ENABLE_LOW_INFO_FILTER = False
LOW_INFO_VARIANCE_THRESHOLD = 100
LOW_INFO_CHECK_HR_TOO = False

# --- Frame Matching (Template Matching) ---
# Using TM_CCOEFF_NORMED, where higher score (closer to 1.0) is better.
MATCH_THRESHOLD = 0.65 # Threshold for cv2.TM_CCOEFF_NORMED. Adjust based on content.
MATCH_RESIZE_HEIGHT = 540 # Internal height for resizing frames *before* template matching.
MATCH_RESIZE_WIDTH = int(MATCH_RESIZE_HEIGHT * (4/3)) # Internal width.

# Percentage of LR video duration for the *initial* HR candidate search window (e.g., for the first LR frame).
# 0.06 means +/- 6% of total LR video duration from the LR frame's timestamp.
INITIAL_MATCH_CANDIDATE_WINDOW_PERCENTAGE = 0.06

# Fixed seconds window to search for HR candidates around the *expected* HR time
# for *subsequent* LR frames (after a first match is found and state is established).
# This window accounts for local drift (e.g., PAL speedup over shorter segments).
SUBSEQUENT_MATCH_CANDIDATE_WINDOW_SECONDS = 5 # e.g., +/- 5 seconds from expected HR time

# Fallback fixed window in seconds if percentage-based calculation is problematic (e.g., duration is 0)
# or for very short videos.
FALLBACK_MATCH_CANDIDATE_WINDOW_SECONDS = 10.0 # e.g., +/- 10 seconds

# --- Similarity Filtering (Perceptual Hash - Post-Match, Pre-Align) ---
PHASH_SIMILARITY_THRESHOLD = 4 # pHash difference. Lower is stricter. -1 to disable.

# --- Image Alignment (ImgAlign) ---
IMG_ALIGN_SCALE = 2 # Integer scale factor (e.g., HR is 2x LR).

# --- SISR Filtering (Post-Align) ---
ENABLE_SISR_FILTERING = False
SISR_MIN_CLIP_DISTANCE = 0.10
SISR_COMPLEXITY_THRESHOLD = 0.3
SISR_MAX_BRIGHTNESS = 220 # Images with average brightness above this are skipped by SISR

# --- Derived Paths (DO NOT EDIT MANUALLY if OUTPUT_BASE_FOLDER is set correctly) ---
EXTRACTED_SUBFOLDER_NAME = "1_EXTRACTED"
MATCHED_SUBFOLDER_NAME = "2_MATCHED"
ALIGNED_SUBFOLDER_NAME = "3_ALIGNED"
SISR_FILTERED_SUBFOLDER_NAME = "4_SISR_FILTERED"
PROGRESS_FILENAME = "progress.json"
METADATA_FILENAME = "metadata.json" # For storing duration and timestamps

# --- FFMPEG/FFPROBE Paths (Optional) ---
# If None, the script will try to find them in PATH.
FFMPEG_PATH = None
FFPROBE_PATH = None
