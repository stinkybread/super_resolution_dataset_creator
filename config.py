# --- START OF FILE config.py ---

# config.py

import os

# --- Input/Output Paths ---
LR_INPUT_VIDEO_FOLDER = "E:\\Movie7\\LR"  # Path to the folder containing Low-Resolution videos
HR_INPUT_VIDEO_FOLDER = "E:\\Movie7\\HR" # Path to the folder containing High-Resolution videos
OUTPUT_BASE_FOLDER = "E:\\Movie7\\Output" # Base directory where all processed files will be saved

# --- Frame Extraction & Preprocessing ---
BEGIN_TIME = "00:00:00"  # Start time for frame extraction "HH:MM:SS"
END_TIME = "00:00:00"    # End time for frame extraction "HH:MM:SS"
SCENE_THRESHOLD = 0.25   # Scene change detection sensitivity for FFmpeg (0.0 to 1.0).

LR_WIDTH, LR_HEIGHT = None, None # Target width/height for LR frames. None to keep original.
LR_SCALE = None          # Scale factor for LR frames. Overrides width/height.
HR_WIDTH, HR_HEIGHT = None, None # Target width/height for HR frames. None to keep original.
HR_SCALE = None          # Scale factor for HR frames. Overrides width/height.

# --- Deinterlacing Options ---
DEINTERLACE_MODE = 'none' # Options: 'none', 'lr', 'hr', 'both', 'auto'
DEINTERLACE_FILTER = 'bwdif=mode=send_frame:parity=auto'

# --- High-Quality Chroma Upsampling (SDR Color Fidelity) ---
# Enable this to fix the "blocky color" issue. Highly recommended for standard videos.
ENABLE_CHROMA_UPSAMPLING = True
CHROMA_UPSAMPLING_FILTER = 'scale=sws_flags=lanczos:in_color_matrix=bt709,format=yuv444p'

# --- HDR to SDR Tone Mapping (HDR Color Correction) ---
# Enable this ONLY if your source videos are in HDR (e.g., 4K HDR Blu-ray).
ENABLE_HDR_TONE_MAPPING = False # Default to False.
HDR_TONE_MAPPING_FILTER = 'zscale=t=linear:npl=100,tonemap=tonemap=hable,zscale=p=bt709:t=bt709:m=bt709,format=bgr24'

# --- Pan & Scan / Mismatched Content Fix ---
# Enable this if matching a widescreen source against a fullscreen (pan-and-scan) version.
# This will crop the widescreen image to match the fullscreen content before alignment.
# This is computationally intensive but crucial for these scenarios.
ATTEMPT_PAN_AND_SCAN_FIX = True

# --- Autocropping Configuration ---
CROP_BLACK_BORDERS = True
CROP_WHITE_BORDERS = False  # Enable to crop white/light-colored borders
CROP_BLACK_THRESHOLD = 15  # Pixels below this value are considered black border
CROP_WHITE_THRESHOLD = 240 # Pixels above this value are considered white border
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

# Percentage of LR video duration for the *initial* HR candidate search window.
INITIAL_MATCH_CANDIDATE_WINDOW_PERCENTAGE = 0.06

# Fixed seconds window to search for HR candidates for *subsequent* LR frames.
SUBSEQUENT_MATCH_CANDIDATE_WINDOW_SECONDS = 5

# Fallback fixed window in seconds for very short videos or if duration is unknown.
FALLBACK_MATCH_CANDIDATE_WINDOW_SECONDS = 10.0

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
# --- END OF FILE config.py ---
