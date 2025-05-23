## Acknowledgements for Code & Inspiration
1. Anthorpic's Calude (https://claude.ai/new)
2. Enhance Everything Discord
3. neosr - https://github.com/neosr-project/neosr

-------------------------------

# Video-to-Paired-Image Dataset Creation Pipeline (SRDC)

This project provides a comprehensive Python-based pipeline for generating high-quality, paired Low-Resolution (LR) and High-Resolution (HR) image datasets from video sources. Such datasets are crucial for training Super-Resolution (SR) models and other image restoration tasks.

The pipeline automates various stages including frame extraction, deinterlacing, scene change detection, content-aware filtering, image matching, alignment, and advanced selection based on image complexity and visual uniqueness using CLIP.

## Features

*   **Video Frame Extraction**: Extracts frames from LR and HR video pairs using FFmpeg.
*   **Scene Change Detection**: Selects frames primarily from stable scenes, avoiding rapid transitions.
*   **Deinterlacing**: Optional automatic or manual deinterlacing for interlaced video sources.
*   **Autocropping**: Removes black borders from extracted frames.
*   **Low-Information Filtering**: Discards overly plain or uninformative frames/pairs based on image variance.
*   **LR/HR Frame Matching**: Accurately pairs corresponding LR and HR frames using template matching (with GPU acceleration).
*   **Similarity Filtering**: Removes duplicate or near-duplicate image pairs using perceptual hashing (pHash).
*   **Image Alignment**: Spatially aligns LR and HR image pairs using the [ImgAlign tool](https://github.com/NicholasGU/ImgAlign).
*   **Advanced SISR Filtering**: Utilizes the `SuperResImageSelector` module to:
    *   Calculate image complexity (entropy, edge density, sharpness) and brightness.
    *   Extract CLIP (Contrastive Language-Image Pre-Training) visual features.
    *   Select a diverse set of images suitable for SR training by filtering based on complexity, brightness, and CLIP feature distance.
*   **Progress Tracking & Resumption**: Saves progress in a `progress.json` file, allowing the pipeline to be resumed.
*   **Modular Design**: Each processing stage is distinct and can be configured or (in some cases) skipped.
*   **Configurable**: Extensive options available directly in the main script (`srdc_pipeline.py`).

## System Requirements

*   Python 3.7+
*   **Python Packages**:
    *   `torch` & `torchvision`
    *   `transformers`
    *   `Pillow (PIL)`
    *   `opencv-python` (cv2)
    *   `scipy`
    *   `numpy`
    *   `tqdm`
    *   `psutil`
    *   `imagehash`
    *   (A `requirements.txt` file would be beneficial)
*   **External Tools**:
    *   **FFmpeg**: Must be installed and accessible in your system's PATH. ([Download FFmpeg](https://ffmpeg.org/download.html))
    *   **ImgAlign**: Must be compiled and accessible in your system's PATH. ([Download ImgAlign](https://github.com/NicholasGU/ImgAlign))
*   **Hardware**:
    *   A CUDA-capable GPU is highly recommended for significant speedups in PyTorch (CLIP) and OpenCV operations. The pipeline includes CPU fallbacks.
    *   Sufficient RAM, especially for the `SuperResImageSelector` when processing a large number of candidate images.
    *   Disk space for storing extracted frames and processed datasets.

## File Structure
.
├── srdc_pipeline.py # Main pipeline script (srdc_v3.py in your files)
├── sisr_image_selector.py # Module for SISR-suitability filtering
├── LR_VIDEOS/ # Your input Low-Resolution videos
│ └── video1.mp4
├── HR_VIDEOS/ # Your input High-Resolution videos
│ └── video1.mp4
└── OUTPUT_BASE_FOLDER/ # Root output directory (configurable)
├── 1_EXTRACTED/
│ ├── LR/
│ │ └── video1/ # Frames per video
│ │ └── frame_000001.png
│ └── HR/
│ └── video1/
│ └── frame_000001.png
├── 2_MATCHED/
│ ├── LR/
│ │ └── video1_frame_000001.png
│ └── HR/
│ └── video1_frame_000001.png
├── 3_ALIGNED/
│ ├── LR/
│ ├── HR/
│ └── Overlay/ # Alignment visualization
├── 4_SISR_FILTERED/ # Optional final filtered output
│ ├── LR/
│ └── HR/
├── progress.json # Tracks pipeline progress for resumption
└── temp_YYYYMMDD_HHMMSS/ # Temporary directory for sisr_image_selector checkpoints


## Configuration

All major configuration options are located at the beginning of the `srdc_pipeline.py` script. Key parameters include:

*   Input/Output paths (`lr_input_video_folder`, `hr_input_video_folder`, `output_base_folder`).
*   Frame extraction settings (`begin_time`, `end_time`, `scene_threshold`, scaling options).
*   Deinterlacing options (`DEINTERLACE_MODE`, `DEINTERLACE_FILTER`).
*   Autocropping settings (`CROP_BLACK_BORDERS`, `CROP_BLACK_THRESHOLD`).
*   Low information filter settings (`ENABLE_LOW_INFO_FILTER`, `LOW_INFO_VARIANCE_THRESHOLD`).
*   Matching parameters (`match_threshold`, `resize_height`, `resize_width`).
*   Similarity (pHash) filter threshold (`similarity_threshold`).
*   ImgAlign scale (`img_align_scale`).
*   SISR filter settings (`ENABLE_SISR_FILTERING`, `SISR_MIN_CLIP_DISTANCE`, `SISR_COMPLEXITY_THRESHOLD`, `SISR_MAX_BRIGHTNESS`).

Please review and adjust these parameters to suit your specific video sources and dataset requirements.

## Usage

1.  **Prepare Input Videos**: Place your LR videos in the `lr_input_video_folder` and corresponding HR videos (with matching filenames) in the `hr_input_video_folder`.
2.  **Configure Pipeline**: Open `srdc_pipeline.py` and modify the configuration variables at the top as needed.
3.  **Run Pipeline**: Execute the main script from your terminal:
    ```bash
    python srdc_pipeline.py
    ```
4.  **Monitor Progress**: The script will output progress to the console. Intermediate results will be saved in the configured `output_base_folder`.
5.  **Resumption**: If the pipeline is interrupted, it can often be resumed by running the script again. It will attempt to pick up from the last completed stage/video based on `progress.json`.

### `sisr_image_selector.py` Standalone Usage

While integrated into the main pipeline, `sisr_image_selector.py` can also be run as a standalone tool:

```bash
python sisr_image_selector.py <input_folder> [--min_distance 0.15] [--complexity_threshold 0.4] [--max_brightness 200]
