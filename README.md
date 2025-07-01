# SRDC: A Video-to-Paired-Image Dataset Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/your-username/your-repo/pulls)

SRDC is a powerful, automated Python pipeline designed to convert pairs of Low-Resolution (LR) and High-Resolution (HR) videos into high-quality, aligned image datasets. These datasets are essential for training and evaluating Super-Resolution (SR), de-noising, and other image restoration models.

The pipeline intelligently handles complex real-world video challenges, such as mismatched aspect ratios (widescreen vs. fullscreen), color space differences (SDR/HDR), and temporal drift, to produce clean, content-matched, and spatially aligned image pairs.

<!-- TODO: Consider creating a GIF showing an LR input, HR input, and the final aligned Overlay output to showcase the pipeline's effectiveness. -->
<!-- <p align="center"><img src="docs/srdc_pipeline_demo.gif" width="800"></p> -->

## Key Features

*   **High-Quality Frame Extraction**: Uses FFmpeg with advanced filters to ensure maximum color fidelity, preventing common issues like blocky color edges (`chroma upsampling`) and washed-out colors from HDR sources (`tone mapping`).
*   **Widescreen / Pan-and-Scan Correction**: A critical pre-processing step that automatically detects and crops widescreen footage to match the content of a corresponding fullscreen (pan-and-scan) version, dramatically improving alignment success.
*   **Versatile Border Cropping**: Intelligently removes black *and/or* white letterbox/pillarbox bars from frames.
*   **Robust Temporal Matching**: A multi-stage process that uses a fast, downscaled template match to find corresponding frames in time, with a sliding window to account for drift (e.g., PAL vs. NTSC speed differences).
*   **Sub-Pixel Image Alignment**: Leverages the powerful `ImgAlign` tool to perform a final, precise spatial alignment of the matched pairs, correcting for minor shifts, rotation, and scaling.
*   **Content-Aware Filtering**:
    *   **Low-Information Filter**: Discards overly simple frames (e.g., solid black or white) based on variance.
    *   **Perceptual Deduplication**: Uses pHash to remove visually redundant or near-identical image pairs.
*   **Advanced Dataset Curation with CLIP**: An optional final stage that uses the `SuperResImageSelector` to build a visually diverse and complex dataset, ideal for training robust SR models. It filters images based on complexity, brightness, and visual uniqueness (CLIP feature distance).
*   **Resumable & Organized**: The entire pipeline is broken into logical stages with progress saved to a `progress.json` file, allowing you to stop and resume processing. Outputs are neatly organized into folders for each stage.
*   **Highly Configurable**: A central `config.py` file allows easy tuning of every aspect of the pipeline without touching the core logic.

## Pipeline Workflow

The script executes a series of stages, with the output of one stage becoming the input for the next.

```
Input Videos (LR/HR)
     │
     ▼
[ 1. Frame Extraction ] ──> (Deinterlacing, Chroma Upsampling, HDR Tone Mapping)
     │
     ▼
[ 2. Pan & Scan Fix ] ──> (Crops widescreen to match fullscreen content)
     │
     ▼
[ 3. Border Cropping ] ──> (Removes letterbox/pillarbox bars)
     │
     ▼
[ 4. Frame Matching ] ──> (Finds LR/HR pairs corresponding in time)
     │
     ▼
[ 5. Deduplication ] ──> (Removes visually similar pairs via pHash)
     │
     ▼
[ 6. Alignment ] ──> (Spatially aligns the final pairs using ImgAlign)
     │
     ▼
[ 7. SISR Filtering (Optional) ] ──> (Selects a diverse subset using CLIP)
     │
     ▼
Final Paired Dataset
```

## Requirements

#### External Tools
These must be installed and accessible in your system's PATH.
*   **FFmpeg**: For video processing. ([Download](https://ffmpeg.org/download.html))
*   **ImgAlign**: For the final alignment step. ([Download & Compile from GitHub](https://github.com/NicholasGU/ImgAlign))

#### Python Environment
*   Python 3.9+ is recommended.
*   A CUDA-capable GPU is **highly recommended** for acceptable performance in the alignment (`ImgAlign`) and SISR filtering (`PyTorch`) stages.
*   Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
*   Install the required packages. A `requirements.txt` can be created from a working environment with `pip freeze > requirements.txt`. A typical set of requirements would be:
    ```text
    # requirements.txt
    opencv-python
    numpy
    torch
    torchvision
    transformers
    Pillow
    imagehash
    scipy
    tqdm
    psutil
    ```
    Install them with:
    ```bash
    pip install -r requirements.txt
    ```

## File Structure

```
.
├── srdc_pipeline.py            # The main pipeline script.
├── config.py                   # All user-configurable settings.
├── README.md                   # This file.
├── LR/                           # Your input Low-Resolution videos.
│   └── movie_v1.mp4
└── HR/                           # Your input High-Resolution videos.
    └── movie_v1.mkv              # <-- Note: Extension can be different.
```
The pipeline will generate an `Output` folder (name configurable) with the following structure:
```
OUTPUT_BASE_FOLDER/
├── 1_EXTRACTED/                # Raw frames extracted from videos.
├── 2_MATCHED/                  # Temporally matched but unaligned pairs.
├── 3_ALIGNED/                  # Spatially aligned, clean pairs.
│   ├── LR/
│   ├── HR/
│   └── Overlay/                # Visualizations of the alignment.
├── 4_SISR_FILTERED/            # Optional, highly-curated final dataset.
│   ├── LR/
│   └── HR/
└── progress.json               # Tracks pipeline state for resumption.
```

## Setup & Usage

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/stinkybread/super_resolution_dataset_creator.git
    cd super_resolution_dataset_creator
    ```
2.  **Install Dependencies**: Ensure Python, FFmpeg, and ImgAlign are installed and in your PATH. Install Python packages as described in the *Requirements* section.

3.  **Prepare Videos**: Place your LR and HR videos into their respective input folders (e.g., `LR/` and `HR/`). The script will match videos based on their filenames, ignoring the extension (e.g., `movie.mp4` will match `movie.mkv`).

4.  **Configure the Pipeline**: Open `config.py` in a text editor. Adjust the paths, and enable/disable features to suit your needs. **Review this file carefully.** The most important settings are:
    *   `LR_INPUT_VIDEO_FOLDER`, `HR_INPUT_VIDEO_FOLDER`, `OUTPUT_BASE_FOLDER`.
    *   `ATTEMPT_PAN_AND_SCAN_FIX`: **Set to `True`** if you are matching widescreen and fullscreen videos.
    *   `ENABLE_CHROMA_UPSAMPLING`: Keep `True` for most SDR videos.
    *   `ENABLE_HDR_TONE_MAPPING`: Set to `True` only for HDR source material.

5.  **Run the Pipeline**:
    ```bash
    python srdc_pipeline.py
    ```
    The script will print its progress for each stage. If it's interrupted, you can typically run it again to resume from where it left off.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
*   The Anthropic team for Claude.
*   The Google AI team for AI Studio.
*   The Enhance Everything Discord community for inspiration and discussion.
*   The [neosr](https://github.com/neosr-project/neosr) project for concepts in super-resolution.
