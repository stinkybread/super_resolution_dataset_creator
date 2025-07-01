# SRDC: A Video-to-Paired-Image Dataset Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/stinkybread/super_resolution_dataset_creator/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/stinkybread/super_resolution_dataset_creator/pulls)

SRDC is a powerful, automated Python pipeline designed to convert pairs of Low-Resolution (LR) and High-Resolution (HR) videos into high-quality, aligned image datasets. These datasets are essential for training and evaluating Super-Resolution (SR), de-noising, and other image restoration models.

The pipeline intelligently handles complex real-world video challenges, such as mismatched aspect ratios (widescreen vs. fullscreen), color space differences (SDR/HDR), and temporal drift, to produce clean, content-matched, and spatially aligned image pairs.

<p align="center">
  <em>(A visual demonstration of an unaligned pair vs. a final aligned overlay)</em><br>
  <img src="https://pub-c7629c2cc932473c8827d6974dac86ea.r2.dev/1751376357.1178532.png" width="1024" alt="Demonstration of image alignment">
</p>

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

#### 1. System Dependencies
These must be installed first and accessible in your system's `PATH`.
*   **NVIDIA CUDA Toolkit**: You must have the NVIDIA drivers and CUDA Toolkit installed to use a GPU. You can check your version with `nvcc --version`.
*   **FFmpeg**: For all video processing. ([Download](https://ffmpeg.org/download.html))
*   **ImgAlign**: For the final alignment step. ([Download & Compile from GitHub](https://github.com/NicholasGU/ImgAlign))

#### 2. Python Environment & Packages

It is highly recommended to use a dedicated Python virtual environment.

**Step 1: Create and Activate Environment**
```bash
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

**Step 2: Install PyTorch with CUDA Support (CRITICAL STEP)**

Do **not** install PyTorch using a generic command. You must install the version that matches your system's CUDA Toolkit.

1.  Go to the **[Official PyTorch Get Started page](https://pytorch.org/get-started/locally/)**.
2.  Use the interactive tool to select your system configuration (e.g., Stable, Windows, Pip, Python, CUDA 12.1).
3.  Copy the generated command and run it in your activated virtual environment.

It will look something like this ( **DO NOT copy this command directly, get the correct one from the website!** ):
```bash
# Example command for CUDA 12.1 - verify on the PyTorch website!
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
This ensures that `torch`, `ImgAlign`, and the SISR filter can all leverage your GPU for maximum performance.

**Step 3: Install Remaining Packages**

Once PyTorch is installed correctly, install the rest of the required packages. Create a file named `requirements.txt` with the following content:
```text
# requirements.txt
opencv-python
numpy
transformers
Pillow
imagehash
scipy
tqdm
psutil
```

Then, run the following command in your activated virtual environment:
```bash
pip install -r requirements.txt
```

## File Structure

Your project folder should be set up like this:
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
├── 1_EXTRACTED/                # Raw frames from videos.
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
2.  **Install Dependencies**: Follow the steps in the **Requirements** section above.

3.  **Prepare Videos**: Place your LR and HR videos into their respective input folders (e.g., `LR/` and `HR/`). The script will match videos based on their filenames, ignoring the extension.

4.  **Configure the Pipeline**: Open `config.py` in a text editor. Adjust the paths, and enable/disable features to suit your needs. See the table below for key settings.

5.  **Run the Pipeline**:
    ```bash
    python srdc_pipeline.py
    ```
    The script will print its progress for each stage. If it's interrupted, you can typically run it again to resume from where it left off.

## Key Configuration Options

All settings are in `config.py`. Here are the most important ones to review:

| Parameter                  | Description                                                                                                                              | Recommendation                                                                      |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `LR_INPUT_VIDEO_FOLDER`    | Path to your low-resolution videos.                                                                                                      | **Required.**                                                                       |
| `HR_INPUT_VIDEO_FOLDER`    | Path to your high-resolution videos.                                                                                                     | **Required.**                                                                       |
| `OUTPUT_BASE_FOLDER`       | Where all generated folders and files will be saved.                                                                                     | **Required.**                                                                       |
| `ATTEMPT_PAN_AND_SCAN_FIX` | **Crucial.** Crops widescreen footage to match fullscreen content before alignment.                                                        | **Set to `True`** if your sources have different aspect ratios (e.g., 2.35:1 vs 4:3). |
| `ENABLE_CHROMA_UPSAMPLING` | Fixes blocky color artifacts present in most standard video encodings.                                                                   | **Keep `True`** for almost all standard-definition (SDR) videos.                    |
| `ENABLE_HDR_TONE_MAPPING`  | Fixes washed-out, grey colors when extracting frames from HDR (High Dynamic Range) video.                                                | **Set to `True` only** if your HR source is HDR (e.g., a 4K Blu-ray).               |
| `CROP_BLACK_BORDERS`       | Enables automatic cropping of black bars (letterboxing).                                                                                 | Set to `True` if your videos have hardcoded black bars.                             |
| `CROP_WHITE_BORDERS`       | Enables automatic cropping of white or very light-colored bars.                                                                          | Set to `True` if your videos have hardcoded white bars.                             |
| `MATCH_THRESHOLD`          | Similarity score (0.0 to 1.0) needed to consider two frames a match.                                                                     | `0.65` is a good start. Lower it for difficult content, raise it for more accuracy. |
| `PHASH_SIMILARITY_THRESHOLD` | How different two frames must be to be kept. Lower is stricter.                                                                          | `4` is a good default. Set to `-1` to disable this filtering stage.                 |
| `ENABLE_SISR_FILTERING`    | Enables the final dataset curation stage using CLIP to select a diverse, complex set of images.                                          | Set to `True` to create a final, high-quality training set from the aligned pairs.  |

## License
This project is licensed under the MIT License.

## Acknowledgements
*   The Anthropic team for Claude.
*   The Google AI team for AI Studio.
*   The Enhance Everything Discord community for inspiration and discussion.
*   The [neosr](https://github.com/neosr-project/neosr) project for concepts in super-resolution.
```
