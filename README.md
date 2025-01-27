## Acknowledgements for Code & Inspiration
1. Anthorpic's Calude (https://claude.ai/new)
2. Enhance Everything Discord
3. neosr - https://github.com/neosr-project/neosr

-------------------------------

The space currently hosts two Python scripts
1 - Image-Pairer
2 - Visdistinct
3 - Viscomplex

# Image-Pairer

A robust Python tool for extracting, matching, and aligning frames from low-resolution (LR) and high-resolution (HR) video pairs. This tool is particularly useful for creating training datasets for super-resolution models or video quality enhancement research.

## Features

- GPU-accelerated frame processing with CUDA support (automatic fallback to CPU)
- Intelligent scene detection and frame extraction using FFmpeg
- Multi-threaded image matching and processing
- Similarity-based filtering to remove redundant frames
- Progress tracking with JSON-based checkpointing
- Precise image alignment using ImgAlign
- Support for multiple video formats (AVI, MP4, MKV)

## Prerequisites

### Required Software
- Python 3.6+
- FFmpeg - https://www.ffmpeg.org/ - Needs to be in system PATH
- ImgAlign tool - https://github.com/sonic41592/ImgAlign - Needs to be in system PATH
- CUDA toolkit (optional, for GPU acceleration)

### Python Dependencies
```bash
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install Pillow
pip install imagehash
pip install tqdm
```

### System Requirements
- Minimum 8GB RAM (16GB recommended)
- NVIDIA GPU with CUDA support (optional)
- Sufficient storage space for extracted frames

## Directory Structure
```
└── BASE_DIRECTORY
    ├── LR              # Input low-resolution videos
    ├── HR              # Input high-resolution videos
    └── Output
        ├── EXTRACTED   # Extracted frames
        ├── MATCHED     # Matched frame pairs
        └── ALIGNED     # Final aligned images
```

## Configuration

Key parameters in the script that you may want to adjust:

```python
match_threshold = 0.3        # Threshold for frame matching
similarity_threshold = 4    # Threshold for filtering similar frames. Used after matching and before aligning to optimize workflow.
img_align_scale = 2        # Scale factor for alignment
begin_time = "00:00:00"  #Set where in the video to begin extracting frames from. If it is a TV show, and are extracting frames from multiple episodes, you may want to skip the first 90 seconds of the intro- so you would use "00:01:30"
end_time = "00:00:00"  #Set where in the video to stop extracting frames from. If it is a TV show, and are extracting frames from multiple episodes, you may want to skip the last 90 seconds of the ending- so you would use "00:20:00"
scene_threshold = 0.25  #Scene change detection sensitivity for ffmpeg. Set how different should an image be to its preeceding frame for frame extraction
similarity_threshold = 4  #Basic check to remove similar images before image matching
lr_width, lr_height = 852, 480  #Set the resolution to set the video when extracting frames
hr_width, hr_height = 1980, 1080  #Set the resolution to set the video when extracting frames
```

## Usage

1. Place your LR and HR video pairs in their respective folders
2. Configure the paths in the script:
```python
lr_input_video_folder = "path/to/LR"
hr_input_video_folder = "path/to/HR"
output_base_folder = "path/to/output"
```

3. Run the script:
```bash
python image-pairer.py
```

## Processing Stages

1. **Preprocessing**: Extracts frames from video pairs using scene detection
2. **Matching**: Pairs corresponding frames between LR and HR versions
3. **Filtering**: Removes similar frames to reduce redundancy
4. **Alignment**: Precisely aligns LR-HR pairs for training

## Notes

- Video filenames must match between LR and HR folders. Use mkvtoolnix to ensure all files are mkv files for ease of use.
- Progress is automatically saved and can be resumed if interrupted
- GPU acceleration is automatically used if available
- The script includes extensive error handling and logging

## Performance Considerations

- Frame extraction speed depends on video length and scene threshold
- GPU acceleration significantly improves matching speed
- Processing time varies based on video resolution and length
- Memory usage scales with frame size and count

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Uses FFmpeg for video processing
- Uses ImgAlign for precise image alignment
- OpenCV for image processing
- CUDA for GPU acceleration

# Visdistinct

visdistinct helps create visually distinct dataset from a group of images. 
Ideally used after your super resolution dataset is created to ensure the model has the right images to be trained on.
Use the following command to ensure you have everything you need to get up and running 

## Prerequisites
```pip install torch torchvision transformers Pillow tqdm py-cpuinfo psutil```

## Features
This script provides a robust solution with several key features:
1. Uses CLIP model for state-of-the-art visual feature extraction
2. Comprehensive error handling and logging
3. Support for multiple image formats
4. Verification of image validity before processing
5. Customizable minimum distance threshold
6. Automatic output folder creation with timestamp
7. Progress logging to both console and file
8. Capability to detect your CPU and optimize accordingly to create threads and workers.

## How to run 
```python visdistinct.py "path/to/image_folder" "path/to/new_image_folder" similarity_threshold```

Example -

```python visdistinct.py "input_folder" "output_folder" 0.1```

# Viscomplex

A Python tool that selects optimal images for super-resolution model training by analyzing complexity and maintaining visual diversity.

## Features

1. Multi-metric complexity analysis (entropy, edge density, texture, sharpness)
2. CLIP-based similarity filtering for dataset diversity
3. Brightness Threshold for image selection
   a. 200: Filters very bright images
   b. 180: More aggressive filtering
   c. 220: Less aggressive filtering
4. GPU acceleration when available
5. System resource optimization
6. Checkpoint system for long runs

## Usage
bashCopypython python sisr_image_selector.py "path/to/images" --complexity_threshold 0.4 --min_distance 0.15 --max_brightness 200

## Requirements

1. torch
2. transformers
3. opencv-python
4. pillow
5. scipy
6. tqdm
7. psutil

## How It Works

This script identifies high-quality images for super-resolution training through a two-stage process:
Complexity Analysis:

Uses multiple metrics: entropy (information density), edge density (detail level), local variance (texture), and Laplacian variance (sharpness)
These metrics were chosen over alternatives (like BRISQUE or NIQE) because they:

1. Directly measure features relevant to super-resolution (edges, textures, details)
2. Are computationally efficient
3. Don't require pre-trained models

Weights: entropy (0.4), edge density (0.4), sharpness (0.2)
Higher weights on entropy/edges prioritize detailed images with clear structures

Similarity Filtering:

Uses CLIP embeddings for semantic similarity
Process:

1. Sort by complexity score
2. Keep images above complexity threshold
3. Calculate pairwise cosine distances between CLIP embeddings
4. Start with highest complexity image
5. Iteratively add images that are sufficiently different (min_distance)


CLIP chosen because it:

1. Captures both semantic and visual similarities
2. More robust than pixel-based or perceptual metrics
3. GPU-accelerated when available

The two-stage approach ensures both image quality and dataset diversity, critical for super-resolution model training.

Ideal for creating high-quality, diverse super-resolution training datasets.
