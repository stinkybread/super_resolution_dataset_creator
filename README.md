The space currently hosts two Python scripts
1 - Image-Pairer
2 - Visdistinct

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
- FFmpeg
- ImgAlign tool
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
scene_threshold = 0.25     # Scene change detection sensitivity for ffmpeg
similarity_threshold = 4    # Threshold for filtering similar frames
img_align_scale = 2        # Scale factor for alignment
begin_time = "00:00:00"  #Set where in the video to begin extracting frames from. If it is a TV show, and are extracting frames from multiple episodes, you may want to skip the first 90 seconds of the intro- so you would use "00:01:30"
end_time = "00:00:00"  #Set where in the video to stop extracting frames from. If it is a TV show, and are extracting frames from multiple episodes, you may want to skip the last 90 seconds of the ending- so you would use "00:20:00"
scene_threshold = 0.25  #Set how different should an image be to its preeceding frame for frame extraction
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
