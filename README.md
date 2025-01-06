The space currently hosts two Python scripts

2 - Visdistinct 

visdistinct helps create visually distinct dataset from a group of images. 
Ideally used after your super resolution dataset is created to ensure the model has the right images to be trained on.
Use the following command to ensure you have everything you need to get up and running 

```pip install torch torchvision transformers Pillow tqdm py-cpuinfo psutil```


This script provides a robust solution with several key features:
1. Uses CLIP model for state-of-the-art visual feature extraction
2. Comprehensive error handling and logging
3. Support for multiple image formats
4. Verification of image validity before processing
5. Customizable minimum distance threshold
6. Automatic output folder creation with timestamp
7. Progress logging to both console and file
8. Capability to detect your CPU and optimize accordingly to create threads and workers.
