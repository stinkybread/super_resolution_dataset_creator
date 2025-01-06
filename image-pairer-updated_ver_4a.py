import cv2
import os
import shutil
import subprocess
from tqdm import tqdm
import glob
import concurrent.futures
import numpy as np
import json
import imagehash
from PIL import Image
from collections import defaultdict

# Check CUDA availability and print OpenCV version
print(f"OpenCV version: {cv2.__version__}")
print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")

use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

if use_gpu:
    print("CUDA-capable GPU detected. Using GPU acceleration.")
    
    # Print CUDA device info
    cuda_device = cv2.cuda.getDevice()
    print(f"CUDA device being used: {cuda_device}")
    print(f"CUDA device name: {cv2.cuda.getDeviceName(cuda_device)}")
else:
    print("No CUDA-capable GPU detected. Using CPU.")

def update_progress(progress_file, stage, video=None, completed=False):
    progress = {}
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    
    if video:
        if stage not in progress:
            progress[stage] = {'videos': []}
        if video not in progress[stage]['videos']:
            progress[stage]['videos'].append(video)
    elif completed:
        progress[stage] = {'completed': True}
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

def get_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def extract_frames_ffmpeg(video_path, output_folder, begin_time, end_time, scene_threshold, width=None, height=None, scale=None):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Construct FFmpeg command for scene change detection and frame extraction
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
    ]
    
    # Only add time constraints if both begin_time and end_time are not "00:00:00"
    if begin_time != "00:00:00" or end_time != "00:00:00":
        ffmpeg_command.extend(["-ss", begin_time])  # Start time
        ffmpeg_command.extend(["-to", end_time])    # End time
    
    # Construct the video filter
    vf_filters = [f"select='gt(scene,{scene_threshold})'"]
    
    # Add scaling if specified, otherwise don't scale
    if width and height:
        vf_filters.append(f"scale={width}:{height}")
    elif scale:
        vf_filters.append(f"scale=iw*{scale}:ih*{scale}")
    
    vf_filters.append("showinfo")
    
    ffmpeg_command.extend([
        "-vf", ",".join(vf_filters),
        "-vsync", "vfr",
        "-q:v", "2",
        f"{output_folder}/frame_%06d.png"
    ])
    
    # Run FFmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Frames extracted successfully from {os.path.basename(video_path)}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {os.path.basename(video_path)}: {e}")
        

def preprocess_videos(lr_input_folder, hr_input_folder, extracted_images_folder, begin_time, end_time, scene_threshold, 
                      lr_width=None, lr_height=None, lr_scale=None, 
                      hr_width=None, hr_height=None, hr_scale=None,
                      progress_file=None):
    progress = get_progress(progress_file)
    if 'preprocess' in progress and progress['preprocess'].get('completed'):
        print("Preprocessing already completed. Skipping...")
        return os.path.join(extracted_images_folder, "LR"), os.path.join(extracted_images_folder, "HR")

    # Create LR and HR base folders inside EXTRACTED_IMAGES folder
    lr_base_path = os.path.join(extracted_images_folder, "LR")
    hr_base_path = os.path.join(extracted_images_folder, "HR")
    os.makedirs(lr_base_path, exist_ok=True)
    os.makedirs(hr_base_path, exist_ok=True)
    
    # Define allowed video extensions
    allowed_extensions = ('.avi', '.mp4', '.mkv')
    
    # Get list of video files
    lr_video_files = [f for f in os.listdir(lr_input_folder) if f.lower().endswith(allowed_extensions)]
    hr_video_files = [f for f in os.listdir(hr_input_folder) if f.lower().endswith(allowed_extensions)]
    
    # Ensure LR and HR video lists match
    if set(lr_video_files) != set(hr_video_files):
        print("Warning: LR and HR video lists do not match. Processing only matching videos.")
        matching_videos = set(lr_video_files) & set(hr_video_files)
        lr_video_files = list(matching_videos)
        hr_video_files = list(matching_videos)
    
    for video_file in tqdm(lr_video_files, desc="Processing videos"):
        if 'preprocess' in progress and video_file in progress['preprocess'].get('videos', []):
            print(f"Skipping already processed video: {video_file}")
            continue

        video_name = os.path.splitext(video_file)[0]
        
        # Extract frames for LR
        lr_video_path = os.path.join(lr_input_folder, video_file)
        lr_output_folder = os.path.join(lr_base_path, video_name)
        extract_frames_ffmpeg(lr_video_path, lr_output_folder, begin_time, end_time, scene_threshold, lr_width, lr_height, lr_scale)
        
        # Extract frames for HR
        hr_video_path = os.path.join(hr_input_folder, video_file)
        hr_output_folder = os.path.join(hr_base_path, video_name)
        extract_frames_ffmpeg(hr_video_path, hr_output_folder, begin_time, end_time, scene_threshold, hr_width, hr_height, hr_scale)
        
        update_progress(progress_file, 'preprocess', video=video_file)
    
    update_progress(progress_file, 'preprocess', completed=True)
    return lr_base_path, hr_base_path

def process_image_pair(lr_img, hr_images, lr_folder, hr_folder, output_lr_folder, output_hr_folder, match_threshold, resize_height, resize_width, video_name):
    global use_gpu

    lr_img_path = os.path.join(lr_folder, lr_img)
    lr_frame = cv2.imread(lr_img_path)
    
    if use_gpu:
        try:
            lr_frame_gpu = cv2.cuda_GpuMat()
            lr_frame_gpu.upload(lr_frame)
            lr_frame_gpu = cv2.cuda.resize(lr_frame_gpu, (resize_width, resize_height))
            lr_gray_gpu = cv2.cuda.cvtColor(lr_frame_gpu, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            print(f"Error in GPU processing: {e}")
            print("Falling back to CPU processing.")
            use_gpu = False
    
    if not use_gpu:
        lr_frame = cv2.resize(lr_frame, (resize_width, resize_height))
        lr_gray = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2GRAY)
    
    best_match = None
    best_score = float('inf')
    
    for hr_img in hr_images:
        hr_img_path = os.path.join(hr_folder, hr_img)
        hr_frame = cv2.imread(hr_img_path)
        
        if use_gpu:
            try:
                hr_frame_gpu = cv2.cuda_GpuMat()
                hr_frame_gpu.upload(hr_frame)
                hr_frame_gpu = cv2.cuda.resize(hr_frame_gpu, (resize_width, resize_height))
                hr_gray_gpu = cv2.cuda.cvtColor(hr_frame_gpu, cv2.COLOR_BGR2GRAY)
                
                result_gpu = cv2.cuda.createTemplateMatching(lr_gray_gpu, cv2.TM_SQDIFF_NORMED)
                result = result_gpu.match(hr_gray_gpu)
                score = result.minVal()
            except cv2.error as e:
                print(f"Error in GPU processing: {e}")
                print("Falling back to CPU processing for this iteration.")
                use_gpu = False
        
        if not use_gpu:
            hr_frame = cv2.resize(hr_frame, (resize_width, resize_height))
            hr_gray = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(lr_gray, hr_gray, cv2.TM_SQDIFF_NORMED)
            score = np.min(result)
        
        if score < best_score:
            best_score = score
            best_match = hr_img
    
    if best_score < match_threshold:
        lr_src = os.path.join(lr_folder, lr_img)
        hr_src = os.path.join(hr_folder, best_match)
        lr_dst = os.path.join(output_lr_folder, f"{video_name}_{os.path.splitext(lr_img)[0]}.png")
        hr_dst = os.path.join(output_hr_folder, f"{video_name}_{os.path.splitext(lr_img)[0]}.png")
        shutil.copy(lr_src, lr_dst)
        shutil.copy(hr_src, hr_dst)
        return 1
    return 0

def process_folder_pair(lr_base_path, hr_base_path, output_lr_base_path, output_hr_base_path, match_threshold, resize_height, resize_width, distance_modifier, progress_file):
    progress = get_progress(progress_file)
    if 'match' in progress and progress['match'].get('completed'):
        print("Matching already completed. Skipping...")
        return

    lr_folders = [f for f in os.listdir(lr_base_path) if os.path.isdir(os.path.join(lr_base_path, f))]
    hr_folders = [f for f in os.listdir(hr_base_path) if os.path.isdir(os.path.join(hr_base_path, f))]

    for folder in lr_folders:
        if folder in hr_folders:
            if 'match' in progress and folder in progress['match'].get('videos', []):
                print(f"Skipping already matched folder: {folder}")
                continue

            lr_folder_path = os.path.join(lr_base_path, folder)
            hr_folder_path = os.path.join(hr_base_path, folder)
            
            lr_images = [f for f in os.listdir(lr_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            hr_images = [f for f in os.listdir(hr_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            lr_images.sort()
            hr_images.sort()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for lr_img in lr_images:
                    future = executor.submit(
                        process_image_pair,
                        lr_img,
                        hr_images,
                        lr_folder_path,
                        hr_folder_path,
                        output_lr_base_path,
                        output_hr_base_path,
                        match_threshold,
                        resize_height,
                        resize_width,
                        folder
                    )
                    futures.append(future)
                
                pair_count = sum(future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Matching and saving images for {folder}"))
            
            print(f"Total pairs created for {folder}: {pair_count}")
            update_progress(progress_file, 'match', video=folder)

    update_progress(progress_file, 'match', completed=True)

def filter_similar_images(matched_lr_path, matched_hr_path, similarity_threshold=4, progress_file=None):
    progress = get_progress(progress_file)
    if 'filter' in progress and progress['filter'].get('completed'):
        print("Filtering already completed. Skipping...")
        return 0

    print("Starting similarity detection and filtering...")
    
    lr_images = [f for f in os.listdir(matched_lr_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    video_groups = defaultdict(list)
    for img in lr_images:
        video_name = img.split('_')[0]
        video_groups[video_name].append(img)
    
    total_removed = 0
    
    for video_name, images in tqdm(video_groups.items(), desc="Filtering similar images"):
        if 'filter' in progress and video_name in progress['filter'].get('videos', []):
            print(f"Skipping already filtered video: {video_name}")
            continue

        hashes = {}
        to_remove = set()
        
        for img in images:
            lr_path = os.path.join(matched_lr_path, img)
            hr_path = os.path.join(matched_hr_path, img)
            
            with Image.open(lr_path) as img_file:
                img_hash = imagehash.phash(img_file)
            
            for existing_img, existing_hash in hashes.items():
                if img_hash - existing_hash < similarity_threshold:
                    if os.path.getsize(lr_path) > os.path.getsize(os.path.join(matched_lr_path, existing_img)):
                        to_remove.add(existing_img)
                        hashes[img] = img_hash
                        del hashes[existing_img]
                    else:
                        to_remove.add(img)
                    break
            else:
                hashes[img] = img_hash
        
        for img in to_remove:
            os.remove(os.path.join(matched_lr_path, img))
            os.remove(os.path.join(matched_hr_path, img))
        
        total_removed += len(to_remove)
        update_progress(progress_file, 'filter', video=video_name)
    
    print(f"Similarity filtering completed. Removed {total_removed} similar images.")
    update_progress(progress_file, 'filter', completed=True)
    return total_removed

def align_images(output_lr_base_path, output_hr_base_path, aligned_output_base_path, img_align_scale, progress_file=None):
    progress = get_progress(progress_file)
    if 'align' in progress and progress['align'].get('completed'):
        print("Alignment already completed. Skipping...")
        return

    if os.path.exists(aligned_output_base_path):
        shutil.rmtree(aligned_output_base_path)
    os.makedirs(aligned_output_base_path)

    lr_images = [f for f in os.listdir(output_lr_base_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Starting image alignment process for {len(lr_images)} image pairs...")

    video_groups = defaultdict(list)
    for img in lr_images:
        video_name = img.split('_')[0]
        video_groups[video_name].append(img)

    for video_name, images in tqdm(video_groups.items(), desc="Aligning videos", unit="video"):
        if 'align' in progress and video_name in progress['align'].get('videos', []):
            print(f"Skipping already aligned video: {video_name}")
            continue

        print(f"\nAligning images for {video_name}...")

        temp_lr_path = os.path.join(aligned_output_base_path, f"temp_lr_{video_name}")
        temp_hr_path = os.path.join(aligned_output_base_path, f"temp_hr_{video_name}")
        temp_aligned_path = os.path.join(aligned_output_base_path, f"temp_aligned_{video_name}")
        os.makedirs(temp_lr_path, exist_ok=True)
        os.makedirs(temp_hr_path, exist_ok=True)
        os.makedirs(temp_aligned_path, exist_ok=True)

        for img in tqdm(images, desc=f"Aligning pairs for {video_name}", unit="pair"):
            shutil.copy(os.path.join(output_lr_base_path, img), os.path.join(temp_lr_path, img))
            shutil.copy(os.path.join(output_hr_base_path, img), os.path.join(temp_hr_path, img))

            img_align_command = [
                "ImgAlign",
                "-s", str(img_align_scale),
                "-m", "0",
                "-g", temp_hr_path,
                "-l", temp_lr_path,
                "-c",
                "-i", "-1",
                "-j",
                "-ai",
                "-o", temp_aligned_path
            ]

            try:
                process = subprocess.Popen(img_align_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                process.wait()

                if process.returncode != 0:
                    print(f"Error aligning image pair {img}. Return code: {process.returncode}")
            except subprocess.CalledProcessError as e:
                print(f"Error aligning image pair {img}: {e}")

            for file in os.listdir(temp_lr_path):
                os.remove(os.path.join(temp_lr_path, file))
            for file in os.listdir(temp_hr_path):
                os.remove(os.path.join(temp_hr_path, file))

        for folder in ['HR', 'LR', 'Overlay']:
            src_folder = os.path.join(temp_aligned_path, 'Output', folder)
            dst_folder = os.path.join(aligned_output_base_path, folder)
            os.makedirs(dst_folder, exist_ok=True)
            if os.path.exists(src_folder):
                for file in os.listdir(src_folder):
                    shutil.move(os.path.join(src_folder, file), os.path.join(dst_folder, file))

        artifacts_file = os.path.join(temp_aligned_path, 'Output', "Artifacts.txt")
        if os.path.exists(artifacts_file):
            print(f"\nFound Artifacts.txt for {video_name}. Cleaning up...")
            with open(artifacts_file, 'r') as f:
                artifacts = f.read().splitlines()
            
            artifacts_removed = False
            for folder in ['HR', 'LR', 'Overlay']:
                folder_path = os.path.join(aligned_output_base_path, folder)
                if os.path.exists(folder_path):
                    for artifact in artifacts:
                        matching_files = glob.glob(os.path.join(folder_path, f"{artifact}.*"))
                        for file_path in matching_files:
                            try:
                                os.remove(file_path)
                                print(f"Removed {file_path}")
                                artifacts_removed = True
                            except OSError as e:
                                print(f"Error removing {file_path}: {e}")
            
            if not artifacts_removed:
                print(f"No artifacts found for {video_name}")

        update_progress(progress_file, 'align', video=video_name)

        shutil.rmtree(temp_lr_path)
        shutil.rmtree(temp_hr_path)
        shutil.rmtree(temp_aligned_path)

    update_progress(progress_file, 'align', completed=True)
    print("\nImage alignment process completed.")

if __name__ == "__main__":
    # Configuration options
    match_threshold = 0.3
    resize_height = 720
    resize_width = 1280
    distance_modifier = 0.75
    img_align_scale = 2
    begin_time = "00:00:00"
    end_time = "00:00:00"
    scene_threshold = 0.25
    similarity_threshold = 4

    lr_width, lr_height = 852, 480
    lr_scale = None
    hr_width, hr_height = None, None
    hr_scale = None

    # Set these paths according to your directory structure
    lr_input_video_folder = "E:\\MODEL VID\\LR"
    hr_input_video_folder = "E:\\MODEL VID\\HR"
    output_base_folder = "E:\\MODEL VID\\Output"
    extracted_images_folder = os.path.join(output_base_folder, "EXTRACTED")
    output_lr_base_path = os.path.join(output_base_folder, "MATCHED", "LR")
    output_hr_base_path = os.path.join(output_base_folder, "MATCHED", "HR")
    aligned_output_base_path = os.path.join(output_base_folder, "ALIGNED")
    progress_file = os.path.join(output_base_folder, "progress.json")
     
    # Create output directories if they don't exist
    os.makedirs(extracted_images_folder, exist_ok=True)
    os.makedirs(output_lr_base_path, exist_ok=True)
    os.makedirs(output_hr_base_path, exist_ok=True)

    # Preprocess videos and get LR and HR base paths
    lr_base_path, hr_base_path = preprocess_videos(
        lr_input_video_folder, hr_input_video_folder, extracted_images_folder, 
        begin_time, end_time, scene_threshold, 
        lr_width, lr_height, lr_scale, 
        hr_width, hr_height, hr_scale,
        progress_file
    )

    # Process folder pairs
    process_folder_pair(lr_base_path, hr_base_path, output_lr_base_path, output_hr_base_path, 
                        match_threshold, resize_height, resize_width, distance_modifier, progress_file)

    # Filter similar images
    filtered_count = filter_similar_images(output_lr_base_path, output_hr_base_path, similarity_threshold, progress_file)
    print(f"Filtered {filtered_count} similar images.")

    # Align images
    align_images(output_lr_base_path, output_hr_base_path, aligned_output_base_path, img_align_scale, progress_file)

    print("Image processing and alignment completed.")