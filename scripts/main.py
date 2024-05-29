import cv2
import os
import argparse
from generate_image_features import generate_metadata, save_metadata
from model import generate_context_and_inferences

def video_to_images(video_path, output_folder, fps):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % fps == 0:
            image_path = os.path.join(output_folder, f"frame{count}.jpg")
            cv2.imwrite(image_path, image)
        success, image = vidcap.read()
        count += 1

def process_images(image_folder):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")]
    for image_path in image_paths:
        metadata = generate_metadata(image_path)
        save_metadata(image_path, metadata)

def main():
    parser = argparse.ArgumentParser(description="Process video or image")
    parser.add_argument("input_path", type=str, help="Path to the input video or image folder")
    parser.add_argument("output_folder", type=str, help="Path to the output folder")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract from video")
    args = parser.parse_args()

    if args.input_path.endswith(".mp4") or args.input_path.endswith(".avi"):
        video_to_images(args.input_path, args.output_folder, args.fps)
    else:
        process_images(args.input_path)

    image_folder = args.output_folder
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")]

    for image_path in image_paths:
        # Here you would load the features and use your model
        # For demonstration, we assume you have features
        features = None  # Replace with actual feature loading
        context_inferences = generate_context_and_inferences(features)
        print(f"Context and Inferences for {image_path}: {context_inferences}")

if __name__ == "__main__":
    main()
