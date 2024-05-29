# Visual Inference Project

## Project Overview

The Visual Inference Project is designed to process videos by splitting them into image sequences, generating metadata using object detection, and then generating context and inferences from these images using a custom model.

## File Descriptions

- `generate_image_features.py`: Extracts object detection features from images and saves the metadata.
- `model.py`: Defines the model architecture for generating context and inferences from image features.
- `train.py`: Trains the model using the provided training data.
- `main.py`: Orchestrates the entire workflow: splitting video into images, extracting features, and generating inferences.
- `eval.py`: Evaluates the model using a test dataset.
- `test.py`: Tests the trained model on a set of images and generates context and inferences.

## Requirements

- Python 3.6 or higher
- PyTorch 1.8.0
- Torchvision 0.9.0
- Pillow 8.1.2
- OpenCV 4.5.1.48

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage
Generate Image Features
To generate features for a single image, run:

```bash
python generate_image_features.py <image_path>
```

## Train the Model
To train the model, update the train_data variable in train.py with your training data and run:

```bash
python train.py
```

## Process Video or Images
To process a video and generate context and inferences, run:

```bash
python main.py <input_path> <output_folder> --fps <frames_per_second>
```

Replace <input_path> with the path to the video file or image folder, <output_folder> with the path to the output folder, and <frames_per_second> with the desired frames per second to extract from the video.

## Evaluate the Model
To evaluate the model, update the test_data variable in eval.py with your test data and run:

```bash
python eval.py
```

## Test the Model
To test the trained model on a set of images and generate context and inferences, run:

```bash
python test.py <model_path> <image_folder> <output_folder>
```

Replace <model_path> with the path to the trained model, <image_folder> with the path to the folder containing test images, and <output_folder> with the path to the folder to save the output.

## Project Structure
```bash
visual-inference/
│
├── scripts/
│   ├── generate_image_features.py
│   ├── train.py
│   ├── main.py
│   ├── eval.py
│   ├── test.py
│   └── model.py
│
├── model/
│   
│
├── requirements.txt
└── README.md

```

## Acknowledgments
This project uses the following libraries:

PyTorch
Torchvision
Pillow
OpenCV


## License
This project is licensed under the GNU General Public License v3.0 

