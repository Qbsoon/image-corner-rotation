from ultralytics import YOLO
import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Run model inference on images.')
parser.add_argument('input_dir', type=str, help='Directory containing images for inference.')
parser.add_argument('--save_dir', type=str, default='output/', help='Directory to save inference results.')
parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help='Verbosity level: 0 for silent, 1 for general output, 2 for detailed output.')
parser.add_argument('--model', type=str, default='best.pt', help='Path to the YOLO model weights.')
args = parser.parse_args()

input_dir = args.input_dir
save_dir = args.save_dir
verbose = args.verbose
model_path = args.model

if input_dir is None:
	raise ValueError("Input directory must be specified.")

if not os.path.isdir(input_dir):
    raise ValueError(f"Input directory not found: {input_dir}")

try:
	os.makedirs(save_dir, exist_ok=True)
except Exception as e:
	raise ValueError(f"Could not create or check save directory: {save_dir}. Error: {e}")

if not os.path.isfile(model_path):
	raise ValueError(f"Model file not found: {model_path}")

model = YOLO(model_path)

image_paths = []
for root, dirs, files in os.walk(input_dir):
	for file in files:
		if file.lower().endswith(('.jpg', '.jpeg', '.png')):
			image_paths.append(os.path.join(root, file))

if len(image_paths) == 0:
	raise ValueError("No images found in the specified input directory.")

def crop_image_from_prediction(card, prediction):
		if len(prediction.keypoints.xy) == 0:
			if verbose == 2:
				print("No keypoints found in the prediction.")
			return None

		keypoints = prediction.keypoints.xy[0].cpu().numpy()
		if keypoints.shape[0] != 4:
			if verbose == 2:
				print(f"Expected 4 points, but found {keypoints.shape[0]}")
			return None

		keypoints = keypoints.astype("float32")

		width = int(np.linalg.norm(keypoints[1] - keypoints[0]))
		height = int(np.linalg.norm(keypoints[3] - keypoints[0]))

		dst_pts = np.array([
			[0, 0],
			[width - 1, 0],
			[width - 1, height - 1],
			[0, height - 1]
		], dtype="float32")

		M = cv2.getPerspectiveTransform(keypoints, dst_pts)
		warped = cv2.warpPerspective(card, M, (width, height))

		return warped

progress = tqdm(total=len(image_paths), desc="Processing images", unit="image")
for img_path in image_paths:
	img = cv2.imread(img_path)
	height, width = img.shape[:2]
	card = np.ones((height+100, width+100, 3), dtype=np.uint8) * 255
	card[50:50+height, 50:50+width] = img
	cv2.imwrite(f"debug/{os.path.basename(img_path)}", card)
	if img is None and verbose>=1:
		print(f"Could not read image: {img_path}")
		progress.update(1)
		continue
	results = model.predict(
		source=img,
		save=False,
		conf=0.25,
		imgsz=416,
		verbose=verbose==2
	)
	rotated = crop_image_from_prediction(card, results[0])
	if rotated is None and verbose>=1:
		print(f"Could not rotate image: {img_path}")
		progress.update(1)
		continue
	try:
		cv2.imwrite(f"{save_dir}/{os.path.basename(img_path)}", rotated)
	except Exception as e:
		if verbose>=1:
			print(f"Error saving image {img_path} to {save_dir}: {e}")
		progress.update(1)
		continue
	if verbose==2:
		print(f"Processed {img_path}, saved to {save_dir}/{os.path.basename(img_path)}")
	progress.update(1)