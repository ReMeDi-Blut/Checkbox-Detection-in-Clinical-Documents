import cv2
import numpy as np
import os
from ultralytics import YOLO
import pymupdf  # PyMuPDF for PDF processing
from sklearn.cluster import DBSCAN
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import login

###########################
# CONFIGURABLE PARAMETERS #
###########################

HF_TOKEN = ""

login(token=HF_TOKEN)

# Model and file paths
MODEL_PATH = "./models/detector-model.pt"  # Path to your YOLO detection model
SCAN_PATH = "./scan.pdf"                   # Input file (PDF or image) to process
OUTPUT_DIR = "./output"                    # Directory where cropped cluster images will be saved

# Detection parameters
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.2

# PDF conversion parameters
PDF_DPI = 200   # Adjust DPI for conversion quality/resolution

# Clustering parameters (in pixels)
CLUSTER_EPS_VERT = 60    # Maximum vertical distance (in pixels) for clustering
CLUSTER_EPS_HOR = 1000    # Maximum horizontal distance (in pixels) for clustering
CLUSTER_MIN_SAMPLES = 1   # Minimum samples for a cluster

# Padding around cluster bounding box (in pixels)
CLUSTER_PADDING = 50

# --- Pixtral (vLLM) Parameters ---
PIXTRAL_MODEL_NAME = "mistralai/Pixtral-12B-2409"
SAMPLING_PARAMS = SamplingParams(max_tokens=2048)
PROMPT = ""

#####################
# UTILITY FUNCTIONS #
#####################

def is_nested(outer, inner):
    """
    Check if the inner box is completely within the outer box.
    Both boxes are represented as (x1, y1, x2, y2).
    """
    return (outer[0] <= inner[0] and outer[1] <= inner[1] and
            outer[2] >= inner[2] and outer[3] >= inner[3])

def convert_pdf_to_images(pdf_path, dpi=PDF_DPI):
    """
    Convert each page of a PDF to an OpenCV (BGR) image using PyMuPDF.
    """
    doc = pymupdf.open(pdf_path)
    images = []
    zoom = dpi / 72  # PDFs are typically 72 DPI by default
    mat = pymupdf.Matrix(zoom, zoom)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        # Create a NumPy array from the pixmap's samples.
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # Convert the image to BGR color space as required by OpenCV.
        if pix.n == 4:  # RGBA format
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB format
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images

def load_input_images(input_path):
    """
    Load images from the input path.
    If the input is a PDF, convert its pages to images.
    Otherwise, assume it's an image file and load it with cv2.imread.
    Returns a list of images.
    """
    ext = os.path.splitext(input_path)[1].lower()
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    if ext == ".pdf":
        print(f"Converting PDF {input_path} to images...")
        return convert_pdf_to_images(input_path)
    elif ext in image_extensions:
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Unable to load image: {input_path}")
            return []
        return [img]
    else:
        print(f"Unsupported file extension: {ext}")
        return []

def detect_checkboxes(image):
    """
    Run detection on an image and return a list of filtered bounding boxes.
    Each box is a tuple: (x1, y1, x2, y2). Nested boxes are removed.
    """
    results = DETECTION_MODEL.predict(source=image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return []
    
    # Filter out nested boxes.
    filtered_boxes = []
    for i, box in enumerate(boxes):
        nested = False
        for j, other_box in enumerate(boxes):
            if i != j and is_nested(other_box.xyxy[0], box.xyxy[0]):
                nested = True
                break
        if not nested:
            filtered_boxes.append(box)
    
    # Convert each box to an integer tuple.
    bbox_list = []
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox_list.append((x1, y1, x2, y2))
    return bbox_list

def cluster_boxes(boxes, min_samples=CLUSTER_MIN_SAMPLES):
    """
    Given a list of boxes (each as (x1, y1, x2, y2)), cluster them based on the centers.
    Uses a custom distance metric that applies separate horizontal and vertical eps values.
    Returns a dictionary: { cluster_label: [box1, box2, ...] }.
    """
    if not boxes:
        return {}
    
    # Compute the center (cx, cy) of each box.
    centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for (x1, y1, x2, y2) in boxes])
    
    def custom_metric(p, q):
        """
        Compute a normalized distance between two points.
        Two points are considered neighbors if:
            abs(dx) < CLUSTER_EPS_HOR  and  abs(dy) < CLUSTER_EPS_VERT.
        The metric normalizes each distance and returns the maximum.
        """
        dx = abs(p[0] - q[0])
        dy = abs(p[1] - q[1])
        return max(dx / CLUSTER_EPS_HOR, dy / CLUSTER_EPS_VERT)
    
    # Setting eps=1 means that two points are neighbors if custom_metric(p, q) < 1.
    clustering = DBSCAN(eps=1, min_samples=min_samples, metric=custom_metric).fit(centers)
    labels = clustering.labels_
    
    clusters = {}
    for label, box in zip(labels, boxes):
        clusters.setdefault(label, []).append(box)
    return clusters

def get_cluster_bbox(cluster_boxes, padding=CLUSTER_PADDING):
    """
    Given a list of boxes for a cluster, compute a bounding box that encloses them all,
    with optional extra padding.
    """
    xs = []
    ys = []
    for (x1, y1, x2, y2) in cluster_boxes:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    x1 = max(min(xs) - padding, 0)
    y1 = max(min(ys) - padding, 0)
    x2 = max(xs) + padding
    y2 = max(ys) + padding
    return (x1, y1, x2, y2)

#####################
# MAIN PROCESS FLOW #
#####################

if __name__ == "__main__":
    # Load the detection model.
    DETECTION_MODEL = YOLO(MODEL_PATH)
    
    # Create the output directory if it does not exist.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load input images (whether from a PDF or an image file).
    images = load_input_images(SCAN_PATH)
    if not images:
        print("No images to process.")
        exit(1)
    
    # Process each image.
    for page_index, image in enumerate(images, start=1):
        print(f"Processing image/page {page_index}...")
        boxes = detect_checkboxes(image)
        if not boxes:
            print(f"  No checkboxes found in image/page {page_index}.")
            continue

        # Cluster the detected checkboxes using separate vertical and horizontal eps values.
        clusters = cluster_boxes(boxes)
        for cluster_id, box_list in clusters.items():
            bbox = get_cluster_bbox(box_list)
            x1, y1, x2, y2 = bbox

            # Modify: Use the full width of the original image.
            x1 = 0
            x2 = image.shape[1]

            # Crop the area from the image (full width, only the vertical region changes).
            cropped_area = image[y1:y2, x1:x2]
            output_filename = f"page_{page_index}_cluster_{cluster_id}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, cropped_area)
            print(f"  Saved cluster {cluster_id} from image/page {page_index} to {output_path}")

    # Initialize the Pixtral model via vLLM.
    llm = LLM(model=PIXTRAL_MODEL_NAME, tokenizer_mode="mistral")

    print("\nProcessing output images with Pixtral for descriptions:")
    # Process each output image.
    for filename in os.listdir(OUTPUT_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            # Construct the image URL based on your OUTPUT_BASE_URL.
            image_url = f"{filename}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
            outputs = llm.chat(messages, sampling_params=SAMPLING_PARAMS)
            # Retrieve and print the description.
            description = outputs[0].outputs[0].text
            print(f"Image '{filename}' description: {description}")
