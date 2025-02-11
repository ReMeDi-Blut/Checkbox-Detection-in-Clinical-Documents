# Checkbox Clustering and Description Pipeline

This repository provides a complete pipeline for detecting, clustering, and extracting checked categories from accumulations of checkboxes either from PDFs or image files. The pipeline works in two main stages:

1. **Detection and Clustering:**  
   - Uses a YOLO checkbox detection model (via ([YOLOv8]https://github.com/LynnHaDo/Checkbox-Detection)) to detect checkboxes in an input PDF or image.
   - Clusters the detected checkboxes using DBSCAN with customizable vertical and horizontal thresholds.
   - Extracts cropped cluster images that span the full width of checkbox areas.

2. **Postprocessing with Pixtral:**  
   - Uses the [Pixtral](https://huggingface.co/mistralai/Pixtral-12B-2409) model via [vLLM](https://github.com/vllm-project/vllm).
   - Instructs to extract only the checked marks to predefined categories.

## Requirements

- [OpenCV](https://opencv.org/) (`opencv-python`)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) (`pymupdf`)
- [scikit-learn](https://scikit-learn.org/stable/) (`scikit-learn`)
- [vLLM](https://github.com/vllm-project/vllm)
- [huggingface_hub](https://huggingface.co/docs/huggingface_hub/)
- [numpy](https://numpy.org/)

You can install these dependencies using pip:

```bash
pip install opencv-python ultralytics pymupdf scikit-learn vllm huggingface_hub numpy
