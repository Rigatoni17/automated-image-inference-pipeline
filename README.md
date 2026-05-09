# Automated Image Ingestion & Inference Pipeline (MLOps)

## Project Overview
This project demonstrates an end-to-end Machine Learning Operations (MLOps) pipeline designed to ingest, process, and classify complex unstructured data (images) using Convolutional Neural Networks (CNNs). 

While the underlying model focuses on botanical classification, the primary engineering challenge was designing a robust data pipeline capable of handling massive variations in data quality, lighting, and physical occlusions.

## Role & Contributions
This project was initially developed as a collaborative academic requirement. As the lead engineer on this project, my specific contributions included:
* Architecting the end-to-end data ingestion and transformation pipeline.
* Engineering the automated data augmentation workflows to simulate real-world occlusions.
* Integrating the fine-tuned ResNet18 architecture for final processing.

## Data Engineering & Pipeline Architecture

### 1. Data Extraction & Ingestion
* Ingested a high-volume dataset of over 8,000 unstructured image files.
* Partitioned data into strict 80/10/10 splits to prevent data leakage during the processing phase.

### 2. Data Transformation (Augmentation & Profiling)
* **Data Quality Control:** Implemented aggressive, automated data augmentation using `torchvision` to simulate real-world data corruption.
* **Occlusion Simulation:** Engineered a 'Random Erasing' transformation pipeline to drop rectangular pixel blocks, mimicking physical occlusions to stress-test the model's feature extraction capabilities.

### 3. Model Architecture & Inference
* Deployed a fine-tuned **ResNet18** architecture, utilizing residual skip-connections to solve vanishing gradient issues during deep network processing.
* Compared spatial-aware convolutions against flattened 1D baseline architectures to validate the necessity of structural data preservation.

## Repository Contents
* `automated_inference_pipeline.ipynb`: The core ETL and model training pipeline.
* `team_25_project_report.pdf`: Comprehensive technical documentation, architectural breakdown, and quantitative analysis.
