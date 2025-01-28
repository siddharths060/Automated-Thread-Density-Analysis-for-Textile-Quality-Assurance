# Fabric Thread Detection Using Vision Transformer (ViT)

## Overview
This project aims to detect and count the warp and weft threads in fabric images using a **Vision Transformer (ViT)** model. The model is trained on labeled fabric images and performs **segmentation**, **thread detection**, and **counting**. The main goal is to identify the number of horizontal (warp) and vertical (weft) threads in a given fabric, which can be useful for textile quality control and automation.

The Vision Transformer (ViT) is a transformer-based model originally designed for image classification, which has been adapted here for fabric thread analysis.

## Key Features:
- **Image Preprocessing**: Includes noise reduction and adaptive thresholding for effective segmentation.
- **Fabric Segmentation**: Identifies and extracts the fabric region from the background.
- **Thread Detection**: Uses edge detection (Sobel operator, Canny edge) and line transformation (Hough transform) to detect and classify individual warp and weft threads.
- **Warp & Weft Count**: Computes and displays the number of warp and weft threads in a specific region (1 square inch) of the fabric.

## Project Structure

