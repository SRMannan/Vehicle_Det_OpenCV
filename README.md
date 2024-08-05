# Vehicle Detection with OpenCV

## Overview

This project leverages OpenCV to create a comprehensive vehicle detection system capable of identifying vehicles, tracking their speed, and recognizing number plates from video feeds or images. Designed to enhance traffic management and monitoring, this system provides real-time vehicle insights.

## Features

- **Vehicle Detection**: Accurately detects and outlines vehicles within the input video stream or image using advanced computer vision techniques.
- **Speed Tracking**: Estimates the speed of detected vehicles by analyzing their movement across frames in video streams, offering valuable data for traffic analysis.
- **Number Plate Recognition**: Extracts and recognizes number plates from detected vehicles, utilizing Optical Character Recognition (OCR) to facilitate automated vehicle identification.

## Installation

To run this project, ensure you have the following prerequisites installed:

- Python 3.x
- OpenCV
- EasyOcr (for OCR)

You can install the required Python libraries using pip:

```bash
pip install opencv-python
pip install easyocr
