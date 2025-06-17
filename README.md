# Fake-Invoice-Detector
 Fake Invoice Detector using Computer Vision
# Fake Invoice Detector using Computer Vision

This is a Flask-based web application that detects whether an uploaded invoice is original or fake using computer vision and OCR (Optical Character Recognition).

## Project Description

The system analyzes the visual layout and extracted text from invoice images or PDFs. It compares the invoice against known structural patterns and expected fields to determine its authenticity.

## Features

- Upload invoice as image or PDF through a web interface
- Extract key fields using Tesseract OCR:
  - Invoice Number
  - Date
  - Vendor Name
  - Total Amount
- Analyze layout using OpenCV:
  - Count horizontal and vertical lines
  - Calculate header variance
- Determine if invoice is fake based on:
  - Missing critical fields
  - Irregular layout structure
- Display the result with extracted text and features

## Technology Stack

- Python
- Flask (Web Framework)
- OpenCV (Image Processing)
- Tesseract OCR (Text Extraction)
- HTML/CSS (Frontend)

## Setup Instructions

1. Install Python 3.8 or higher

2. Install Tesseract OCR:
