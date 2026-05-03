# ML Image Editor Pro

An advanced image processing web application built using Streamlit and OpenCV.

This app allows users to upload images and apply multiple real-time image processing techniques through an interactive UI.

---

## 🚀 Features

- Blur & Sharpen  
- Brightness & Contrast Adjustment  
- Edge Detection  
- Grayscale Conversion  
- Noise Reduction (Denoising)  
- Color Filters (Sepia, Cool, Warm, Invert)  
- Crop & Rotate  
- Face Detection  
- Super Resolution (Image Quality Enhancement)  

---

## 🗂️ Project Structure

streamlit_ml_image_preprocesing/
│
├── app.py # Main Streamlit application
├── filters.py # Image processing functions
├── utils.py # Helper functions
├── requirements.txt # Project dependencies
---

## ⚙️ Installation

### 1. Clone or Extract the Project

---

### 2. Create Virtual Environment (Recommended)

#### Activate Environment

**Windows**
venv\Scripts\activate

**Mac/Linux**
source venv/bin/activate

---

### 3. Install Dependencies
pip install -r requirements.txt

---

## ▶️ Run the Application
streamlit run app.py

Open your browser and go to:
http://localhost:8501

---

## 🧠 Tech Stack

- Python  
- Streamlit  
- OpenCV  
- NumPy  
- Pillow  

---

## 📌 How It Works

1. Upload an image  
2. Convert image to NumPy array  
3. Apply processing using OpenCV  
4. Convert back to display format  
5. Display in Streamlit  

---

## 📜 License

This project is open-source and available under the MIT License.
