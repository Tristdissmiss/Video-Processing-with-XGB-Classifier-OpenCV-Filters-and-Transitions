
### **README: Video Processing with XGB Classifier, OpenCV Filters, and Transitions**

---

#### **Overview**
This project implements a **video analysis pipeline** combining machine learning predictions, frame smoothing, OpenCV filters, and creative transitions. The objective is to create a polished highlight reel that showcases smooth transitions and ball tracking, providing insights into raw and smoothed data overlays.

---

#### **Key Components**
1. **Using an XGBoost Classifier for Predictions**:
   - XGBoost (`XGBClassifier`) was used to predict ball movement across frames, generating a predictions file (`predictionsXGB.csv`), later renamed to `smoothed_predictions.csv`.
   - A custom **frame smoothing algorithm** mitigates noise and improves tracking accuracy.

2. **Integration with OpenCV**:
   - OpenCV was utilized to:
     - Apply filters (e.g., black-and-white transformations).
     - Implement ball tracking with trails.
     - Skip inactive video segments where no ball activity occurs.
     - Add transitions between video frames.

3. **Adding Transitions**:
   - Transitions such as `fade`, `wipe_left`, and `dissolve` are incorporated between significant frames, creating a professional highlight reel.

---

#### **Project Structure**
- **`filter_predictions.py`**:  
  Handles prediction smoothing, plots raw vs. smoothed predictions, and prepares the `smoothed_predictions.csv` file for OpenCV integration.  

- **`opencv_intro.py`**:  
  Processes the video using filters, implements ball tracking, skips inactive frames, and outputs the final processed video.

- **`create_transition.py`**:  
  Adds transitions between frames to ensure smooth transitions in the highlight reel.

---

#### **Key Insights**
1. **Why Raw and Smoothed Predictions Align**:  
   - The raw prediction line overlaps the smoothed line for most of the video because smoothing parameters (`window_size=5`, `min_duration=1`, `hysteresis=0.6`) retain the structure of raw predictions while filtering minor noise.  
   - The XGB classifier already provides accurate predictions, requiring minimal adjustments.

2. **Highlight Reel Outcome**:  
   - The pipeline generates a video with ball tracking and seamless transitions between active segments, resulting in a professional highlight reel.  

---

#### **Videos**
- **Pre-Transition Video**: [Watch Here](https://youtu.be/2C-3KkhcQPc)  
- **Post-Transition Video**: [Watch Here](https://youtu.be/a0e8n5CshF0)  

---

#### **Requirements**

##### **Python Version**
- Python 3.8 or higher.

##### **Dependencies**
Install the required Python packages using the following command:  
```bash
pip install numpy pandas scikit-learn xgboost opencv-python matplotlib
```

