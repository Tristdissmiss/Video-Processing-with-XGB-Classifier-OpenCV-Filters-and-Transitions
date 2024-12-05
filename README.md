### **README: Video Processing with XGB Classifier, OpenCV Filters, and Transitions**

---

#### **Overview**
This project implements a **video analysis pipeline** combining machine learning predictions, frame smoothing, OpenCV filters, and creative transitions. The objective is to create a polished highlight reel showcasing the smooth transitions and ball tracking, providing an insight into raw and smoothed data overlay.

Key components of the workflow include:
1. **Using an XGBoost Classifier for Predictions**:
   - XGBoost (`XGBClassifier`) was used to predict ball movement across frames, generating a predictions file (`predictionsXGB.csv`).Which i changed the name of in the final to be smoothed_predictions.
   - The predictions are smoothed using a custom **frame smoothing algorithm** to mitigate noise and improve tracking accuracy.

2. **Integration with OpenCV**:
   - OpenCV was utilized to:
     - Apply filters (e.g., black-and-white transformation).
     - Implement ball tracking with trails.
     - Skip inactive video segments where no ball activity occurs. 
     - Transitions 

3. **Adding Transitions**:
   - Transitions such as `fade`, `wipe_left`, and `dissolve` are incorporated between significant frames to create a highlight reel.

---

#### **Project Structure**
- **`filter_predictions.py`**: Handles prediction smoothing, plotting raw vs. smoothed predictions, and preparing smoothed predictions for OpenCV integration.
- **`opencv_intro.py`**: Processes the video using filters, ball tracking, and skipping inactive frames. Outputs the final processed video.
- **`create_transition.py`**: Adds transitions between specified frames, creating smooth transitions in the final highlight reel.

---

#### **Key Insights**
1. **Why Raw and Smoothed Predictions Align**:
   - **Observation**: The raw predictions line overlaps the smoothed line for most of the video.
   - **Reason**: The smoothing parameters (`window_size=5`, `min_duration=1`, `hysteresis=0.6`) retain the structure of raw predictions while filtering minor noise. The XGB classifier already provides accurate predictions, requiring minimal smoothing adjustments.

2. **Highlight Reel Outcome**:
   - The pipeline generates a video showing ball tracking and transitions between active segments, giving a professional highlight reel feel.
   - Transitions enhance visual appeal by connecting key events seamlessly. 

   

---

#### **Requirements**
##### **Python Version**
- Python 3.8 or higher.

##### **Dependencies**
Install the following Python packages:
```bash
pip install numpy pandas scikit-learn xgboost opencv-python matplotlib


### **Pre and Post Transition Videos**:
Pre-Transition Video: https://youtu.be/2C-3KkhcQPc

Post-Transition Video: https://youtu.be/a0e8n5CshF0"# Video-Processing-with-XGB-Classifier-OpenCV-Filters-and-Transitions" 
