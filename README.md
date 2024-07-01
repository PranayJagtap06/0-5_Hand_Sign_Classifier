# HandSignClassifier 🖐️👌

<p align="center">
  <i>
    Transform hand gestures into numbers 0-5 instantly using advanced machine learning and EfficientNetB0 technology.
  </i>
</p>

<p align="center">
  <img src="https://github.com/PranayJagtap06/0-5_Hand_Sign_Classifier/blob/2988847c3d4cd925c969ee60aec318bd6238f1d5/assets/handsign-unsplash.jpg" width="400" alt="HandSign">
</p>

## About the Project
Meet the 0-5 Hand Sign Classifier app, a web application that identifies hand sign photos as numbers from 0 to 5. This app created with Streamlit, uses a machine learning model. It was trained through transfer learning on the EfficientNetB0 model.
How to Use
Here’s how you can use the app:
Use the "Browse files" button to upload a hand sign image (0-5).
Hit the "Classify" button to analyze the image.
The app will show you which number (0-5) the hand sign represents.
Model Details
The app uses a classification model trained through transfer learning on the EfficientNetB0 model. The app lets the top 10 layers of the EfficientNetB0 model learn making it possible for the model to recognize features of hand sign images. MLflow helps in the easy management and deployment of the model.
Technical Details
Streamlit used to build the app
Model uses transfer learning on EfficientNetB0
EfficientNetB0's top 10 layers can learn
MLflow used for deploying the model
Supports uploads of images in many formats like JPEG, PNG, etc.
Getting Started
Begin by uploading a hand sign image (0-5) and pressing the "Classify" button. The app will handle everything else!
Problems and Suggestions
If you face any problems or have suggestions for the app, please open an issue in this repository or reach out to the developer.
Authorization
This app operates under [insert license type e.g., MIT License].
