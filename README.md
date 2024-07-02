# HandSignClassifier 🖐️👌

<p align="center">
  <i>
    Transform hand gestures into numbers 0-5 instantly using advanced machine learning and EfficientNetB0 technology.
  </i>
</p>

<p align="center">
  <img src="https://github.com/PranayJagtap06/0-5_Hand_Sign_Classifier/blob/2988847c3d4cd925c969ee60aec318bd6238f1d5/assets/handsign-unsplash.jpg" width="400" alt="HandSign">
</p>

## About the Project 💻

Meet the **HandSignClassifier** app, a web application that identifies hand sign photos as numbers from 0 to 5. This app created with Streamlit, uses a machine learning model. It was trained through fine tuning transfer learning on the EfficientNetB0 model.

[*Visit Streamlit App*](https://0-5handsignclassifier.streamlit.app)

## How to Use 👨‍💻

Here’s how you can use the app:

 1. Expand the "Select an image" block and use the "Browse files" button to upload a hand sign image (0-5).
 2. Hit the "Classify" button to analyze the image.
 3. The app will show you which number (0-5) the hand sign represents.

## Model Details ℹ️

The app uses a classification model trained through fine tuning transfer learning on the EfficientNetB0 model. The app lets the top 10 layers of the EfficientNetB0 model learn making it possible for the model to recognize features of hand sign images. MLflow helps in the easy management, experiment tracking and deployment of the model.

## Technical Details ⚙️
 - Streamlit used to build the app
 - Model uses transfer learning on EfficientNetB0
 - EfficientNetB0's top 10 layers can learn
 - MLflow used for deploying the model
 - Supports uploads of images in formats like JPG, JPEG and PNG only.

## Getting Started 💨

 1. **Create A Dagshub Account:**
    Go to [DagsHub](https://www.dagshub.com), signup for free and create a repository or connect your GitHub repository where you'll store your python files or jupyter notebooks. We'll use DagsHub's MLflow server for experiment tracking and model deployment.
    
 3. **Setup MLflow Experiment Tracking:**
     - Train your `EfficientNetB0` model.
     - Import `dagshub`, `mlflow` & `os` in your jupyter notebook.

           import dagshub
           import mlflow
           import os

     - Setup MLflow experiment tracking function:

           def create_experiment(experiment_name,run_name, run_metrics, model, la_path = None, confusion_matrix_path = None, 
                      precission_recall_path = None, roc_path = None, run_params = None):

             # You copy and paste below code snippet from from your dagshub repo, click on "Remote" in "Experiments" tab and copy the code snippet with your `repo_owner` & `repo_name`
             dagshub.init(repo_owner='replace your repo's owner name', repo_name='replace your repo's name', mlflow=True)
   
             # Either use `dagshub.init()` or `mlflow.set_tracking_uri()`. Comment out these two line if using `dagshub.init()`
             # You can get your MLlfow tracking uri from your dagshub repo by opening "Remote" dropdown menu, go to "Experiments" tab and copy the MLflow experiment tracking uri and paste below
             remote_server_uri = "replace your MLflow tracking uri"
             mlflow.set_tracking_uri(remote_server_uri)
        
             mlflow.set_experiment(experiment_name)
            
             with mlflow.start_run(run_name=run_name):
        
               if not run_params == None:
                   for param in run_params:
                       mlflow.log_param(param, run_params[param])
                
               for metric, value in run_metrics.items():
                   if isinstance(value, list):
                       # If the metric is a list, log each value as a separate step
                       for step, v in enumerate(value):
                           mlflow.log_metric(metric, v, step=step)
                   else:
                       # If it's a single value, log it normally
                       mlflow.log_metric(metric, value)
    
               tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
               artifact_paths = {
                   'loss_accuracy': la_path,
                   'confusion_matrix': confusion_matrix_path,
                   'precision_recall_curve': precission_recall_path,
                   'roc_curve': roc_path
               }
            
               for artifact_name, path in artifact_paths.items():
                   if path and os.path.exists(path):
                       if tracking_url_type_store != "file":
                           mlflow.log_artifact(path, artifact_name)
                   elif path:
                       print(f"Warning: Artifact file not found: {path}")
    
               if tracking_url_type_store != "file":
                   mlflow.tensorflow.log_model(model, "model")
    
               mlflow.set_tags({"tag1":"Transfer Learning", "tag2":"Image Classification"})
            
             print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))

     - After training your model and defining the MLflow experiment tracking function run the function, this will log your artifacts and model on DagsHub MLflow server:

           from datetime import datetime
           experiment_name = "img_classification_transfer_learning"
           run_name = "run_"+str(datetime.now().strftime("%d-%m-%y_%H:%M:%S"))
           create_experiment(experiment_name,run_name,run_metrics,your-model)  # Replace `your-model` with your trained model

 4. **Register Your Logged Model:**
    Open your MLflow UI from your DagsHub repository's Remote menu's Experiment tab or simply enter your MLflow tracking URI in browser address bar. In your MLflow UI open yuor recent run and go to "Artifacts" tab and click "Register Model". Then select "Create New Model", enter your preferred model name and hit "Register". After successfully registering your model you will see your registered model in the "Models" tab in your MLflow UI. You can add tags or aliases to your registered model versions.

 5. **Clone Repo:**
    Now go ahead and clone this repo...

        git clone https://github.com/PranayJagtap06/0-5_Hand_Sign_Classifier.git

 6. **Export Your MLflow Tracking URI:**
    Run these commands in terminal/cmd...

        echo "export MLFLOW_TRACKING_URI='your_mlflow_tracking_uri'" >> ~/.bashrc  # run this command in terminal for linux
        setx MLFLOW_TRACKING_URI <your_mlflow_tracking_uri>  # run this command in command prompt in windows as administrator

 7. **Make Changes in main.py:**
    In your `main.py` file make changes to `load_model()`...

        model_name = "your_registered_model_name"
        model_version = your_registered_model_version
        model_uri = f"models:/{model_name}/{model_version}"

 8. **Run Streamlit**
    Now finally run streamlit...

        streamlit run main.py

## Contribution 🤝

Contributions to the IndianFutureReserves project are always welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License 📝

This project is licensed under the [MIT License](LICENSE).

## Contact 📧

For any inquiries or feedback, please feel free to reach out to the project maintainer at [pranaydgo06@duck.com](pranaydgo06@duck.com).
