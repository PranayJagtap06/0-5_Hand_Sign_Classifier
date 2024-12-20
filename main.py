import tensorflow as tf
import streamlit as st
import mlflow
import base64
import os
from PIL import Image
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

about = "Imagine a world where hand signs are read in real-time without problems."\
    " This is where HandSignClassifier comes in, using the best weapons of machine learning and artificial intelligence to make it happen."\
    " I implemented an app that uses transfer learning with the state-of-the-art EfficientNetB0 model, with top 10 layers trainable, to recognize hand signs from 0-5."\
    " Just upload an image of your hand sign, and our model will predict the corresponding numerical class in a second with accuracy."\
    " The results are not most accurate all the time, there is still scope of improvement by training the model on larger dataset."\
    " The present model was trained on '[DL.AI] Hand Signs 05 Dataset' which consists of 1080 trainning samples."

st.set_page_config(page_title='0-5 Hand Sign Classifier', page_icon='🖐️', menu_items={'About': f"{about}"})
st.title(body="HandSignClassifier: ML-Powered Hand Sign Recognition 🤖🖐️✌️")
st.markdown("*Transform hand gestures into numbers 0-5 instantly using advanced machine learning and EfficientNetB0 technology.*")

# Initialize session state
if 'preprocessed_img' not in st.session_state:
    st.session_state.preprocessed_img = None
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading model... This may take a few moments."):

            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
            model_name = "0-5_hand_sign_classifier_1"
            model_uri = f"models:/{model_name}@champion"
            model = mlflow.tensorflow.load_model(model_uri=model_uri)
            # time.sleep(2)  # Simulate loading time, remove in production
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Failed to load model. Please try again later.")
        return None

# Load model
if st.session_state.model is None:
    st.session_state.model = load_model()

@st.cache_data
def preprocess_image(uploaded_img):
    img_shape = 224
    img_bytes = uploaded_img.getvalue()
    img = tf.io.decode_image(img_bytes, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = tf.expand_dims(img, axis=0)
    return img

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

st.markdown("""---""")
st.header("Upload an Image 📷")
upload_cols = st.columns([2, 1])
file_upload = upload_cols[0].expander(label="Select an image")
uploaded_img = file_upload.file_uploader("Choose a png or jpg file", type=["png", "jpg"])
classify_btn = upload_cols[0].button(":red[Classify]")

if uploaded_img is not None:
    try:
        img = Image.open(uploaded_img)
        upload_cols[1].image(img)
        st.session_state.preprocessed_img = preprocess_image(uploaded_img)
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        st.error("Failed to process the uploaded image. Please try again with a different image.")


def classify_img():
    start_time = datetime.now()
    class_names = tf.constant([0,1,2,3,4,5])
    try:
        with st.spinner("Classifying image..."):
            probs = st.session_state.model.predict(st.session_state.preprocessed_img)
            pred_class = class_names[tf.argmax(probs, axis=1).numpy()[0]]

        st.markdown("""---""")
        st.header("Classification Result ✨✨✨")
        st.markdown(f"Classified as hand sign: {pred_class.numpy()}")
    
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Classification completed in {duration:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        st.error("An error occurred during classification. Please try again.")

if classify_btn and st.session_state.preprocessed_img is not None:
    classify_img()

# Personal Website
st.markdown("""---""")
st.markdown("Created by [Pranay Jagtap](https://pranayjagtap.netlify.app)")

# Get the base64 string of the image
img_base64 = get_image_base64("assets/pranay_sq.jpg")

# Create the HTML for the circular image
html_code = f"""
<style>
    .circular-image {{
        width: 125px;
        height: 125px;
        border-radius: 55%;
        overflow: hidden;
        display: inline-block;
    }}
    .circular-image img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
</style>
<div class="circular-image">
    <img src="data:image/jpeg;base64,{img_base64}" alt="Pranay Jagtap">
</div>
"""

# Display the circular image
st.markdown(html_code, unsafe_allow_html=True)
st.markdown("Electrical Engineer | Machine learning Enthusiast "\
            "<br>Nagpur, Maharashtra, India", unsafe_allow_html=True)

