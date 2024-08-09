import streamlit as st
from PIL import Image
import numpy as np
import mtcnn
from keras.models import load_model
# Initialize the MTCNN face detector
detector = mtcnn.MTCNN()

def CM():
    model = load_model("ModelCM001 0.6607.keras")

    # Streamlit app title and description
    header = st.title("Rating Facial Beauty for Caucasian Men")
    subheader = st.subheader("This model is built on the SCUT-FBP5500 Dataset with preprocessing of the MTCNN Facial Detection")
    p1 = st.text("This model is an Image Regression Model with a Validation MSE of 0.6607")
    p2 = st.text("To rate a face, upload an image")

    # File uploader for image
    file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])

    if file is not None:
        # Open the uploaded image file
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to an array
        image_array = np.array(image)

        # Detect faces in the image
        faces = detector.detect_faces(image_array)
        
        if faces:
            # Extract the bounding box of the first face detected
            bounding_box = faces[0]['box']
            x, y, width, height = bounding_box

            # Crop the image to the bounding box
            cropped_image = image.crop((x, y, x + width, y + height))

            # Resize the cropped image to 200x200
            input_size = (200, 200)
            resized_image = cropped_image.resize(input_size)
            
            # Convert the resized image to a numpy array and scale it
            scaled_image = np.array(resized_image) / 255.0

            # Expand dimensions to match the model's input shape
            scaled_image = np.expand_dims(scaled_image, axis=0)

            # Make a prediction using the model
            prediction = model.predict(scaled_image, verbose=False)
            rating = prediction[0][0]

            # Display the cropped face and the prediction
            st.image(cropped_image, caption="Cropped Face", use_column_width=True)
            st.progress(rating / 10)
            st.text(f"The face is rated {round(rating,2)}/10".format(":.2f"))
        else:
            st.text("No faces detected.")

def AM():
    model = load_model("ModelAM001 0.9702.keras")

    # Streamlit app title and description
    header = st.title("Rating Facial Beauty for Asian Men")
    subheader = st.subheader("This model is built on the SCUT-FBP5500 Dataset with preprocessing of the MTCNN Facial Detection")
    p1 = st.text("This model is an Image Regression Model with a Validation MSE of 0.9702")
    p2 = st.text("To rate a face, upload an image")

    # File uploader for image
    file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])

    if file is not None:
        # Open the uploaded image file
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to an array
        image_array = np.array(image)

        # Detect faces in the image
        faces = detector.detect_faces(image_array)
        
        if faces:
            # Extract the bounding box of the first face detected
            bounding_box = faces[0]['box']
            x, y, width, height = bounding_box

            # Crop the image to the bounding box
            cropped_image = image.crop((x, y, x + width, y + height))

            # Resize the cropped image to 200x200
            input_size = (200, 200)
            resized_image = cropped_image.resize(input_size)
            
            # Convert the resized image to a numpy array and scale it
            scaled_image = np.array(resized_image) / 255.0

            # Expand dimensions to match the model's input shape
            scaled_image = np.expand_dims(scaled_image, axis=0)

            # Make a prediction using the model
            prediction = model.predict(scaled_image, verbose=False)
            rating = prediction[0][0]

            # Display the cropped face and the prediction
            st.image(cropped_image, caption="Cropped Face", use_column_width=True)
            st.progress(rating / 10)
            st.text(f"The face is rated {rating}")
        else:
            st.text("No faces detected.")

if 'runpage' not in st.session_state:
    st.session_state.runpage = None

def main_page():
    st.title("Welcome to the facial beauty analyzer built on the SCUTFBP500 dataset for Asian and Caucasian Men.")
    st.header("Ethnicity:")
    Abtn = st.button("Asian")
    Cbtn = st.button("Caucasian")

    if Abtn:
        st.session_state.runpage = AM
        st.experimental_rerun()
    if Cbtn:
        st.session_state.runpage = CM
        st.experimental_rerun()

# Check the session state and render the appropriate page
if st.session_state.runpage is None:
    main_page()
else:
    st.session_state.runpage()
