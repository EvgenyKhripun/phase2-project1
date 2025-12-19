import streamlit as st
from PIL import Image
from utils import load_model, process_image, predict

st.set_page_config(page_title="Image Classification", layout="wide")

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Classify Images", "Model Info"])

# --- Load model once ---
@st.cache_resource
def get_model():
    return load_model("model/model_weights.pth")

model = get_model()

# --- Page: Classification ---
if page == "Classify Images":
    st.title("Image Classification with ResNet18")

    # --- Image upload from device ---
    st.subheader("Upload image(s) from your device")
    uploaded_files = st.file_uploader(
        "Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    # --- Image upload from URL ---
    st.subheader("Or load image from URL")
    image_url = st.text_input("Enter image URL here:")
    load_url_btn = st.button("Load Image from URL")

    images_to_classify = []

    # From uploaded files
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f).convert('RGB')
            images_to_classify.append(img)

    # From URL via button
    if load_url_btn and image_url:
        try:
            img, _ = process_image(image_url)
            images_to_classify.append(img)
        except Exception as e:
            st.error(f"Не удалось загрузить изображение с URL: {e}")

    # Display and predict
    for idx, img in enumerate(images_to_classify):
        # Resize image for display
        img_display = img.resize((600, 600))
        st.image(img_display, caption=f"Image {idx+1}", use_column_width=False)

        # Process image and predict
        _, tensor = process_image(img)
        pred_class, elapsed = predict(model, tensor)

        # Display prediction above the image
        st.markdown(f"<h3 style='text-align: center; color: black;'>Predicted class: {pred_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: gray;'>Inference time: {elapsed:.3f} sec</p>", unsafe_allow_html=True)

# --- Page: Model Info ---
if page == "Model Info":
    st.title("Model Info & Metrics")
    st.write("Number of classes:", len(model.model.fc.weight))