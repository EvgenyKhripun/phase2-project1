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

    uploaded_files = st.file_uploader(
        "Choose image files", type=["png","jpg","jpeg"], accept_multiple_files=True
    )
    image_url = st.text_input("Or enter image URL:")

    images_to_classify = []

    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            images_to_classify.append(img)

    if image_url:
        try:
            img, _ = process_image(image_url)
            images_to_classify.append(img)
        except:
            st.error("Не удалось загрузить изображение с URL")

    for idx, img in enumerate(images_to_classify):
        st.image(img, caption=f"Image {idx+1}", use_column_width=True)
        _, tensor = process_image(img)
        pred_class, elapsed = predict(model, tensor)
        st.write(f"Predicted class: {pred_class}")
        st.write(f"Inference time: {elapsed:.3f} sec")

# --- Page: Model Info ---
if page == "Model Info":
    st.title("Model Info & Metrics")
    st.write("Number of classes:", len(model.model.fc.weight))