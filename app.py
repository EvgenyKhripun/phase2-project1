import streamlit as st
from PIL import Image
from utils import load_model, process_image, predict
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

st.set_page_config(page_title="Image Classification", layout="wide")

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Classify Images", "Model Info"])

# --- Load model once ---
@st.cache_resource
def get_model():
    return load_model("model/full_model.pth", "model/model_weights.pth", DEVICE)

model = get_model()

# --- Page: Classification ---
if page == "Classify Images":
    st.title("Image Classification with ResNet50")

    # Single image upload
    st.subheader("Upload image(s)")
    uploaded_files = st.file_uploader(
        "Choose image files", type=["png","jpg","jpeg"], accept_multiple_files=True
    )

    # URL input
    image_url = st.text_input("Or enter image URL:")

    images_to_classify = []

    # From files
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f)
            images_to_classify.append(img)

    # From URL
    if image_url:
        try:
            img, _ = process_image(image_url)
            images_to_classify.append(img)
        except:
            st.error("Не удалось загрузить изображение с URL")

    # Display and predict
    for idx, img in enumerate(images_to_classify):
        st.image(img, caption=f"Image {idx+1}", use_column_width=True)
        _, tensor = process_image(img)
        pred_class, elapsed = predict(model, tensor, DEVICE)
        st.write(f"Predicted class: {pred_class}")
        st.write(f"Inference time: {elapsed:.3f} sec")

# --- Page: Model Info ---
if page == "Model Info":
    st.title("Model Training Info & Metrics")

    # Example: Loss/Accuracy curves
    st.subheader("Training curves")
    train_history = pd.read_csv("model/train_history.csv") if st.file_uploader("Upload train history CSV") else None
    if train_history is not None:
        fig, ax = plt.subplots()
        ax.plot(train_history['epoch'], train_history['train_loss'], label='Train Loss')
        ax.plot(train_history['epoch'], train_history['val_loss'], label='Val Loss')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

    # Example: Confusion Matrix
    st.subheader("Confusion Matrix")
    cm_file = st.file_uploader("Upload confusion matrix CSV", type="csv")
    if cm_file:
        cm_df = pd.read_csv(cm_file, index_col=0)
        fig, ax = plt.subplots()
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    st.subheader("Additional Info")
    st.write("Number of classes: 10")
    st.write("F1-score: 0.85 (пример)")
    st.write("Training time: 1h 20min")
    st.write("Dataset composition: balanced across 10 classes")