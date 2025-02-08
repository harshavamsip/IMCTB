# import libraries
import os
import re
import string
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from pickle import load
from PIL import Image

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

import tensorflow as tf
from models.custom_model import Captioner, TokenOutput

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "https://github.com/eeshawn11/DSI-Capstone/issues",
        "About": "Thanks for dropping by!"
        }
    )

if "WORKSPACE_PATH" not in st.session_state:
    switch_page("Home")

if "disabled" not in st.session_state:
    st.session_state.disabled = False

max_caption_length = 73
MODEL_PATH = os.path.join(st.session_state.WORKSPACE_PATH, "models")
VOCAB_PATH = os.path.join(MODEL_PATH, "tokenizer_vocab.pkl")
OUTPUT_LAYER_PATH = os.path.join(MODEL_PATH, "output_layer_bias.pkl")

def get_image_from_url(url):
    image_path = tf.keras.utils.get_file(origin=url)
    return image_path

def load_image(image_path, image_shape=(224, 224, 3), preserve_aspect_ratio=False):
    if type(image_path) == str:
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
    else:
        img = Image.open(image_path).convert("RGB")
    img = tf.image.resize(img, image_shape[:-1], preserve_aspect_ratio=preserve_aspect_ratio)
    return img

def predict_caption(image_path):
    image = load_image(image_path)
    pred = model.simple_gen(image)
    attention_plot = model.run_and_show_attention(image)
    return pred, attention_plot

@st.cache_resource(show_spinner="Building model...")
def build_model():
    # image encoder
    feature_extractor = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        include_preprocessing=True,
        input_shape=(224, 224, 3),
    )

    # tokenizer
    def standardize(s):
        s = tf.strings.lower(s)
        s = tf.strings.regex_replace(s, f"[{re.escape(string.punctuation)}]", "")
        s = tf.strings.join(["[START]", s, "[END]"], separator=" ")
        return s

    with open(VOCAB_PATH, 'rb') as f:
        vocab = load(f)
    # load original vocab
    tokenizer = tf.keras.layers.TextVectorization(
        vocabulary=vocab, standardize=standardize, ragged=True
    )

    with open(OUTPUT_LAYER_PATH, 'rb') as f:
        output_layer_bias = load(f)

    output_layer = TokenOutput(tokenizer, banned_tokens=("", "[UNK]", "[START]"))
    output_layer.set_bias(output_layer_bias)

    model = Captioner(
        tokenizer,
        feature_extractor=feature_extractor,
        output_layer=output_layer,
        units=256,
        max_length=max_caption_length,
        dropout_rate=0.5,
        num_layers=2,
        num_heads=2,
    )

    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(MODEL_PATH + "/attention_weights/").expect_partial()

    return model

with st.sidebar:
    st.markdown(
        """
        Created by [**eeshawn**](https://eeshawn.com)

        - Connect on [**LinkedIn**](https://www.linkedin.com/in/shawn-sing/)
        - Project source [**code**](https://github.com/eeshawn11/DSI-Capstone/)
        - Check out my other projects on [**GitHub**](https://github.com/eeshawn11/)
        """
        )

model = build_model()

intro_placeholder = st.empty()
results_placeholder = st.empty()

tab1, tab2 = st.tabs(["Upload Image", "Load from URL"])
with tab1:
    uploaded_file = st.file_uploader(
        "Upload an image to generate a caption",
        type=["PNG", "JPG", "JPEG"],
        key="uploaded_file"
    )
with tab2:
    with st.form("url_input", clear_on_submit=True):
        image_url = st.text_input(
            "Image URL",
            help="URL should end with .png, .jpg or .jpeg"
            )
        submitted = st.form_submit_button("Get Image")
    if submitted:
        url_regex_match = re.match(r"((?:https?:\\/\\/)?.*\\.(?:png|jpg|jpeg))", image_url)
        if url_regex_match is not None:
            try:
                get_image_from_url(image_url)
            except:
                st.error("Sorry, unable to retrieve the image from the provided URL. Please try a different URL or upload an image instead!")
        else:
            st.error("Invalid URL, please try again. Image URL should end with .png, .jpg or .jpeg")

st.markdown("---")

if uploaded_file:
    st.session_state.disabled = True
else:
    st.session_state.disabled = False

try:
    if uploaded_file:
        caption, attention_plot = predict_caption(uploaded_file)
    elif submitted and url_regex_match is not None:
        image_path = get_image_from_url(image_url)
        caption, attention_plot = predict_caption(image_path)
    else:
        with intro_placeholder.container():
            st.markdown(
                """
                ##### Upload an image or provide an image URL to generate a caption!
                """
            )
    
    if caption:
        with results_placeholder.container():
            st.markdown("### Predicted Caption")
            st.success(caption)
            st.image(uploaded_file if uploaded_file else image_url, use_column_width=True)
            st.markdown("#### Attention Map")
            st.pyplot(attention_plot)
            st.markdown("---")
except:
    pass
