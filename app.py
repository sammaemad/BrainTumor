import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model


# Load the model 
model = load_model("brain_tumor_model.h5")

st.title("Brain Tumor Detection with Grad-CAM")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((150, 150))
    img_array = np.array(img) / 255.0
    img_tensor = img_array[np.newaxis, ...]

    st.image(img, caption="Uploaded MRI", use_column_width=True)

    prediction = model.predict(img_tensor)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"
    st.subheader(f"Prediction: {label} ({prediction:.2f})")

    def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(heatmap, original_image, alpha=0.4):
        heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(
            cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR),
            1 - alpha,
            heatmap_colored,
            alpha,
            0,
        )
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    if st.button("Show Grad-CAM"):
        heatmap = make_gradcam_heatmap(img_tensor, model)
        gradcam_img = overlay_heatmap(heatmap, img)
        st.image(gradcam_img, caption="Grad-CAM", use_column_width=True)
