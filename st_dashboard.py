import streamlit as st
from PIL import Image

from utils import color_segmented_map, classify_img, query_VQA

st.set_page_config(layout="wide")


st.title("FloodNet Dataset, Test The Models")
col1, col2, col3, col4 = st.columns([1, 1, 0.5, 0.8])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    seg_mask = color_segmented_map(img)
    pred_class = classify_img(img)
    img = img.resize((720, 720), Image.Resampling.LANCZOS)
    flood_status = "Flooded" if pred_class == 1 else "Non flooded"
    with col1:
        st.subheader("Computed Classification")
        st.image(img, use_container_width=True)
        st.markdown(f"**Predicted Image Class:** `{flood_status}`")
        
    with col2:
        st.subheader("Computed Segmentation")
        st.image(Image.fromarray(seg_mask), use_container_width=True)
    with col3:
        st.markdown("<br><br><br><br>",  unsafe_allow_html=True)
        st.image(".\\Report_IMGS\\color-class-legend.png", use_container_width=True)
    with col4:
        st.subheader("VQA Answer")
        st.markdown(
                '<span style="font-size:18px">Legend: '
                '<span style="color:red">Questions</span> '
                '<span style="color:green">Predicted Answers</span> --- '
                '<span style="color:blue">Ground Truth</span></span>',
                unsafe_allow_html=True
        )
        questions, answers, gts = query_VQA(img, uploaded_file.name)
        for i in range(len(questions)):
            st.markdown(f":red[{questions[i]}] :green[{answers[i]}] --- :blue[{gts[i]}]")
        
                

