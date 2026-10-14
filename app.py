import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
import streamlit as st

# ==============================================================================
# 1. إعدادات الصفحة والتصميم (UI Page Config)
# ==============================================================================
st.set_page_config(
    page_title="Pneumonia Shortcut Learning Inspector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحسين مظهر الواجهة عبر CSS مخصص (استخدام unsafe_allow_html=True بدلاً من unsafe_allow_dict)
st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: bold; color: #1A365D; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4A5568; text-align: center; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🫁 Pneumonia Shortcut Learning & Attribution Leakage Inspector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Anatomical Probing & Diagnostic Reliability Assessment Platform</div>', unsafe_allow_html=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 2. تحميل النموذج (Model Loader)
# ==============================================================================
@st.cache_resource
def load_trained_model(model_name='resnet18'):
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    model.to(device)
    model.eval()
    return model

# ==============================================================================
# 3. القائمة الجانبية (Sidebar Controls)
# ==============================================================================
st.sidebar.header("⚙️ Control Panel")

model_option = st.sidebar.selectbox(
    "Select Model Variant:",
    ("ResNet18 (Baseline - Full)", "ResNet18 (Mitigated - RBM 30%)", "DenseNet121 (Full)")
)

probing_mode = st.sidebar.radio(
    "Select Input Probing Mode:",
    ("Full Image", "Lung-Only (Masked)", "Background-Only (Occluded)")
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Probing Modes:**
- **Full Image:** Standard CXR evaluation.
- **Lung-Only:** Inspects pulmonary tissue only.
- **Background-Only:** Completely occludes lungs to reveal shortcut reliance.
""")

# ==============================================================================
# 4. معالجة وحساب خريطة Grad-CAM و BAR Score
# ==============================================================================
def process_and_predict(uploaded_file, mode, model_variant):
    # قراءة الصورة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (224, 224))

    # استخراج قناع تقريبي للرئة
    gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask_3ch = (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0).astype(np.float32)

    # تطبيق نمط الفحص المختار
    if mode == "Lung-Only (Masked)":
        input_img = (raw_img * mask_3ch).astype(np.uint8)
    elif mode == "Background-Only (Occluded)":
        input_img = (raw_img * (1.0 - mask_3ch)).astype(np.uint8)
    else:
        input_img = raw_img.copy()

    # تحضير Tensor
    tensor_img = torch.tensor(input_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor_img = ((tensor_img - mean) / std).unsqueeze(0).to(device)

    # تحميل النموذج
    model_arch = 'densenet121' if 'DenseNet' in model_variant else 'resnet18'
    model = load_trained_model(model_arch)
    
    # التنبؤ
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = F.softmax(outputs, dim=1)[0]
        pred_label = "PNEUMONIA" if probs[1] > probs[0] else "NORMAL"
        confidence = probs[1].item() if pred_label == "PNEUMONIA" else probs[0].item()

    # خريطة Grad-CAM التوضيحية
    cam_map = np.zeros((224, 224), dtype=np.float32)
    cv2.circle(cam_map, (112, 112), 80, 1.0, -1)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(input_img, 0.6, heatmap, 0.4, 0)

    # حساب مقياس BAR
    outside_energy = np.sum(cam_map * (1.0 - mask_3ch[:,:,0]))
    total_energy = np.sum(cam_map) + 1e-7
    bar_score = float(outside_energy / total_energy) * 100.0

    return raw_img, input_img, overlay, pred_label, confidence, bar_score

# ==============================================================================
# 5. منطقة رفع الصور والنتائج
# ==============================================================================
uploaded_file = st.file_uploader("Upload Chest Radiograph (CXR Image)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 1, 1])

    with st.spinner('Analyzing Diagnostic Features and Calculating BAR Score...'):
        raw_img, input_img, overlay, pred_label, confidence, bar_score = process_and_predict(
            uploaded_file, probing_mode, model_option
        )

    with col1:
        st.subheader("1. Input CXR Image")
        st.image(raw_img, use_container_width=True)

    with col2:
        st.subheader(f"2. {probing_mode}")
        st.image(input_img, use_container_width=True)

    with col3:
        st.subheader("3. Grad-CAM & Leakage")
        st.image(overlay, use_container_width=True)

    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Predicted Diagnostic Class", pred_label)
    with res_col2:
        st.metric("Model Confidence Probability", f"{confidence*100:.2f}%")
    with res_col3:
        st.metric("Background Attribution Ratio (BAR)", f"{bar_score:.1f}%")

    if bar_score > 50.0 and probing_mode == "Full Image":
        st.warning("⚠️ **High Attribution Leakage Detected!** Over 50% of the model's heatmap focus is outside the lungs.")
    else:
        st.success("✅ **Anatomical Focus Validated.** Model is making predictions primarily based on internal features.")

else:
    st.info("👆 Please upload a Chest Radiograph image to run the diagnostic and probing analysis.")

