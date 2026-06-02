import streamlit as st
import numpy as np
import librosa
import joblib
import os

# 页面全局配置
st.set_page_config(page_title="传统乐器智能识别系统", layout="centered")

# 顶部标题美化
st.markdown("""
# 🎵 <span style='color:#364599'>传统乐器智能识别系统</span>
""", unsafe_allow_html=True)
st.divider()

# 缓存加载模型
@st.cache_resource
def load_model():
    root = os.path.dirname(os.path.abspath(__file__))
    clf = joblib.load(os.path.join(root, "svm_model.pkl"))
    scaler = joblib.load(os.path.join(root, "svm_scaler.pkl"))
    le = joblib.load(os.path.join(root, "svm_label_encoder.pkl"))
    return clf, scaler, le

clf, scaler, le = load_model()

# 特征参数
SR = 22050
N_MFCC = 13
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 256
PRE_COEF = 0.97

def extract_features(file):
    try:
        y, _ = librosa.load(file, sr=SR)
        y = librosa.effects.preemphasis(y, coef=PRE_COEF)
        mfccs = librosa.feature.mfcc(y=y, sr=SR,n_mfcc=N_MFCC,n_mels=N_MELS,n_fft=N_FFT,hop_length=HOP_LENGTH,window="hamming")
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        feat_stack = np.vstack([mfccs, delta, delta2])
        return np.concatenate([np.mean(feat_stack,axis=1), np.std(feat_stack,axis=1)])
    except Exception as e:
        st.error(f"音频解析失败：{str(e)}")
        return None

# 上传区域美化
st.subheader("📁 上传音频文件")
audio_file = st.file_uploader("支持格式：WAV / MP3 / FLAC / OGG，单文件上限200MB", type=["wav","mp3","flac","ogg"])

if audio_file:
    st.audio(audio_file)
    with st.spinner("🔍 正在提取声学特征、AI识别中，请稍候..."):
        feat = extract_features(audio_file)
        if feat is not None:
            x = scaler.transform(feat.reshape(1,-1))
            prob = clf.predict_proba(x)[0]
            pred_idx = np.argmax(prob)
            pred_name = le.classes_[pred_idx]
            conf = prob[pred_idx]

            # 识别结果卡片美化
            st.markdown("## ✅ 识别结论")
            st.success(f"乐器类别：**{pred_name}** ｜ 置信度：{conf:.2%}")

            # 各类别概率可视化
            st.markdown("### 📊 全类别预测概率分布")
            prob_dict = {le.classes_[i]:prob[i] for i in range(len(prob))}
            st.bar_chart(prob_dict, color="#4279d8")
