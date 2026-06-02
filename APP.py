import streamlit as st
import numpy as np
import librosa
import joblib
import os

st.set_page_config(page_title="传统乐器识别系统", layout="centered")
st.title("🎵 传统乐器智能识别系统")
st.divider()

# 模型路径（必须和 GitHub 一致）
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载模型（只加载一次）
@st.cache_resource
def load_model():
    clf    = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "svm_scaler.pkl"))
    le     = joblib.load(os.path.join(MODEL_DIR, "svm_label_encoder.pkl"))
    return clf, scaler, le

clf, scaler, le = load_model()

# 特征提取（和你原来代码完全一样）
SR = 22050
N_MFCC = 13
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 256
PRE_COEF = 0.97

def extract_features(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SR)
        y = librosa.effects.preemphasis(y, coef=PRE_COEF)
        mfccs = librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=N_MFCC, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH, window="hamming"
        )
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        feat = np.vstack([mfccs, delta, delta2])
        return np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)])
    except:
        return None

# 上传音频
audio_file = st.file_uploader("上传音频文件", type=["wav", "mp3", "flac", "ogg"])

if audio_file is not None:
    st.audio(audio_file)
    with st.spinner("正在提取特征并识别..."):
        feat = extract_features(audio_file)
        if feat is None:
            st.error("音频处理失败")
        else:
            x = scaler.transform(feat.reshape(1, -1))
            prob = clf.predict_proba(x)[0]
            idx = np.argmax(prob)
            label = le.classes_[idx]
            conf = prob[idx]

            st.success(f"✅ 识别结果：**{label}**")
            st.info(f"置信度：{conf:.2%}")

            st.subheader("各类别概率")
            data = {le.classes_[i]: prob[i] for i in range(len(le.classes_))}
            st.bar_chart(data)
