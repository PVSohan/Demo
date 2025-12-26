"""
InstruPlay AI - Streamlit Web Interface
Music Instrument Recognition System
"""

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import tempfile

# Page configuration
st.set_page_config(
    page_title="InstruPlay AI - Instrument Recognition",
    page_icon="music_icon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        text-align: center;
        color: #666;
        margin-bottom: 1rem;
    }
    .instrument-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .detected {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .not-detected {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    /* Make tabs more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: white;
        border-radius: 5px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8f4ff;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .stTabs [aria-selected="true"]:hover {
        background-color: #1664a0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31,119,180,0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Configuration
CONFIG = {
    "sample_rate": 22050,
    "mel_bands": [64, 96, 128],
    "n_fft": 2048,
    "hop_length": 512,
    "detection_threshold": 0.40,  # Lowered from 0.5 to catch marginal predictions
    "instruments": [
        "Acoustic Guitar",
        "Cello",
        "Clarinet",
        "Electric Guitar",
        "Flute",
        "Organ",
        "Piano",
        "Saxophone",
        "Trumpet",
        "Violin",
        "Voice",
    ],
}


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Try multiple possible model filenames
        model_paths = [
            "best_model.keras",
            "instrument_classifier_v2.keras",
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                st.success(f"✅ Loaded model: {model_path}")
                return model

        st.error(
            "Model file not found. Please upload best_model.keras or instrument_classifier_v2.keras"
        )
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def extract_features_from_audio(audio, target_time_dim=259):
    """Extract multi-resolution mel spectrograms from audio array"""
    try:
        features = []

        for n_mels in CONFIG["mel_bands"]:
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=CONFIG["sample_rate"],
                n_fft=CONFIG["n_fft"],
                hop_length=CONFIG["hop_length"],
                n_mels=n_mels,
                power=2.0,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
            mel_db = mel_db.astype(np.float32)

            # Pad or crop
            if mel_db.shape[1] < target_time_dim:
                pad_width = target_time_dim - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
            elif mel_db.shape[1] > target_time_dim:
                mel_db = mel_db[:, :target_time_dim]

            features.append(np.expand_dims(mel_db, axis=-1))

        return features
    except Exception as e:
        st.error(f"Error extracting features from audio: {e}")
        return None


def extract_features(audio_file, target_time_dim=259):
    """Extract multi-resolution mel spectrograms from audio file"""
    try:
        audio, _ = librosa.load(audio_file, sr=CONFIG["sample_rate"], mono=True)
        features = extract_features_from_audio(audio, target_time_dim)
        return features, audio
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None


def predict_instruments(model, features):
    """Predict instruments from features"""
    try:
        # Add batch dimension
        X = [np.expand_dims(f, axis=0) for f in features]
        predictions = model.predict(X, verbose=0)[0]
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


def sliding_window_predict(model, audio_file, window_size=1.0, hop_size=0.5):
    """Predict instruments over time using sliding window"""
    try:
        audio, sr = librosa.load(audio_file, sr=CONFIG["sample_rate"])
        duration = librosa.get_duration(y=audio, sr=sr)
        times = np.arange(0, duration - window_size, hop_size)
        all_preds = []

        progress_bar = st.progress(0)
        for idx, t in enumerate(times):
            start = int(t * sr)
            end = int((t + window_size) * sr)
            segment = audio[start:end]

            if len(segment) < int(window_size * sr):
                pad = np.zeros(int(window_size * sr) - len(segment))
                segment = np.concatenate([segment, pad])

            features = extract_features_from_audio(segment)
            if features:
                X = [np.expand_dims(f, axis=0) for f in features]
                pred = model.predict(X, verbose=0)
                all_preds.append(pred[0])

            progress_bar.progress((idx + 1) / len(times))

        progress_bar.empty()
        return np.array(all_preds), times
    except Exception as e:
        st.error(f"Error in sliding window prediction: {e}")
        return None, None


# Header
st.markdown('<div class="main-header">🎵 InstruPlay AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automatic Music Instrument Recognition System</div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image(
        "https://via.placeholder.com/300x100/1f77b4/ffffff?text=InstruPlay+AI",
        use_container_width=True,
    )
    st.header("📋 About")
    st.info(
        """
    InstruPlay AI uses Convolutional Neural Networks (CNNs) to automatically detect 
    instruments in music tracks.
    
    **Features:**
    - Multi-instrument detection
    - Real-time analysis
    - Confidence scoring
    - Timeline visualization
    - JSON/PDF export
    """
    )

    st.header("🎼 Supported Instruments")
    for inst in CONFIG["instruments"]:
        st.write(f"• {inst}")

# Main content
tab1, tab2, tab3 = st.tabs(
    ["🎵 Upload & Analyze", "📊 Analysis Results", "📥 Export Reports"]
)

with tab1:
    st.header("Upload Audio File")

    uploaded_file = st.file_uploader(
        "Choose a music file (WAV format recommended)",
        type=["wav", "mp3", "ogg", "flac"],
        help="Upload your audio file to analyze instruments",
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Audio Player")
            st.audio(uploaded_file, format="audio/wav")

        with col2:
            st.subheader("File Information")
            audio, sr = librosa.load(tmp_path, sr=CONFIG["sample_rate"])
            duration = librosa.get_duration(y=audio, sr=sr)
            st.write(f"**Duration:** {duration:.2f} seconds")
            st.write(f"**Sample Rate:** {sr} Hz")
            st.write(f"**Samples:** {len(audio)}")

        # Waveform visualization
        st.subheader("Audio Waveform")
        fig, ax = plt.subplots(figsize=(12, 3))
        times = np.linspace(0, duration, len(audio))
        ax.plot(times, audio, linewidth=0.5, alpha=0.7)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Audio Waveform")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Spectrogram visualization
        st.subheader("Mel Spectrogram (128 bands)")
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="viridis"
        )
        ax.set_title("Mel Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)
        plt.close()

        # Analysis button
        if st.button(
            "🔍 Analyze Instruments", type="primary", use_container_width=True
        ):
            with st.spinner("Loading model..."):
                model = load_model()

            if model is not None:
                with st.spinner("Analyzing audio... This may take a moment."):
                    # Quick prediction
                    features, _ = extract_features(tmp_path)
                    if features:
                        predictions = predict_instruments(model, features)

                        if predictions is not None:
                            # Store results in session state
                            st.session_state["predictions"] = predictions
                            st.session_state["audio_path"] = tmp_path
                            st.session_state["filename"] = uploaded_file.name
                            st.session_state["duration"] = duration

                            st.success(
                                "✅ Analysis complete! Check the 'Analysis Results' tab."
                            )

                            # Timeline analysis
                            with st.spinner("Generating timeline analysis..."):
                                preds_timeline, times_timeline = sliding_window_predict(
                                    model, tmp_path
                                )
                                if preds_timeline is not None:
                                    st.session_state["timeline_preds"] = preds_timeline
                                    st.session_state["timeline_times"] = times_timeline

with tab2:
    st.header("Analysis Results")

    if "predictions" in st.session_state:
        predictions = st.session_state["predictions"]

        st.subheader("Detected Instruments")

        # Adjustable threshold
        threshold = st.slider(
            "🎯 Detection Threshold",
            min_value=0.1,
            max_value=0.9,
            value=CONFIG["detection_threshold"],
            step=0.05,
            help="Lower threshold to see marginal predictions. Default: 0.40",
        )

        # Create columns for instrument cards
        cols = st.columns(3)
        for idx, inst in enumerate(CONFIG["instruments"]):
            with cols[idx % 3]:
                confidence = predictions[idx]
                is_detected = confidence > threshold
                is_marginal = 0.30 < confidence <= threshold

                if is_detected:
                    card_class = "detected"
                    status = "✅ Present"
                elif is_marginal:
                    card_class = "detected"  # Show as detected but with warning
                    status = "⚠️ Likely (Low Confidence)"
                else:
                    card_class = "not-detected"
                    status = "❌ Not Present"

                st.markdown(
                    f"""
                <div class="instrument-card {card_class}">
                    <h4>{inst}</h4>
                    <p><strong>{status}</strong></p>
                    <p>Confidence: {confidence:.2%}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.progress(float(confidence))

        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Instruments Detected", sum(1 for p in predictions if p > threshold)
            )
        with col2:
            st.metric(
                "Marginal Detections",
                sum(1 for p in predictions if 0.30 < p <= threshold),
            )
        with col3:
            st.metric("Average Confidence", f"{np.mean(predictions):.2%}")
        with col4:
            st.metric("Maximum Confidence", f"{np.max(predictions):.2%}")

        # Confidence bar chart
        st.subheader("Confidence Scores")
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = [
            "green" if p > threshold else "orange" if p > 0.3 else "red"
            for p in predictions
        ]
        bars = ax.barh(CONFIG["instruments"], predictions, color=colors, alpha=0.7)
        ax.set_xlabel("Confidence Score")
        ax.set_xlim(0, 1)
        ax.axvline(
            x=threshold,
            color="blue",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Detection Threshold ({threshold:.0%})",
        )
        ax.axvline(
            x=0.3,
            color="orange",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label="Marginal (30%)",
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)
        plt.close()

        # Timeline analysis
        if "timeline_preds" in st.session_state:
            st.subheader("Instrument Intensity Over Time")
            preds_timeline = st.session_state["timeline_preds"]
            times_timeline = st.session_state["timeline_times"]

            fig, ax = plt.subplots(figsize=(14, 6))
            for i, inst in enumerate(CONFIG["instruments"]):
                ax.plot(times_timeline, preds_timeline[:, i], label=inst, linewidth=2)
            ax.set_xlabel("Time (seconds)", fontsize=12)
            ax.set_ylabel("Predicted Probability", fontsize=12)
            ax.set_title("Instrument Presence Timeline", fontsize=14, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("👈 Upload and analyze an audio file first to see results.")

with tab3:
    st.header("Export Reports")

    if "predictions" in st.session_state:
        predictions = st.session_state["predictions"]
        filename = st.session_state["filename"]

        st.subheader("Download Analysis Reports")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**JSON Report**")
            st.write("Download detailed analysis data in JSON format")

            # Create JSON report
            report = {
                "filename": filename,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": st.session_state.get("duration", 0),
                "detected_instruments": {
                    inst: {
                        "confidence": float(predictions[idx]),
                        "detected": bool(predictions[idx] > 0.5),
                    }
                    for idx, inst in enumerate(CONFIG["instruments"])
                },
                "summary": {
                    "total_detected": int(sum(1 for p in predictions if p > 0.5)),
                    "average_confidence": float(np.mean(predictions)),
                    "max_confidence": float(np.max(predictions)),
                },
            }

            if "timeline_preds" in st.session_state:
                report["timeline"] = [
                    {
                        "time": float(t),
                        **{
                            inst: float(st.session_state["timeline_preds"][j, i])
                            for i, inst in enumerate(CONFIG["instruments"])
                        },
                    }
                    for j, t in enumerate(st.session_state["timeline_times"])
                ]

            json_str = json.dumps(report, indent=2)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_str,
                file_name=f"{filename}_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

        with col2:
            st.write("**Text Report**")
            st.write("Download human-readable analysis report")

            # Create text report
            text_report = f"""
InstruPlay AI - Instrument Recognition Report
{'='*50}

File: {filename}
Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Duration: {st.session_state.get('duration', 0):.2f} seconds

DETECTED INSTRUMENTS:
{'='*50}
"""
            for idx, inst in enumerate(CONFIG["instruments"]):
                if predictions[idx] > CONFIG["detection_threshold"]:
                    status = "✓ PRESENT"
                elif predictions[idx] > 0.3:
                    status = "⚠ MARGINAL"
                else:
                    status = "✗ NOT PRESENT"
                text_report += (
                    f"\n{inst:20s} : {status:12s} (Confidence: {predictions[idx]:.2%})"
                )

            text_report += f"""

SUMMARY:
{'='*50}
Detection Threshold: {CONFIG["detection_threshold"]:.0%}
Total Instruments Detected: {sum(1 for p in predictions if p > CONFIG["detection_threshold"])}
Marginal Detections: {sum(1 for p in predictions if 0.3 < p <= CONFIG["detection_threshold"])}
Average Confidence: {np.mean(predictions):.2%}
Maximum Confidence: {np.max(predictions):.2%}

Report generated by InstruPlay AI
"""

            st.download_button(
                label="📥 Download Text Report",
                data=text_report,
                file_name=f"{filename}_analysis.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # Display preview
        st.subheader("Report Preview")
        with st.expander("View JSON Report"):
            st.json(report)

        with st.expander("View Text Report"):
            st.code(text_report)

    else:
        st.info("👈 Analyze an audio file first to generate reports.")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>InstruPlay AI</strong> - Powered by CNN & TensorFlow</p>
    <p>Automatic Music Instrument Recognition System</p>
</div>
""",
    unsafe_allow_html=True,
)
