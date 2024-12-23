import streamlit as st
import numpy as np
import pandas as pd
import librosa
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Model
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Parkinson's Disease Voice Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 1em;
        background-color: #2ecc71;
        color: white;
    }
    .prediction-box {
        padding: 2em;
        border-radius: 10px;
        margin: 1em 0;
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
    }
    .info-box {
        background-color: #000000;
        padding: 1.5em;
        border-radius: 8px;
        margin: 1em 0;
        border-left: 4px solid #2ecc71;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stAlert {
        background-color: #000000;
        border: 2px solid #2ecc71;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .disclaimer {
        background-color: #fff3cd;
        color: #856404;
        padding: 1em;
        border-radius: 8px;
        margin: 1em 0;
    }
    .about-section {
        background-color: #000000;
        padding: 1.5em;
        border-radius: 8px;
        margin: 1em 0;
        border-left: 4px solid #3498db;
    }
    .guidelines-section {
        background-color: #000000;
        padding: 1.5em;
        border-radius: 8px;
        margin: 1em 0;
        border-left: 4px solid #2ecc71;
    }
    .result-section {
        background-color: #000000;
        padding: 2em;
        border-radius: 10px;
        margin: 1em 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_prediction_model():
    """
    Load the pickled model state and reconstruct the model
    """
    try:
        model_path = Path('models/best_parkinsons_model.pkl')
        
        if not model_path.exists():
            st.error("Model file not found. Please ensure 'models/best_parkinsons_model.pkl' exists.")
            return None, None
        
        with open(model_path, 'rb') as f:
            model_state = pickle.load(f)
        
        model = Model.from_config(model_state['model_config'])
        model.set_weights(model_state['model_weights'])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model, model_state['scaler']
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def extract_features(audio_file):
    """
    Extract exactly 29 features from audio file to match the training data dimensions
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=22050)
        
        # 1. Basic Features
        # Fundamental frequency (F0) statistics
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'))
        f0 = f0[~np.isnan(f0)]  # Remove NaN values
        if len(f0) == 0:  # If no valid F0 values found
            f0_mean, f0_std, f0_min = 0, 0, 0
        else:
            f0_mean = np.mean(f0)
            f0_std = np.std(f0)
            f0_min = np.min(f0)
        
        # 2. Jitter Features
        y_trimmed = y[y != 0]  # Remove zero values
        jitter_abs = np.mean(np.abs(np.diff(y_trimmed)))
        jitter_rel = jitter_abs / np.mean(np.abs(y_trimmed))
        rap = np.mean(np.abs(np.diff(y_trimmed, n=3)))  # Relative Average Perturbation
        ppq = np.mean(np.abs(np.diff(y_trimmed, n=5)))  # Period Perturbation Quotient
        
        # 3. Shimmer Features
        shimmer_abs = np.mean(np.abs(np.diff(np.abs(y_trimmed))))
        shimmer_rel = shimmer_abs / np.mean(np.abs(y_trimmed))
        apq3 = np.mean(np.abs(np.diff(np.abs(y_trimmed), n=3)))  # Amplitude Perturbation Quotient (3)
        apq5 = np.mean(np.abs(np.diff(np.abs(y_trimmed), n=5)))  # Amplitude Perturbation Quotient (5)
        
        # 4. Noise Features
        # Harmonics-to-Noise Ratio (HNR)
        hnr = np.mean(librosa.feature.rms(y=y))
        # Noise-to-Harmonics Ratio (NHR)
        nhr = 1 / (hnr + 1e-10)  # Adding small constant to avoid division by zero
        
        # 5. Spectral Features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # 6. MFCC Features (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
        mfcc_means = np.mean(mfccs[1:], axis=1)  # Skip first MFCC coefficient
        
        features = np.array([
            f0_mean, f0_std, f0_min,                   # 3 features
            jitter_abs, jitter_rel, rap, ppq,          # 4 features
            shimmer_abs, shimmer_rel, apq3, apq5,      # 4 features
            hnr, nhr,                                  # 2 features
            np.mean(spec_cent),                        # 1 feature
            np.mean(spec_bw),                          # 1 feature
            np.mean(spec_rolloff),                     # 1 feature
            *mfcc_means                                # 13 features
        ])
        
        feature_names = [
            'f0_mean', 'f0_std', 'f0_min',
            'jitter_abs', 'jitter_rel', 'rap', 'ppq',
            'shimmer_abs', 'shimmer_rel', 'apq3', 'apq5',
            'hnr', 'nhr',
            'spec_cent', 'spec_bw', 'spec_rolloff'
        ] + [f'mfcc_{i}' for i in range(1, 14)]
        
        print("\nFeatures extracted:")
        for i, (name, value) in enumerate(zip(feature_names, features)):
            print(f"{i+1}. {name}: {value}")
        print(f"\nTotal features: {len(features)}")
        
        assert len(features) == 29, f"Expected 29 features, but got {len(features)}"
        
        features_reshaped = features.reshape(1, -1)
        
        return features_reshaped, y, sr
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None, None

def plot_waveform(y, sr):
    fig = go.Figure()
    time_points = np.arange(len(y)) / sr
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=y, 
        mode='lines', 
        name='Waveform',
        line=dict(color='#2ecc71', width=1.5)
    ))
    fig.update_layout(
        title="Voice Waveform Analysis",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    return fig

def plot_mfcc(mfcc):
    fig = px.imshow(
        mfcc.reshape(1, -1),
        aspect='auto',
        labels=dict(x="MFCC Coefficients", y=""),
        title="Acoustic Feature Analysis",
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    return fig

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Analysis Confidence", 'font': {'size': 24, 'color': '#2c3e50'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#2c3e50"},
            'bar': {'color': "#2ecc71"},
            'steps': [
                {'range': [0, 30], 'color': "#ebfaf0"},
                {'range': [30, 70], 'color': "#a1f4c2"},
                {'range': [70, 100], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "#e74c3c", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    return fig

def create_feature_importance_plot(features, importance_scores):
    fig = go.Figure(go.Bar(
        x=importance_scores,
        y=features,
        orientation='h',
        marker_color='#2ecc71'
    ))
    fig.update_layout(
        title="Feature Importance Analysis",
        xaxis_title="Importance Score",
        yaxis_title="Acoustic Features",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    return fig

def main():
    with st.sidebar:
        st.image("logo.jpg", width=150)
        
        st.markdown("## About")
        st.markdown("""
        <div class="about-section">
            This advanced clinical tool utilizes deep learning technology
            to analyze voice patterns for potential indicators of
            Parkinson's Disease. The analysis is based on acoustic
            features extracted from voice recordings.
            
            The system analyzes multiple aspects of voice including:
            - Frequency variations
            - Amplitude perturbations
            - Harmonic components
            - Non-linear dynamics
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Recording Guidelines")
        st.markdown("""
        <div class="guidelines-section">
            For optimal results, please follow these guidelines:
            - Use a high-quality microphone
            - Record in a quiet environment
            - Maintain 6-8 inches from microphone
            - Speak naturally at normal volume
            - Record for at least 5 seconds
            - Sustain a vowel sound (like 'aaah')
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Medical Disclaimer")
        st.markdown("""
        <div class="disclaimer">
            This tool is designed to assist healthcare professionals
            in their assessment process. It should not be used as a
            standalone diagnostic tool. Always consult with qualified
            healthcare providers for proper medical diagnosis.
            
            The analysis provided is based on statistical models and
            should be interpreted by healthcare professionals in
            conjunction with other clinical findings.
        </div>
        """, unsafe_allow_html=True)

    st.title("🎯 Parkinson's Voice Analysis System")
    
    st.markdown("""
    <div class="info-box">
        Welcome to the Parkinson's Voice Analysis System. This tool uses advanced
        machine learning algorithms to analyze voice recordings and identify potential
        indicators associated with Parkinson's Disease.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Voice Recording",
        type=["wav", "mp3"],
        help="Upload a clear voice recording for analysis"
    )

    if uploaded_file:
        with st.spinner("Analyzing voice patterns..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Extracting acoustic features...")
            progress_bar.progress(30)
            features, y, sr = extract_features(uploaded_file)
            
            if features is not None:
                status_text.text("Loading analysis model...")
                progress_bar.progress(60)
                model, scaler = load_prediction_model()
                
                if model is not None:
                    try:
                        status_text.text("Generating predictions...")
                        progress_bar.progress(90)
                        
                        # Scale the features
                        features_scaled = scaler.transform(features)
                        # Reshape for LSTM input (batch_size, timesteps, features)
                        features_reshaped = features_scaled.reshape(1, 1, -1)
                        # Make prediction
                        prediction = model.predict(features_reshaped, verbose=0)
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Display results
                        st.markdown("""
                        <div class="result-section">
                            <h2>Analysis Results</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if prediction[0][0] > 0.5:
                                st.error("### Assessment: Potential Indicators Present")
                                st.markdown("""
                                    The analysis suggests presence of vocal patterns that may be
                                    associated with Parkinson's Disease. Please consult with a
                                    healthcare provider for proper evaluation.
                                """)
                            else:
                                st.success("### Assessment: No Significant Indicators")
                                st.markdown("""
                                    The analysis does not detect significant vocal patterns
                                    associated with Parkinson's Disease. Regular check-ups
                                    are still recommended.
                                """)
                        
                        with col2:
                            st.plotly_chart(create_gauge_chart(prediction[0][0]), use_container_width=True)
                        
                        st.markdown("## Acoustic Analysis")
                        tab1, tab2 = st.tabs(["Voice Pattern", "Spectral Features"])
                        
                        with tab1:
                            st.plotly_chart(plot_waveform(y, sr), use_container_width=True)
                        
                        with tab2:
                            st.plotly_chart(plot_mfcc(features[0]), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.error("Please ensure the audio file is clear and properly formatted.")
                    
                    st.markdown("## Acoustic Analysis")
                    
                    tab1, tab2 = st.tabs(["Voice Pattern", "Spectral Features"])
                    
                    with tab1:
                        st.plotly_chart(plot_waveform(y, sr), use_container_width=True)
                        st.markdown("""
                            The waveform shows the temporal characteristics of the voice signal.
                            Key aspects analyzed include amplitude variations, frequency components,
                            and signal stability.
                        """)
                    
                    with tab2:
                        st.plotly_chart(plot_mfcc(features), use_container_width=True)
                        st.markdown("""
                            The spectral features represent the acoustic characteristics
                            of the voice, including frequency content, spectral envelope,
                            and temporal dynamics.
                        """)
                    
                    st.markdown("## Technical Parameters")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Recording Duration",
                            value=f"{len(y)/sr:.2f}s",
                            delta="Recommended: >5s"
                        )
                    
                    with col2:
                        st.metric(
                            label="Sample Rate",
                            value=f"{sr} Hz",
                            delta="High Quality" if sr >= 44100 else "Standard"
                        )
                    
                    with col3:
                        st.metric(
                            label="Features Analyzed",
                            value=len(features),
                            delta="Complete" if len(features) == 22 else "Partial"
                        )

    else:
        st.info("👆 Upload a voice recording to begin analysis")
        
        st.markdown("## Sample Analysis Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            t = np.linspace(0, 2, 1000)
            y = np.sin(2 * np.pi * 5 * t) * np.exp(-t)
            fig = plot_waveform(y, 1000)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            sample_features = np.random.randn(22)
            fig = plot_mfcc(sample_features)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Created by Garv Anand(with team Pravartak)</p>
            <p>For support or questions: garvanand03@gmail.com</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()