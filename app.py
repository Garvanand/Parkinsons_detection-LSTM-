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
import time
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

def load_with_animation():
    """
    Display a loading animation with progress bar while processing
    """
    with st.spinner('Processing audio file...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        progress_bar.empty()

st.set_page_config(
    page_title="Advanced Parkinson's Voice Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Custom button styling */
    .stButton>button {
        width: 100%;
        height: 3.5em;
        margin-top: 1em;
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    
    /* Analysis cards */
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Metric containers */
    .metric-container {
        background: linear-gradient(145deg, #1a1a1a, #2a2a2a);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #2ecc71;
    }
    
    /* Alerts */
    .stAlert {
        background-color: rgba(46, 204, 113, 0.1);
        border: 1px solid #2ecc71;
        border-radius: 8px;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2a2a;
        border-radius: 4px;
        color: #ffffff;
        padding: 8px 16px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #2ecc71;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #2ecc71;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: #2a2a2a;
        color: #fff;
        text-align: center;
        padding: 5px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
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
    Extract and engineer features from audio file to match the training data dimensions
    """
    try:
        y, sr = librosa.load(audio_file, sr=22050)
        
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'))
        f0 = f0[~np.isnan(f0)]  
        if len(f0) == 0: 
            f0_mean, f0_std, f0_min = 0, 0, 0
        else:
            f0_mean = np.mean(f0)
            f0_std = np.std(f0)
            f0_min = np.min(f0)
        
        y_trimmed = y[y != 0]  
        jitter_abs = np.mean(np.abs(np.diff(y_trimmed)))
        jitter_rel = jitter_abs / np.mean(np.abs(y_trimmed))
        rap = np.mean(np.abs(np.diff(y_trimmed, n=3))) 
        ppq = np.mean(np.abs(np.diff(y_trimmed, n=5)))  
        shimmer_abs = np.mean(np.abs(np.diff(np.abs(y_trimmed))))
        shimmer_rel = shimmer_abs / np.mean(np.abs(y_trimmed))
        apq3 = np.mean(np.abs(np.diff(np.abs(y_trimmed), n=3)))  
        apq5 = np.mean(np.abs(np.diff(np.abs(y_trimmed), n=5)))  
        
        hnr = np.mean(librosa.feature.rms(y=y))
        nhr = 1 / (hnr + 1e-10)  
        
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
        mfcc_means = np.mean(mfccs[1:], axis=1)

        # Additional engineered features
        voice_instability = jitter_rel * shimmer_rel
        frequency_variations = np.max(f0) / (np.min(f0) + 1e-10) if len(f0) > 0 else 0
        amplitude_perturbation = (apq3 + apq5) / 2
        combined_jitter = np.mean([jitter_rel, rap, ppq])
        combined_shimmer = np.mean([shimmer_rel, apq3, apq5])
        signal_stability = hnr / (nhr + 1e-10)
        shimmer_to_jitter_ratio = combined_shimmer / (combined_jitter + 1e-10)
        
        # Combine all features
        features = np.array([
            f0_mean, f0_std, f0_min,                
            jitter_abs, jitter_rel, rap, ppq,        
            shimmer_abs, shimmer_rel, apq3, apq5,     
            hnr, nhr,                                
            np.mean(spec_cent),                    
            np.mean(spec_bw),                       
            np.mean(spec_rolloff),                   
            *mfcc_means,
            # Additional engineered features
            voice_instability,
            frequency_variations,
            amplitude_perturbation,
            combined_jitter,
            combined_shimmer,
            signal_stability,
            shimmer_to_jitter_ratio
        ])
        
        feature_names = [
            'f0_mean', 'f0_std', 'f0_min',
            'jitter_abs', 'jitter_rel', 'rap', 'ppq',
            'shimmer_abs', 'shimmer_rel', 'apq3', 'apq5',
            'hnr', 'nhr',
            'spec_cent', 'spec_bw', 'spec_rolloff'
        ] + [f'mfcc_{i}' for i in range(1, 14)] + [
            'voice_instability',
            'frequency_variations',
            'amplitude_perturbation',
            'combined_jitter',
            'combined_shimmer',
            'signal_stability',
            'shimmer_to_jitter_ratio'
        ]
        
        print("\nFeatures extracted:")
        for i, (name, value) in enumerate(zip(feature_names, features)):
            print(f"{i+1}. {name}: {value}")
        print(f"\nTotal features: {len(features)}")
        
        assert len(features) == 36, f"Expected 36 features, but got {len(features)}"
        
        features_reshaped = features.reshape(1, -1)
        
        return features_reshaped, y, sr
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None, None

def plot_waveform_and_features(y, sr):
    fig = plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, color='#2ecc71')
    plt.title('Waveform Analysis', color='white')
    plt.grid(True, alpha=0.3)
    
   
    plt.subplot(3, 1, 2)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max),
                           y_axis='mel', x_axis='time', cmap='viridis')
    plt.title('Mel Spectrogram Analysis', color='white')
    
    plt.subplot(3, 1, 3)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
    plt.title('Chromagram Analysis', color='white')
    
    plt.tight_layout()
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
    fig = go.Figure()
    
    sorted_idx = np.argsort(importance_scores)
    features = np.array(features)[sorted_idx]
    importance_scores = np.array(importance_scores)[sorted_idx]
    
    fig.add_trace(go.Bar(
        y=features,
        x=importance_scores,
        orientation='h',
        marker=dict(
            color=importance_scores,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Importance')
        )
    ))
    
    fig.update_layout(
        title="Voice Feature Analysis",
        xaxis_title="Relative Importance",
        yaxis_title="Acoustic Features",
        template="plotly_dark",
        height=600,
        showlegend=False
    )
    
    return fig

def main():
    with st.sidebar:
        st.image("logo.jpg", use_column_width=True)
        
        st.markdown("###  Analysis Options")
        analysis_mode = st.selectbox(
            "Choose Analysis Mode",
            ["Quick Scan", "Detailed Analysis", "Research Mode"]
        )
        
        st.markdown("###  Advanced Settings")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            0.0, 1.0, 0.5,
            help="Adjust the sensitivity of the detection"
        )
        
        feature_importance = st.checkbox(
            "Show Feature Importance",
            True,
            help="Display the contribution of each voice feature"
        )
        
        show_technical = st.checkbox(
            "Show Technical Details",
            False,
            help="Display detailed technical metrics"
        )

    st.title(" Advanced Parkinson's Voice Analysis")
    
    st.markdown("""
    <div class="analysis-card">
        <h1> Advanced Parkinson's Voice Analysis</h1>
        <p>Upload a voice recording for comprehensive analysis using advanced AI algorithms.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Voice Recording",
        type=["wav", "mp3"],
        help="Upload a clear voice recording for analysis. Recommended length: 5-10 seconds."
    )

    if uploaded_file:
        load_with_animation()
        
        features, y, sr = extract_features(uploaded_file)
        
        if features is not None:
            model, scaler = load_prediction_model()
            
            if model is not None:
                features_scaled = scaler.transform(features)
                features_reshaped = features_scaled.reshape(1, 1, -1)
                prediction = model.predict(features_reshaped)[0][0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                        <div class="analysis-card">
                            <h2>Analysis Results</h2>
                    """, unsafe_allow_html=True)
                    
                    if prediction > confidence_threshold:
                        st.error("####  Potential Indicators Detected")
                        risk_level = "High" if prediction > 0.8 else "Moderate"
                        st.markdown(f"**Risk Level**: {risk_level}")
                    else:
                        st.success("####  No Significant Indicators")
                        st.markdown("**Risk Level**: Low")
                    
                    st.markdown(f"**Confidence Score**: {prediction:.2%}")
                
                with col2:
                    st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
                
                tabs = st.tabs(["Voice Analysis", "Feature Importance", "Technical Metrics"])
                
                with tabs[0]:
                    st.pyplot(plot_waveform_and_features(y, sr))
                
                with tabs[1]:
                    if feature_importance:
                        feature_names = [
                            "Fundamental Frequency", "Frequency Variation",
                            "Amplitude Stability", "Voice Tremor",
                            "Harmonic Ratio"
                        ]
                        importance_scores = [0.8, 0.6, 0.7, 0.5, 0.4]  # Example scores
                        st.plotly_chart(create_feature_importance_plot(
                            feature_names, importance_scores
                        ), use_container_width=True)
                
                with tabs[2]:
                    if show_technical:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Signal-to-Noise Ratio", f"{np.mean(y):.2f} dB")
                        with col2:
                            st.metric("Sample Rate", f"{sr} Hz")
                        with col3:
                            st.metric("Duration", f"{len(y)/sr:.2f}s")
                
                # Recommendations section
                st.markdown("""
                    <div class="analysis-card">
                        <h3> Recommendations</h3>
                        <ul>
                            <li>Consider consulting with a healthcare provider for professional evaluation</li>
                            <li>Regular voice monitoring can help track changes over time</li>
                            <li>Voice exercises may help maintain vocal health</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

    else:
        st.info("Upload a voice recording to begin analysis")
        
        st.markdown("""
            <div class="analysis-card">
                <h3>Recording Guidelines</h3>
                <ul>
                    <li>Use a quiet environment</li>
                    <li>Maintain consistent distance from microphone (6-8 inches)</li>
                    <li>Speak naturally at normal volume</li>
                    <li>Record for at least 5 seconds</li>
                    <li>Sustain a vowel sound (like 'aaah')</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Created by Garv Anand(with team Pravartak)</p>
            <p>For support or questions: garvanand03@gmail.com</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()