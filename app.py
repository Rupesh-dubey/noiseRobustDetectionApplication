import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from inference import load_model, preprocess_image, predict_image
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="TB Chest X-Ray Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal-prediction {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .tb-prediction {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 TB Chest X-Ray Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-powered tuberculosis detection using Vision Transformer and EfficientNet</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("""
    **Model Architecture:**
    - Vision Transformer + EfficientNet-B7
    - Hybrid CNN-Transformer approach
    - Pyramid Pooling Module (PPM)
    - Input size: 456×456 pixels
    """)
    
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. Upload a chest X-ray image
    2. Supported formats: PNG, JPG, JPEG
    3. The model will predict:
       - **NORMAL**: No tuberculosis detected
       - **TUBERCULOSIS**: TB signs detected
    4. View confidence scores and visualization
    """)
    
    # Model loading
    @st.cache_resource
    def get_model():
        try:
            model = load_model()
            return model, True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, False
    
    model, model_loaded = get_model()
    
    if not model_loaded:
        st.error("⚠️ Model could not be loaded. Please check the model file path.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear chest X-ray image for tuberculosis detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
            
            # Image info
            st.info(f"📏 **Image Info:** {image.size[0]}×{image.size[1]} pixels, Mode: {image.mode}")
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<h2 class="sub-header">🔍 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Predict button
            if st.button("🚀 Analyze X-ray", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... Please wait."):
                    try:
                        # Make prediction
                        prediction, confidence, probabilities = predict_image(model, image)
                        
                        # Display prediction
                        if prediction == "NORMAL":
                            st.markdown(f'''
                            <div class="prediction-box normal-prediction">
                                <h3 style="color: #28a745;">✅ NORMAL</h3>
                                <p class="confidence-text">Confidence: {confidence:.1f}%</p>
                                <p>No signs of tuberculosis detected in the chest X-ray.</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="prediction-box tb-prediction">
                                <h3 style="color: #dc3545;">⚠️ TUBERCULOSIS</h3>
                                <p class="confidence-text">Confidence: {confidence:.1f}%</p>
                                <p>Signs of tuberculosis detected. Please consult a medical professional.</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Confidence chart
                        st.markdown('<h3 class="sub-header">📊 Confidence Scores</h3>', unsafe_allow_html=True)
                        
                        # Create dataframe for plotting
                        classes = ['NORMAL', 'TUBERCULOSIS']
                        df = pd.DataFrame({
                            'Class': classes,
                            'Probability': [prob * 100 for prob in probabilities],
                            'Color': ['#28a745' if cls == 'NORMAL' else '#dc3545' for cls in classes]
                        })
                        
                        # Plotly bar chart
                        fig = px.bar(
                            df, 
                            x='Class', 
                            y='Probability',
                            color='Class',
                            color_discrete_map={'NORMAL': '#28a745', 'TUBERCULOSIS': '#dc3545'},
                            title='Prediction Probabilities',
                            labels={'Probability': 'Confidence (%)'}
                        )
                        fig.update_layout(
                            showlegend=False,
                            height=400,
                            title_x=0.5
                        )
                        fig.update_traces(
                            texttemplate='%{y:.1f}%',
                            textposition='outside'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results
                        with st.expander("📈 Detailed Results"):
                            results_df = pd.DataFrame({
                                'Class': classes,
                                'Probability': [f"{prob:.4f}" for prob in probabilities],
                                'Percentage': [f"{prob*100:.2f}%" for prob in probabilities]
                            })
                            st.dataframe(results_df, use_container_width=True)
                        
                        # Medical disclaimer
                        st.warning("""
                        ⚠️ **Medical Disclaimer:** 
                        This AI model is for educational and research purposes only. 
                        It should not be used as a substitute for professional medical advice, 
                        diagnosis, or treatment. Always consult with qualified healthcare 
                        professionals for medical decisions.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.error("Please try uploading a different image or contact support.")
    
    # Additional information
    st.markdown("---")
    
    # Model performance section
    st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Model Type", "Hybrid ViT-EfficientNet")
    
    with col4:
        st.metric("Input Resolution", "456×456")
    
    with col5:
        st.metric("Classes", "2 (Normal/TB)")
    
    # About section
    with st.expander("ℹ️ About this Application"):
        st.markdown("""
        ### Model Architecture
        This application uses a hybrid Vision Transformer (ViT) combined with EfficientNet-B0 backbone:
        
        - **Backbone:** EfficientNet-B7 (pretrained)
        - **Transformer:** 8-layer Vision Transformer
        - **Enhancement:** Pyramid Pooling Module (PPM)
        - **Training:** Trained on chest X-ray dataset with data augmentation
        
        ### Key Features
        - **Hybrid Architecture:** Combines CNN feature extraction with transformer attention
        - **Multi-scale Processing:** PPM captures features at different scales
        - **High Resolution:** 456×456 input for detailed feature extraction
        - **Robust Training:** Label smoothing, data augmentation, and early stopping
        
        ### Technical Details
        - Image preprocessing includes normalization and center cropping
        - Model uses Swish activation and batch normalization
        - Trained with cosine annealing learning rate schedule
        - Early stopping prevents overfitting
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Developed with ❤️ using Streamlit | AI for Healthcare</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()