

import streamlit as st
import pickle
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Spam Message Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'spam_detected' not in st.session_state:
    st.session_state.spam_detected = 0

# Define theme-specific colors
def get_colors():
    if st.session_state.theme == "light":
        return {
            "bg": "#e9edf3",
            "secondary_bg": "#dfe6f0",  # Light gray for text areas
            "text": "#31333F",
            "card_bg": "white",
            "card_border": "#c2cadb",
            "shadow": "rgba(0, 0, 0, 0.1)",
            "accent": "#ff4b4b",
            "success": "#00cc99",
            "warning": "#ffcc00",
            "info": "#3d85c6",
            "stat_default": "linear-gradient(135deg, #5e72e4, #3c4fe0)",
            "stat_spam": "linear-gradient(135deg, #ff6b6b, #ee5a24)",
            "stat_safe": "linear-gradient(135deg, #00b894, #00a085)"
        }

colors = get_colors()

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}  /* Hide hamburger menu */
    footer {visibility: hidden;}     /* Hide footer */
    header {visibility: hidden;}     /* Hide header */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Apply styling to entire app
st.markdown(
    f"""
    <style>   
    
    .stApp {{
        background-color: {colors['bg']};
        color: {colors['text']};
    }}
    
    /* Center the tab navigation bar */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1px;
        background-color: {colors['secondary_bg']};
        border-radius: 10px 10px 0 0;
        padding: 10px 10px 0 10px;
        border: 1px solid {colors['card_border']};
        border-bottom: none;
        justify-content: center !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        margin-right: 5px;
        transition: all 0.2s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {colors['card_bg']};
        border-radius: 10px 10px 0 0;
        color: {colors['accent']};
        border: none;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }}
    
    .stTabs [aria-selected="false"]:hover {{
        background-color: {colors['secondary_bg']};
        color: {colors['accent']};
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {colors['card_bg']};
        border-radius: 0 10px 10px 10px;
        padding: 20px;
        box-shadow: 0 4px 6px {colors['shadow']};
        border: 1px solid {colors['card_border']};
        border-top: none;
    }}
    
    /* Button styling */
    .stButton>button {{
        border-radius: 6px;
        height: 2.8em;
        width: 100%;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px {colors['shadow']};
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px {colors['shadow']};
    }}
    
    /* Text Area styling */
    .stTextArea textarea {{
        background-color: {colors['secondary_bg']};
        color: {colors['text']};
        border-radius: 6px;
        border: 1px solid {colors['card_border']};
        padding: 10px;
        font-size: 0.95rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    /* Radio button styling */
    .stRadio > div {{
        background-color: {colors['secondary_bg']};
        padding: 10px;
        border-radius: 6px;
        border: 1px solid {colors['card_border']};
    }}
    
    /* Slider styling */
    .stSlider > div {{
        color: {colors['accent']};
    }}
    
    /* Progress bar */
    .stProgress > div > div > div {{
        background-color: {colors['accent']};
    }}
    
    /* Expander styling */
    .stExpander {{
        background-color: {colors['secondary_bg']};
        border-radius: 6px;
        border: 1px solid {colors['card_border']};
        overflow: hidden;
    }}
    
    .stExpander details {{
        border-radius: 6px;
        box-shadow: 0 2px 4px {colors['shadow']};
    }}
    
    .stExpander summary {{
        font-weight: 600;
        padding: 8px 10px;
    }}
    
    /* Alert styling */
    .stAlert {{
        border-radius: 6px;
        box-shadow: 0 2px 4px {colors['shadow']};
    }}
    
    /* Card styling - SMALLER & MORE COMPACT */
    .metric-card {{
        background: {colors['card_bg']};
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-bottom: 10px;
        border: 1px solid {colors['card_border']};
        box-shadow: 0 3px 4px {colors['shadow']};
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px {colors['shadow']};
    }}
    
    /* Stat card styling - SMALLER & MORE COMPACT */
    .stat-card {{
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 3px 6px {colors['shadow']};
        color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .stat-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px {colors['shadow']};
    }}
    
    /* Main container styling - SMALLER & MORE COMPACT */
    .main-container {{
        background: {colors['card_bg']};
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid {colors['card_border']};
        box-shadow: 0 4px 8px {colors['shadow']};
    }}
    
    /* Prediction items styling */
    .prediction-item {{
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        width: 100%;
        box-shadow: 0 2px 4px {colors['shadow']};
        transition: transform 0.2s ease;
    }}
    
    .prediction-item:hover {{
        transform: translateY(-2px);
    }}
    
    /* Theme toggle button and controls row */
    .controls-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Helper function to create metric cards
def create_metric_card(title, value, suffix=None):
    metric_html = f"""
    <div class="metric-card">
        <h4 style="margin: 0; font-size: 0.85rem; color: {colors['text']};">{title}</h4>
        <h2 style="margin: 4px 0; font-size: 1.5rem; color: {colors['text']};">{value}</h2>
        {f'<p style="margin: 0; color: {colors["text"]}; font-size: 0.75rem;">{suffix}</p>' if suffix else ''}
    </div>
    """
    return metric_html

# Helper function to create stat cards
def create_stat_card(title, value, suffix=None, card_type="default"):
    gradient = colors['stat_default']
    if card_type == "spam":
        gradient = colors['stat_spam']
    elif card_type == "safe":
        gradient = colors['stat_safe']
    
    stat_html = f"""
    <div class="stat-card" style="background: {gradient};">
        <h3 style="margin: 0; font-size: 1rem;">{title}</h3>
        <h1 style="font-size: 2rem; margin: 6px 0;">{value}</h1>
        {f'<p style="margin: 0; font-size: 0.8rem;">{suffix}</p>' if suffix else ''}
    </div>
    """
    return stat_html

# Title section - Updated with new name
title_html = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px; 
            border-radius: 12px; 
            text-align: center;
            margin-top: -80px;
            margin-bottom: 20px; 
            box-shadow: 0 6px 18px rgba(0,0,0,0.15);
            width: 100%;">
    <h1 style="color: white; margin: 0px; font-size: 2.5rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">Spam Message Analyzer</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.1rem;">
        AI-powered system for detecting spam text messages
    </p>
</div>
"""
st.markdown(title_html, unsafe_allow_html=True)

# Add controls row with theme toggle and stats
col_left, col_center, col_right = st.columns([1, 2, 1])

# Stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuations = string.punctuation

# Load models (with error handling)
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory.")
        return None, None

model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.stop()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [word for word in text if word not in punctuations]
    text = [word for word in text if word.isalpha()]
    return ' '.join(text)

# Create tabs for different features - Removed About tab
tab1, tab2, tab3 = st.tabs(["🔍 Text Analysis", "📝 Batch Processing", "📈 Analytics"])

with tab1:
    st.markdown(
        f"""
        <div class="main-container">
            <h3 style="margin-top: 0; color: {colors['text']}; font-size: 1.2rem; font-weight: 600;">Analyze the Text</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Create two columns for content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input methods
        input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload File"], horizontal=True)
        
        user_input = ""
        if input_method == "Type/Paste Text":
            user_input = st.text_area(
                "Enter message to analyze:",
                height=220,
                placeholder="Type or paste your message here to check for spam content..."
            )
        else:
            uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
            if uploaded_file is not None:
                user_input = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", user_input, height=220, disabled=True)
         # Threshold setting
    
        threshold = st.slider("Spam Detection Threshold", 0.1, 0.9, 0.3, 0.05)
        st.caption(f"Current threshold: {threshold} - Lower values increase sensitivity")

    with col2:
        # Calculate word and character counts
        word_count = len(user_input.split()) if user_input else 0
        char_count = len(user_input) if user_input else 0
        
        # Metrics
        st.markdown(
            f"""
            <div class="main-container">
                <h3 style="margin-top: 3px;padding:5px color: {colors['text']}; font-size: 1.1rem; font-weight: 600;">Content Statistics</h3>
                {create_metric_card("Word Count", word_count)}

                {create_metric_card("Character Count", char_count)} </div>
            """, unsafe_allow_html=True
        )
        
       
    # Prediction button
    analyze_button = st.button("Analyze for Spam", type="primary", use_container_width=True)
    
    # Process analysis if button clicked
    if analyze_button:
        if user_input.strip() != "":
            with st.spinner('Analyzing message...'):
                time.sleep(0.5)  # Small delay for better UX
                
                # Preprocess
                processed_text = preprocess(user_input)
                X_input = vectorizer.transform([processed_text])
                
                # Predict probability for class 1 (Spam)
                proba = model.predict_proba(X_input)[0][1]
                
                # Prediction based on threshold
                prediction = 1 if proba >= threshold else 0
                label = "SPAM DETECTED" if prediction == 1 else "SAFE MESSAGE"
                
                # Update session state
                st.session_state.total_predictions += 1
                if prediction == 1:
                    st.session_state.spam_detected += 1
                
                # Add to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'text': user_input[:50] + "..." if len(user_input) > 50 else user_input,
                    'prediction': label,
                    'probability': proba,
                    'is_spam': prediction == 1
                })
                
                # Display results
                if prediction == 1:
                    result_html = f"""
                    <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24);
                                color: white; 
                                padding: 20px; 
                                border-radius: 10px; 
                                text-align: center;
                                margin: 0 0 16px 0; 
                                box-shadow: 0 6px 16px rgba(0,0,0,0.2);
                                width: 100%;">
                        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">SPAM DETECTED</h2>
                        <p style="font-size: 1.1rem; margin: 8px 0;">
                            Spam Probability: <strong>{proba*100:.2f}%</strong>
                        </p>
                        <p style="font-size: 0.9rem; margin-top: 10px; opacity: 0.9;">
                            This message appears to contain spam content. Exercise caution.
                        </p>
                    </div>
                    """
                else:
                    result_html = f"""
                    <div style="background: linear-gradient(135deg, #00b894, #00a085);
                                color: white; 
                                padding: 20px; 
                                border-radius: 10px; 
                                text-align: center;
                                margin: 0 0 16px 0; 
                                box-shadow: 0 6px 16px rgba(0,0,0,0.2);
                                width: 100%;">
                        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">SAFE MESSAGE</h2>
                        <p style="font-size: 1.1rem; margin: 8px 0;">
                            Spam Probability: <strong>{proba*100:.2f}%</strong>
                        </p>
                        <p style="font-size: 0.9rem; margin-top: 10px; opacity: 0.9;">
                            This message appears to be legitimate content.
                        </p>
                    </div>
                    """
                
                st.markdown(result_html, unsafe_allow_html=True)
                
                # Confidence gauge
                st.markdown(
                    f"""
                    <div class="main-container">
                        <h3 style="margin-top: 0; color: {colors['text']}; font-size: 1.2rem; font-weight: 600;">Confidence Analysis</h3>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Spam Confidence Level", 'font': {'size': 16}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'bgcolor': colors['secondary_bg'],
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig.update_layout(
                    height=250, 
                    font={'family': "Arial"},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    # Batch Analysis
    st.markdown(
        f"""
        <div class="main-container">
            <h3 style="margin-top: 0; color: {colors['text']}; font-size: 1.2rem; font-weight: 600;">Batch Message Analysis</h3>
            <p style="color: {colors['text']}; font-size: 0.95rem;">Analyze multiple messages simultaneously by entering them separated by new lines.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Settings row
    col1, col2 = st.columns([3, 1])
    
    with col2:
        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.3, 0.05, key="batch_threshold")
    
    with col1:
        batch_input = st.text_area(
            "Enter multiple messages (one per line):",
            height=180,
            placeholder="Message 1\nMessage 2\nMessage 3..."
        )
    
    # Interactive line-by-line analysis
    if st.button("Analyze Batch", type="primary"):
        if batch_input.strip():
            messages = [msg.strip() for msg in batch_input.split('\n') if msg.strip()]
            
            # Progress container
            st.markdown(
                f"""
                <div class="main-container">
                    <h3 style="margin-top: 0; color: {colors['text']}; font-size: 1.2rem; font-weight: 600;">Analysis Progress</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Process each message with visual feedback
            results = []
            
            for i, message in enumerate(messages):
                # Show processing status
                progress_bar.progress((i + 1) / len(messages))
                
                # Process the message
                processed_text = preprocess(message)
                X_input = vectorizer.transform([processed_text])
                proba = model.predict_proba(X_input)[0][1]
                prediction = 1 if proba >= threshold else 0
                
                results.append({
                    'message': message,
                    'is_spam': prediction == 1,
                    'probability': proba
                })
                
                # Display immediate result
                if prediction == 1:
                    message_html = f"""
                    <div class="prediction-item" style="background: rgba(255, 107, 107, 0.1); border-left: 4px solid #ff6b6b;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: 600; color: {colors['text']};">Message {i+1}</span>
                            <span style="color: #ff6b6b; font-weight: 600;">
                                SPAM ({proba*100:.1f}%)
                            </span>
                        </div>
                        <p style="margin: 6px 0 0 0; color: {colors['text']};">{message[:100]}{"..." if len(message) > 100 else ""}</p>
                    </div>
                    """
                else:
                    message_html = f"""
                    <div class="prediction-item" style="background: rgba(0, 184, 148, 0.1); border-left: 4px solid #00b894;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: 600; color: {colors['text']};">Message {i+1}</span>
                            <span style="color: #00b894; font-weight: 600;">
                                SAFE ({proba*100:.1f}%)
                            </span>
                        </div>
                        <p style="margin: 6px 0 0 0; color: {colors['text']};">{message[:100]}{"..." if len(message) > 100 else ""}</p>
                    </div>
                    """
                st.markdown(message_html, unsafe_allow_html=True)
                
                time.sleep(0.1)  # Shorter visual delay
            
            # Summary statistics
            spam_count = sum(1 for r in results if r['is_spam'])
            safe_count = len(results) - spam_count
            
            # Summary container
            st.markdown(
                f"""
                <div class="main-container">
                    <h3 style="margin-top: 0; color: {colors['text']}; font-size: 1.2rem; font-weight: 600;">Batch Summary</h3>
                </div>
                """, 
                unsafe_allow_html=True
                )
            
            # Apply the stat cards to columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(create_stat_card("Total Analyses", len(results), "Messages analyzed"), unsafe_allow_html=True)
            with col2:
                st.markdown(create_stat_card("Spam Detected", spam_count, f"{spam_count/len(results)*100:.1f}%", "spam"), unsafe_allow_html=True)
            with col3:
                st.markdown(create_stat_card("Safe Messages", safe_count, f"{safe_count/len(results)*100:.1f}%", "safe"), unsafe_allow_html=True)
            
            # Visualization container
            st.markdown(
                f"""
                <div class="main-container" style = "margin-top: 10px;">
                    <h3 style="margin-top: 0px; color: {colors['text']}; font-size: 1.2rem; font-weight: 600;">Results Visualization</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Add a pie chart for visual summary
            fig = px.pie(
                values=[spam_count, safe_count],
                names=['Spam', 'Safe'],                
                color_discrete_sequence=['#ff6b6b', '#00b894'],
                hole=0.5
            )
            fig.update_layout(
                legend=dict(yanchor="bottom", y=30.02, xanchor="center", x=0.63),
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Please enter some text to analyze.")

with tab3:
    # Analytics Tab
    st.markdown(
        f"""
        <div class="main-container">
            <h3 style="margin-top: 0; color: {colors['text']}; font-size: 1.2rem; font-weight: 600;">Analysis History & Trends</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    if st.session_state.prediction_history:
        # Top stats cards
        spam_count = sum(1 for pred in st.session_state.prediction_history if pred['is_spam'])
        safe_count = len(st.session_state.prediction_history) - spam_count
        
        # Apply the stat cards to columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(create_stat_card("Total Analyses", len(st.session_state.prediction_history), "Messages analyzed"), unsafe_allow_html=True)
        with col2:
            percentage = (spam_count/max(len(st.session_state.prediction_history), 1))*100
            st.markdown(create_stat_card("Spam Content", spam_count, f"{percentage:.1f}% of total", "spam"), unsafe_allow_html=True)
        with col3:
            percentage = (safe_count/max(len(st.session_state.prediction_history), 1))*100
            st.markdown(create_stat_card("Safe Content", safe_count, f"{percentage:.1f}% of total", "safe"), unsafe_allow_html=True)
        
        # Create a container for charts
        st.markdown(
            f"""
            <div class="main-container">
                <h3 style="margin-top: 0; color: {colors['text']}; font-size: 1.2rem; font-weight: 600;">Analysis Visualization</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Charts and visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=[spam_count, safe_count],
                names=['Spam', 'Safe'],
                title="Content Distribution",
                color_discrete_sequence=['#ff6b6b', '#00b894'],
                hole=0.4
            )
            fig_pie.update_layout(
                legend=dict( yanchor="bottom", y=1.02, xanchor="center", x=0.62),
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Histogram of confidence scores
            confidence_values = [pred['probability'] * 100 for pred in st.session_state.prediction_history]
            
            fig_hist = px.histogram(
                x=confidence_values,
                nbins=20,
                title="Spam Confidence Distribution",
                labels={'x': 'Confidence Score (%)', 'y': 'Count'},
                color_discrete_sequence=['#4b6cb7']
            )
            fig_hist.add_vline(x=threshold*100, line_dash="dash", line_color="red",
                             annotation_text=f"Threshold ({threshold*100}%)")
            fig_hist.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # History section
        st.markdown(
            f"""
            <div class="main-container">
                <h3 style="margin-top: 0; color: {colors['text']}; font-size: 1.2rem; font-weight: 600;">Recent Analyses</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Build history items
        for i, pred in enumerate(reversed(st.session_state.prediction_history[-10:])):
            if pred['is_spam']:
                history_html = f"""
                <div class="prediction-item" style="background: rgba(255, 107, 107, 0.1); border-left: 4px solid #ff6b6b;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: {"rgba(0,0,0,0.6)" if st.session_state.theme == "light" else "rgba(255,255,255,0.6)"}; font-size: 0.8rem;">{pred['timestamp']}</span>
                        <span style="color: #ff6b6b; font-weight: 600; background: rgba(255, 107, 107, 0.1); padding: 4px 8px; border-radius: 20px; display: inline-block; font-size: 0.85rem;">
                            SPAM ({pred['probability']*100:.1f}%)
                        </span>
                    </div>
                    <p style="margin: 8px 0 0 0; color: {colors['text']}; font-size: 0.95rem;">{pred['text']}</p>
                </div>
                """
            else:
                history_html = f"""
                <div class="prediction-item" style="background: rgba(0, 184, 148, 0.1); border-left: 4px solid #00b894;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: {"rgba(0,0,0,0.6)" if st.session_state.theme == "light" else "rgba(255,255,255,0.6)"}; font-size: 0.8rem;">{pred['timestamp']}</span>
                        <span style="color: #00b894; font-weight: 600; background: rgba(0, 184, 148, 0.1); padding: 4px 8px; border-radius: 20px; display: inline-block; font-size: 0.85rem;">
                            SAFE ({pred['probability']*100:.1f}%)
                        </span>
                    </div>
                    <p style="margin: 8px 0 0 0; color: {colors['text']}; font-size: 0.95rem;">{pred['text']}</p>
                </div>
                """
            st.markdown(history_html, unsafe_allow_html=True)
    else:
        # Info message
        st.markdown(f"""
        <div class="main-container" style="text-align: center; padding: 30px 20px;">
            <p style="color: {colors['text']}; font-size: 1.1rem;">No analysis history yet.</p>
            <p style="color: {colors['text']}; opacity: 0.7; font-size: 0.9rem;">Analyze messages in the Text Analysis tab to see results here</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    f"""<div style='text-align: center; color: {"gray" if st.session_state.theme == "light" else "#aaa"}; 
              font-size: 0.85rem; margin-top: 15px; margin-bottom:-80px; padding: 10px;
              width: 100%;'>
    Text Spam Analyzer
    <p >Made by <strong style="font-size: 1.0rem">Khizar Ishtiaq</strong> </p> </div>""", 
    unsafe_allow_html=True
)
# Streamlit app for Text Spam Analyzer
#           
# ```

