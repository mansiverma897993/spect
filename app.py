import streamlit as st
import pickle
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Spam Detector App",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS (Clean UI) ---
st.markdown("""
<style>
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e6ed;
        transition: border 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 5px rgba(74, 144, 226, 0.5);
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #4a90e2;
        color: white;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #357abd;
        box-shadow: 0 4px 10px rgba(74,144,226,0.3);
        transform: translateY(-2px);
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .logo-container h1 {
        margin: 0;
        padding: 0;
        font-size: 2.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header with Beautiful Logo ---
st.markdown("""
<div class="logo-container">
    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3z" fill="#4a90e2"/>
        <path d="M12 11.99h6c-.46 4.1-2.92 7.74-6 8.92v-8.92z" fill="#357abd"/>
        <path d="M10.5 16.5l-4-4 1.41-1.41L10.5 13.67l6.09-6.09L18 9l-7.5 7.5z" fill="#ffffff"/>
    </svg>
    <h1>Spam Detection System</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("Enter a message below to check if it's **Spam** or **Not Spam (Ham)**. Powered by Machine Learning via TF-IDF & Logistic Regression.")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("Model file `spam_model.pkl` not found! Please run `python train_model.py` first to generate it.")
    st.stop()

# --- User Input ---
message = st.text_area("✍️ Message Content", placeholder="Type or paste your message here...", height=150)

# --- Prediction Logic ---
if st.button("Detect Spam 🔍"):
    if message.strip() == "":
        st.warning("Please enter a message to classify!")
    else:
        with st.spinner("Analyzing message..."):
            time.sleep(0.5) 
            
            prediction = model.predict([message])[0]
            probability = model.predict_proba([message])[0]
            
            if prediction == 1:
                prob = probability[1] * 100
                st.markdown(f"""
                <div class="result-box" style="background-color: #ffe5e5; color: #d32f2f; border: 1px solid #ffcdd2;">
                    🚨 SPAM DETECTED! <br><span style="font-size: 0.9rem; font-weight: normal;">(Confidence: {prob:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                prob = probability[0] * 100
                st.markdown(f"""
                <div class="result-box" style="background-color: #e8f5e9; color: #388e3c; border: 1px solid #c8e6c9;">
                    ✅ NOT SPAM (HAM) <br><span style="font-size: 0.9rem; font-weight: normal;">(Confidence: {prob:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & Scikit-Learn | Local ML Project")
