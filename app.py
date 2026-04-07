import streamlit as st

from backend.model_service import load_artifact, predict_message


st.set_page_config(
    page_title="Spect: Spam Detection System",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
<style>
    .main {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef3fb 100%);
    }
    .hero {
        background: white;
        border: 1px solid #e4e9f2;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 8px 25px rgba(23, 40, 80, 0.06);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #172037;
        margin-bottom: 0.4rem;
    }
    .hero-sub {
        color: #465269;
        margin-bottom: 0;
    }
    .result-card {
        border-radius: 14px;
        padding: 1rem;
        margin-top: 0.75rem;
        border: 1px solid transparent;
    }
    .result-spam {
        background: #fff2f2;
        border-color: #f8c9ce;
        color: #a2252f;
    }
    .result-ham {
        background: #eefaf2;
        border-color: #bdebc9;
        color: #1d7c3e;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <div class="hero-title">Spect: Spam Detection System</div>
    <p class="hero-sub">
        Production-style spam analysis with an optimized ML pipeline and confidence scores.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifact():
    return load_artifact()


try:
    artifact = get_artifact()
except Exception as exc:
    st.error(str(exc))
    st.info("Run `python train_model.py` once, then restart this app.")
    st.stop()

metrics = artifact.get("metrics", {})
best_params = artifact.get("best_params", {})
dataset_size = artifact.get("dataset_size", 0)

with st.sidebar:
    st.header("Model Summary")
    st.metric("Dataset size", f"{dataset_size}")
    st.metric("Train Accuracy", f"{metrics.get('train_accuracy', 0.0):.4f}")
    st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0.0):.4f}")
    st.metric("Accuracy Gap", f"{metrics.get('accuracy_gap', 0.0):.4f}")
    with st.expander("Best Hyperparameters"):
        st.json(best_params)

left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    st.subheader("Message Classifier")
    text = st.text_area(
        "Enter message",
        placeholder="Paste SMS, email snippet, or chat text...",
        height=220,
    )
    run_predict = st.button("Analyze Message", use_container_width=True)

with right_col:
    st.subheader("Quick Notes")
    st.markdown("- Class labels: `ham` and `spam`")
    st.markdown("- Scores are calibrated probabilities")
    st.markdown("- Best for short text content")

if run_predict:
    if not text.strip():
        st.warning("Please enter a message before running analysis.")
    else:
        result = predict_message(text, artifact)
        spam_prob = result["spam_probability"] * 100
        ham_prob = result["ham_probability"] * 100
        confidence = result["confidence"] * 100

        if result["label"] == "spam":
            st.markdown(
                f"""
                <div class="result-card result-spam">
                    <h3>Spam detected</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                    <p><strong>Spam score:</strong> {spam_prob:.2f}% | <strong>Ham score:</strong> {ham_prob:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-ham">
                    <h3>Likely legitimate message (ham)</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                    <p><strong>Ham score:</strong> {ham_prob:.2f}% | <strong>Spam score:</strong> {spam_prob:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.caption("Spect uses TF-IDF + Logistic Regression with hyperparameter tuning.")
