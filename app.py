import streamlit as st
import os
import json
import time
from datetime import datetime
from ai_inference import predict_image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Deepfake Fraud Detection",
    page_icon="🕵️",
    layout="centered"
)

# ---------------- GLOBAL STYLES ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 1rem;
}
.card {
    background: #161b22;
    padding: 1.5rem;
    border-radius: 14px;
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
}
.center {
    text-align: center;
}
.small-text {
    color: #9aa4b2;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FILE SYSTEM ----------------
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

USERS_FILE = "results/users.json"
HISTORY_FILE = "results/history.json"

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "verifying" not in st.session_state:
    st.session_state.verifying = False

# ================= AUTH PAGE =================
if not st.session_state.logged_in:
    st.markdown("<h1 class='center'>🔐 Cloud Verification Access</h1>", unsafe_allow_html=True)
    st.markdown("<p class='center small-text'>Secure AI-powered deepfake detection platform</p>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📝 Sign Up", "🔑 Login"])

        # -------- SIGN UP --------
        with tab1:
            new_user = st.text_input("Username")
            new_pass = st.text_input("Password", type="password")

            if st.button("Create Account", use_container_width=True):
                with open(USERS_FILE, "r") as f:
                    users = json.load(f)

                if not new_user or not new_pass:
                    st.warning("Please fill all fields")
                elif new_user in users:
                    st.error("Username already exists")
                else:
                    users[new_user] = new_pass
                    with open(USERS_FILE, "w") as f:
                        json.dump(users, f, indent=2)

                    st.success("Account created successfully")
                    st.session_state.logged_in = True
                    st.session_state.current_user = new_user
                    st.session_state.verifying = True
                    st.rerun()

        # -------- LOGIN --------
        with tab2:
            user = st.text_input("Username", key="login_user")
            passwd = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login", use_container_width=True):
                with open(USERS_FILE, "r") as f:
                    users = json.load(f)

                if user in users and users[user] == passwd:
                    st.success("Login successful")
                    st.session_state.logged_in = True
                    st.session_state.current_user = user
                    st.session_state.verifying = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ================= VERIFICATION SCREEN =================
if st.session_state.verifying:
    st.markdown("<h2 class='center'>☁️ Initializing Cloud Verification</h2>", unsafe_allow_html=True)
    with st.spinner("Authenticating & loading AI model..."):
        time.sleep(2)
    st.session_state.verifying = False
    st.rerun()

# ================= SIDEBAR =================
st.sidebar.markdown("### 👤 Session")
st.sidebar.write(f"**User:** {st.session_state.current_user}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 My History")

with open(HISTORY_FILE, "r") as f:
    history = json.load(f)

user_history = [h for h in history if h["user"] == st.session_state.current_user]

if user_history:
    for item in reversed(user_history[-5:]):
        st.sidebar.caption(
            f"{item['timestamp']} | {item['prediction']} ({item['confidence']*100:.1f}%)"
        )
else:
    st.sidebar.caption("No detections yet")

if st.sidebar.button("🗑️ Clear My History"):
    history = [h for h in history if h["user"] != st.session_state.current_user]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    st.sidebar.success("History cleared")
    st.rerun()

# ================= MAIN APP =================
st.markdown("<h1 class='center'>🕵️ Deepfake Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='center small-text'>Upload an image to verify authenticity using AI</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        path = os.path.join("uploads", uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(path, caption="Uploaded Image", width="stretch")

        if st.button("🔍 Verify Media", use_container_width=True):
            with st.spinner("Running AI verification..."):
                label, confidence = predict_image(path)

            st.markdown("### Result")
            st.success(f"**Prediction:** {label}")
            st.progress(int(confidence * 100))
            st.caption(f"Confidence: {confidence*100:.2f}%")

            record = {
                "user": st.session_state.current_user,
                "filename": uploaded_file.name,
                "prediction": label,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(HISTORY_FILE, "r+") as f:
                history = json.load(f)
                history.append(record)
                f.seek(0)
                json.dump(history, f, indent=2)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<p class='center small-text'>⚠ Prototype system — accuracy improves with further training</p>",
    unsafe_allow_html=True
)
