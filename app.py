import streamlit as st
import os
from datetime import datetime
from ai_inference import predict_image
from database import supabase

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Deepfake Fraud Detection",
    layout="wide",
    page_icon="🕵️"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #050914;
}
.block-container {
    padding-top: 2rem;
}

h1 {
    text-align: center;
    font-weight: 700;
}

.upload-box {
    background: #0f172a;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

os.makedirs("uploads", exist_ok=True)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ---------------- AUTH FUNCTIONS ----------------
def create_user(username, password):
    return supabase.table("users").insert({
        "username": username,
        "password": password
    }).execute()

def login_user(username, password):
    result = supabase.table("users") \
        .select("*") \
        .eq("username", username) \
        .eq("password", password) \
        .execute()
    return result.data

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.logged_in:

    st.title("🔐 Cloud Verification Access")
    st.caption("Sign up or login to continue")

    tab1, tab2 = st.tabs(["📝 Sign Up", "🔑 Login"])

    with tab1:
        new_user = st.text_input("Create Username", key="new_user")
        new_pass = st.text_input("Create Password", type="password", key="new_pass")

        if st.button("Create Account"):
            if new_user and new_pass:
                create_user(new_user, new_pass)
                st.success("Account created! Please login.")
            else:
                st.warning("Enter username & password")

    with tab2:
        user = st.text_input("Username", key="login_user")
        passwd = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            data = login_user(user, passwd)

            if data:
                st.success("Login successful")
                st.session_state.logged_in = True
                st.session_state.current_user = user
                st.session_state.user_id = data[0]["id"]
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# ---------------- MAIN APP ----------------
st.markdown("<h1>🕵️ Deepfake Fraud Detection System</h1>",
            unsafe_allow_html=True)

st.caption(f"Logged in as: **{st.session_state.current_user}**")
st.markdown("---")

# Logout
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.current_user = None
    st.rerun()

# ---------------- UPLOAD SECTION ----------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an image to verify",
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

        st.markdown("### 🔎 Result")

        if label.lower() == "real":
            st.success(f"✅ Prediction: {label}")
        else:
            st.error(f"⚠️ Prediction: {label}")

        st.progress(int(confidence * 100))
        st.caption(f"Confidence: {confidence*100:.2f}%")

        # SAVE TO SUPABASE
        supabase.table("detection_history").insert({
            "user_id": st.session_state.user_id,
            "filename": uploaded_file.name,
            "prediction": label,
            "confidence": confidence
        }).execute()

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR HISTORY ----------------
st.sidebar.subheader("📜 Detection History")

history = supabase.table("detection_history") \
    .select("*") \
    .eq("user_id", st.session_state.user_id) \
    .order("created_at", desc=True) \
    .execute()

if history.data:
    for item in history.data[:5]:
        st.sidebar.write(
            f"{item['created_at']} | "
            f"{item['filename']} → {item['prediction']} "
            f"({item['confidence']*100:.1f}%)"
        )
else:
    st.sidebar.caption("No detections yet")

st.warning(
    "Prediction confidence may vary as this is a prototype deepfake detection system."
)
