import streamlit as st
import os
from datetime import datetime
from io import BytesIO
from ai_inference import predict_image
from database import supabase

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Deepfake Detection System",
    layout="centered"
)

os.makedirs("uploads", exist_ok=True)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "show_review" not in st.session_state:
    st.session_state.show_review = False
if "selected_review" not in st.session_state:
    st.session_state.selected_review = None

# ---------------- AUTH FUNCTIONS ----------------
def create_user(username, password):
    return supabase.table("users").insert({
        "username": username,
        "password": password
    }).execute()

def login_user(username, password):
    result = (
        supabase.table("users")
        .select("*")
        .eq("username", username)
        .eq("password", password)
        .execute()
    )
    return result.data

# ---------------- LOGIN / SIGNUP PAGE ----------------
if not st.session_state.logged_in:

    st.title("🔐 Cloud Verification Access")
    st.caption("Sign up or login to continue")

    tab1, tab2 = st.tabs(["📝 Sign Up", "🔑 Login"])

    with tab1:
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Create Account"):
            if new_user and new_pass:
                create_user(new_user, new_pass)
                st.success("Account created! Please login.")
            else:
                st.warning("Enter username & password")

    with tab2:
        user = st.text_input("Username")
        passwd = st.text_input("Password", type="password")

        if st.button("Login"):
            data = login_user(user, passwd)

            if data:
                st.session_state.logged_in = True
                st.session_state.current_user = user
                st.session_state.user_id = data[0]["id"]
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# ---------------- MAIN APP ----------------
st.title("🕵️ Deepfake Fraud Detection System")
st.caption(f"Logged in as: **{st.session_state.current_user}**")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.current_user = None
    st.rerun()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image to verify",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    file_bytes = uploaded_file.getvalue()
    unique_name = f"{datetime.now().timestamp()}_{uploaded_file.name}"

    # save local copy (for model inference)
    path = os.path.join("uploads", uploaded_file.name)
    with open(path, "wb") as f:
        f.write(file_bytes)

    st.image(path, caption="Uploaded Image", width="stretch")

    if st.button("🔍 Verify Media", use_container_width=True):

        with st.spinner("Running AI verification..."):
            label, confidence = predict_image(path)

        st.subheader("Result")
        st.success(f"Prediction: {label}")
        st.progress(int(confidence * 100))
        st.caption(f"Confidence: {confidence*100:.2f}%")

        # -------- SUPABASE STORAGE UPLOAD (FIXED) --------
        file_stream = BytesIO(file_bytes)

        supabase.storage.from_("images").upload(
            path=unique_name,
            file=file_stream,
            file_options={"upsert": "true"}
        )

        # -------- SAVE HISTORY --------
        supabase.table("detection_history").insert({
            "user_id": st.session_state.user_id,
            "filename": unique_name,
            "prediction": label,
            "confidence": confidence
        }).execute()

# ---------------- SIDEBAR HISTORY ----------------
st.sidebar.subheader("📜 Detection History")

history = (
    supabase.table("detection_history")
    .select("*")
    .eq("user_id", st.session_state.user_id)
    .order("created_at", desc=True)
    .execute()
)

if history.data:

    file_options = [item["filename"] for item in history.data]

    selected = st.sidebar.selectbox("Select image", file_options)

    if st.sidebar.button("👁️ Review Selected"):
        st.session_state.show_review = True
        st.session_state.selected_review = selected

    if st.sidebar.button("🗑️ Clear History"):
        supabase.table("detection_history") \
            .delete() \
            .eq("user_id", st.session_state.user_id) \
            .execute()
        st.session_state.show_review = False
        st.session_state.selected_review = None
        st.rerun()

else:
    st.sidebar.caption("No detections yet")

# ---------------- REVIEW IMAGE ----------------
if st.session_state.show_review and st.session_state.selected_review:

    selected_item = next(
        item for item in history.data
        if item["filename"] == st.session_state.selected_review
    )

    st.subheader("📂 Detection Review")

    image_url = supabase.storage.from_("images").get_public_url(
        selected_item["filename"]
    )

    st.image(image_url, caption=selected_item["filename"], width="stretch")

    st.write(f"Prediction: **{selected_item['prediction']}**")
    st.write(f"Confidence: **{selected_item['confidence']*100:.2f}%**")
    st.write(f"Time: {selected_item['created_at']}")

    if st.button("❌ Close Review"):
        st.session_state.show_review = False
        st.session_state.selected_review = None
        st.rerun()

# ---------------- WARNING ----------------
st.warning(
    "Prediction confidence may vary as this is a prototype deepfake detection system."
)
