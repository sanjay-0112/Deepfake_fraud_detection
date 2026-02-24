import streamlit as st
import os
from datetime import datetime
from ai_inference import predict_image
from database import supabase

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Deepfake Detection System", layout="centered")

os.makedirs("uploads", exist_ok=True)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "review_image" not in st.session_state:
    st.session_state.review_image = None

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
                st.success("Login successful")
                st.session_state.logged_in = True
                st.session_state.current_user = user
                st.session_state.user_id = data[0]["id"]
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

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image to verify",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    path = os.path.join("uploads", uploaded_file.name)

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(path, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Verify Media", use_container_width=True):

        with st.spinner("Running AI verification..."):
            label, confidence = predict_image(path)

        # ---------- SHOW RESULT ----------
        st.subheader("Result")
        st.success(f"Prediction: {label}")

        percent = confidence * 100
        st.progress(int(percent))
        st.caption(f"Confidence: {percent:.2f}%")

        # ---------- UPLOAD TO SUPABASE STORAGE ----------
        unique_name = f"{datetime.now().timestamp()}_{uploaded_file.name}"

        try:
            supabase.storage.from_("images").upload(
                unique_name,
                path
            )
        except Exception as e:
            st.warning("Image upload skipped (storage policy issue).")

        # ---------- SAVE HISTORY ----------
        supabase.table("detection_history").insert({
            "user_id": st.session_state.user_id,
            "filename": uploaded_file.name,
            "prediction": label,
            "confidence": confidence,
            "image_path": unique_name,
            "created_at": str(datetime.now())
        }).execute()

# ---------------- SIDEBAR HISTORY ----------------
st.sidebar.subheader("📜 Detection History")

history = supabase.table("detection_history") \
    .select("*") \
    .eq("user_id", st.session_state.user_id) \
    .order("created_at", desc=True) \
    .execute()

if history.data:

    if st.sidebar.button("🧹 Clear History (DB)"):
        supabase.table("detection_history") \
            .delete() \
            .eq("user_id", st.session_state.user_id) \
            .execute()
        st.rerun()

    for item in history.data[:10]:

        if st.sidebar.button(
            f"{item['filename']} ({item['prediction']})",
            key=item["id"]
        ):
            st.session_state.review_image = item

else:
    st.sidebar.caption("No detections yet")

# ---------------- REVIEW PANEL ----------------
if st.session_state.review_image:

    review = st.session_state.review_image

    st.divider()
    st.subheader("🧾 Review History Item")

    try:
        image_url = supabase.storage.from_("images").get_public_url(
            review["image_path"]
        )

        st.image(image_url, caption=review["filename"], use_container_width=True)

    except:
        st.warning("Image preview unavailable.")

    st.write(f"**Prediction:** {review['prediction']}")
    st.write(f"**Confidence:** {review['confidence']*100:.2f}%")
    st.write(f"**Timestamp:** {review['created_at']}")

    if st.button("❌ Close Review"):
        st.session_state.review_image = None
        st.rerun()

# ---------------- WARNING ----------------
st.warning(
    "Prediction confidence may vary as this is a prototype deepfake detection system."
)
