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
if "show_review" not in st.session_state:
    st.session_state.show_review = False
if "selected_review" not in st.session_state:
    st.session_state.selected_review = None


# ---------------- AUTH ----------------
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


# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.title("🔐 Cloud Verification Access")
    st.caption("Sign up or login to continue")

    tab1, tab2 = st.tabs(["📝 Sign Up", "🔑 Login"])

    with tab1:
        u = st.text_input("Create Username")
        p = st.text_input("Create Password", type="password")
        if st.button("Create Account"):
            create_user(u, p)
            st.success("Account created!")

    with tab2:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            data = login_user(u, p)
            if data:
                st.session_state.logged_in = True
                st.session_state.user_id = data[0]["id"]
                st.session_state.current_user = u
                st.rerun()

    st.stop()

# ---------------- MAIN APP ----------------
st.title("🕵️ Deepfake Fraud Detection System")
st.caption(f"Logged in as: **{st.session_state.current_user}**")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if uploaded_file:

    file_bytes = uploaded_file.getvalue()
    unique_name = f"{datetime.now().timestamp()}_{uploaded_file.name}"

    path = os.path.join("uploads", uploaded_file.name)

    with open(path, "wb") as f:
        f.write(file_bytes)

    st.image(path)

    if st.button("🔍 Verify Media"):

        label, confidence = predict_image(path)

        st.success(f"Prediction: {label}")
        st.progress(int(confidence*100))

        # -------- FIXED STORAGE UPLOAD --------
        supabase.storage.from_("images").upload(
            unique_name,
            path,
            file_options={"upsert": "true"}
        )

        # save DB history
        supabase.table("detection_history").insert({
            "user_id": st.session_state.user_id,
            "filename": unique_name,
            "prediction": label,
            "confidence": confidence
        }).execute()

# ---------------- HISTORY ----------------
st.sidebar.subheader("📜 Detection History")

history = supabase.table("detection_history") \
    .select("*") \
    .eq("user_id", st.session_state.user_id) \
    .order("created_at", desc=True) \
    .execute()

if history.data:

    files = [i["filename"] for i in history.data]

    selected = st.sidebar.selectbox("Select image", files)

    if st.sidebar.button("👁️ Review Selected"):
        st.session_state.show_review = True
        st.session_state.selected_review = selected

    if st.sidebar.button("🗑️ Clear History"):
        supabase.table("detection_history") \
            .delete() \
            .eq("user_id", st.session_state.user_id) \
            .execute()
        st.rerun()

if st.session_state.show_review:

    item = next(
        i for i in history.data
        if i["filename"] == st.session_state.selected_review
    )

    st.subheader("📂 Detection Review")

    url = supabase.storage.from_("images").get_public_url(
        item["filename"]
    )

    st.image(url)
    st.write(item["prediction"])
    st.write(item["confidence"])

    if st.button("❌ Close Review"):
        st.session_state.show_review = False
        st.rerun()
