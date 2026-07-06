import sys
import streamlit as st
import pdfplumber
from Resume_scanner import compare


def extract_pdf_data(file):
    data = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                data += text
    return data


def extract_text_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# Command-line argument processing
if len(sys.argv) > 1:

    if len(sys.argv) == 3:
        resume_path = sys.argv[1]
        jd_path = sys.argv[2]

        resume_data = extract_pdf_data(resume_path)
        jd_data = extract_text_data(jd_path)

        compare([resume_data], jd_data, flag="HuggingFace-BERT")

    sys.exit()


# ---------------- Sidebar ---------------- #

st.set_page_config(page_title="Applicant Tracking System", page_icon="📄")

flag = "HuggingFace-BERT"

with st.sidebar:
    st.title("Settings")

    flag = st.selectbox(
        "Embedding Model",
        ["HuggingFace-BERT", "Doc2Vec"]
    )


# ---------------- Tabs ---------------- #

tab1, tab2 = st.tabs(["Home", "📊 Results"])


# Variables to avoid NameError
score = []
uploaded_files = []
comp_pressed = False


# ---------------- Home ---------------- #

with tab1:

    st.title("📄 Applicant Tracking System")

    uploaded_files = st.file_uploader(
        "Upload Resume(s) (PDF)",
        type="pdf",
        accept_multiple_files=True
    )

    JD = st.text_area(
        "Enter Job Description",
        height=200
    )

    comp_pressed = st.button("Compare")

    if comp_pressed:

        if not uploaded_files:
            st.error("Please upload at least one resume.")

        elif JD.strip() == "":
            st.error("Please enter a job description.")

        else:

            with st.spinner("Comparing resumes..."):

                resume_texts = [
                    extract_pdf_data(file)
                    for file in uploaded_files
                ]

                score = compare(
                    resume_texts,
                    JD,
                    flag
                )

            st.success("Comparison completed successfully!")

            st.info("Click the **Results** tab to view the ATS score.")


# ---------------- Results ---------------- #

with tab2:

    st.header("📊 ATS Results")

    if comp_pressed and score:

        for i in range(len(score)):

            score_value = float(score[i])

            with st.expander(uploaded_files[i].name, expanded=True):

                st.metric(
                    label="ATS Match Score",
                    value=f"{score_value:.2f}%"
                )

                st.progress(min(int(score_value), 100))

                if score_value >= 80:
                    st.success("Excellent Match ✅")

                elif score_value >= 60:
                    st.warning("Good Match 👍")

                else:
                    st.error("Needs Improvement ❌")

    else:
        st.info("Upload a resume and click Compare to view results.")
