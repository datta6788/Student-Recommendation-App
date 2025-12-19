import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =====================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =====================================================
st.set_page_config(
    page_title="Student Recommendation System",
    page_icon="🎓",
    layout="centered"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background-color: lightyellow;
}
h1 {
    text-align: center;
    color: #1f2937;
}
h2, h3 {
    color: #2563eb;
}
.block-container {
    padding: 2rem;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("students.csv")

# =====================================================
# MACHINE LEARNING (TRAIN ONCE)
# =====================================================
X = df[["maths", "science", "english", "attendance"]]
y = df["result"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# =====================================================
# RESOURCES
# =====================================================
resources = {
    "Maths": "https://www.khanacademy.org/math",
    "Science": "https://www.khanacademy.org/science",
    "English": "https://www.khanacademy.org/grammar"
}

# =====================================================
# TITLE
# =====================================================
st.title("🎓 Student Recommendation System")
st.subheader("Machine Learning–Based Academic Risk Prediction")
st.markdown("---")

# =====================================================
# STUDENT SELECTION
# =====================================================
student_name = st.selectbox("Select Student", df["name"])
student = df[df["name"] == student_name].iloc[0]

# =====================================================
# ATTENDANCE (FLOAT SAFE)
# =====================================================
attendance = float(student["attendance"])

st.subheader("📌 Attendance")
st.write(f"**{attendance:.1f}%**")

# Attendance advisory (ALWAYS SHOWN)
st.subheader("⚠️ Attendance Advisory")
if attendance < 75:
    st.warning(
        f"Your attendance is {attendance:.1f}%. "
        "It is recommended to maintain at least 75% attendance."
    )
else:
    st.success("Your attendance meets the recommended requirement.")

# =====================================================
# MARKS TABLE
# =====================================================
st.subheader("📊 Marks")

marks_df = pd.DataFrame(
    {
        "Marks": [
            float(student["maths"]),
            float(student["science"]),
            float(student["english"])
        ]
    },
    index=["Maths", "Science", "English"]
)

st.dataframe(marks_df)

# =====================================================
# SUBJECTS TO FOCUS (INFORMATIONAL)
# =====================================================
st.subheader("📚 Subjects to Focus On")

focus_subjects = marks_df[marks_df["Marks"] < 70].index.tolist()

if focus_subjects:
    for subject in focus_subjects:
        st.write(f"🔹 **{subject}** → [Learn Here]({resources[subject]})")
else:
    st.success("No weak subjects 🎉")

# =====================================================
# PREDICT ONCE (CRITICAL FIX)
# =====================================================
scaled_input = scaler.transform([[
    student["maths"],
    student["science"],
    student["english"],
    student["attendance"]
]])

prediction = int(model.predict(scaled_input)[0])
risk_probability = model.predict_proba(scaled_input)[0][1]

# =====================================================
# ML RESULT
# =====================================================
st.subheader("🤖 ML Academic Risk Prediction")

if prediction == 1:
    st.error("⚠️ AT RISK (Predicted FAIL)")
else:
    st.success("✅ SAFE (Predicted PASS)")

st.write(f"📊 **Risk Probability:** {risk_probability * 100:.2f}%")

# =====================================================
# REASON FOR RISK (FIXED LOGIC)
# =====================================================
st.subheader("🧠 Reason for Risk")

reasons = []

if attendance < 75:
    reasons.append(f"Low attendance ({attendance:.1f}%)")

for subject, mark in marks_df["Marks"].items():
    if mark < 70:
        reasons.append(
            f"Weak performance in {subject} ({mark:.1f} marks)"
        )

if prediction == 1:
    if reasons:
        for r in reasons:
            st.write(f"🔴 {r}")
    else:
        st.write(
            "⚠️ Risk detected due to a combination of moderate academic factors, "
            "even though no single factor is critically low."
        )
else:
    st.write("✅ No major risk factors detected.")

# =====================================================
# HOW TO REDUCE THE RISK (ALWAYS SHOWN)
# =====================================================
st.subheader("🛠️ How to Reduce the Risk")

actions = []

if attendance < 75:
    actions.append(
        "Increase attendance to at least 75% by attending all remaining classes."
    )

for subject, mark in marks_df["Marks"].items():
    if mark < 70:
        actions.append(
            f"Focus on improving {subject} through regular practice and revision."
        )

if actions:
    for a in actions:
        st.write(f"✅ {a}")
else:
    st.write(
        "✅ Continue current study routine and maintain attendance and performance."
    )
