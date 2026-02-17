import streamlit as st
import pandas as pd
import random
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PCOS Prediction & Lifestyle Advisor",
    layout="wide",
    page_icon="🩺"
)

# --------------------------------------------------
# DARK THEME CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}
.stButton>button {
    background-color: #E94560;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #FF2E63;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# STEP CONTROLLER
# --------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned-Data.csv")

    df["PCOS"] = df["PCOS"].astype(str).str.lower().map({"yes": 1, "no": 0})
    df.dropna(subset=["PCOS"], inplace=True)

    yes_no_cols = [
        "Family_History_PCOS", "Menstrual_Irregularity",
        "Hormonal_Imbalance", "Hirsutism", "Mental_Health",
        "Insulin_Resistance", "Diabetes", "Smoking"
    ]

    for col in yes_no_cols:
        df[col] = df[col].astype(str).str.lower().map({"yes": 1, "no": 0})

    df["Age"] = df["Age"].astype(str).str.extract(r"(\d+)").astype(float)

    df["Sleep_Hours"] = df["Sleep_Hours"].replace({
        "Less than 6 hours": 5,
        "6-8 hours": 7,
        "9-12 hours": 10.5,
        "More than 8 hours": 9
    })

    df["Exercise_Duration"] = df["Exercise_Duration"].replace({
        "Less than 30 minutes": 15,
        "30 minutes": 30,
        "30 minutes to 1 hour": 45,
        "More than 1 hour": 75
    })

    features = [
        "Age", "Weight_kg", "Sleep_Hours", "Exercise_Duration",
        "Family_History_PCOS", "Menstrual_Irregularity",
        "Hormonal_Imbalance", "Hirsutism", "Mental_Health",
        "Insulin_Resistance", "Diabetes", "Smoking"
    ]

    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df[features] = df[features].fillna(df[features].median())

    return df, features


df, features = load_data()
X = df[features]
y = df["PCOS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# --------------------------------------------------
# FEATURE IMPORTANCE (SIDEBAR)
# --------------------------------------------------
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")
st.sidebar.subheader("🔬 Feature Importance")

fig_importance = go.Figure(
    go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation='h',
        marker_color="#E94560"
    )
)

fig_importance.update_layout(height=400)
st.sidebar.plotly_chart(fig_importance, use_container_width=True)

# --------------------------------------------------
# PDF GENERATOR
# --------------------------------------------------
def generate_pdf(user_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("PCOS Health Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Patient Name: {user_name}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Probability: {st.session_state.prob:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"BMI: {st.session_state.bmi:.2f}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Recommended Diet Plan:", styles["Heading2"]))

    for food in st.session_state.food_list:
        elements.append(Paragraph(f"- {food}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================================================
# STEP 1 – NAME
# ==================================================
if st.session_state.step == 1:

    st.markdown("""
    <div style="background-color:#1C1F26;padding:20px;border-radius:15px">
    <h1 style="color:#E94560;">🩺 PCOS Lifestyle & Health Advisor</h1>
    <p>AI Powered Health Risk Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    user_name = st.text_input("Enter Your Name")

    if st.button("Continue ➡️") and user_name.strip():
        st.session_state.user_name = user_name
        st.session_state.step = 2
        st.rerun()

# ==================================================
# STEP 2 – WELCOME
# ==================================================
elif st.session_state.step == 2:

    st.title(f"🌸 Welcome {st.session_state.user_name}!")
    st.write("This AI system will analyze your health and give personalized advice.")

    if st.button("Start Assessment 🧾"):
        st.session_state.step = 3
        st.rerun()

# ==================================================
# STEP 3 – INPUT
# ==================================================
elif st.session_state.step == 3:

    st.header("🧾 Enter Your Health Details")

    age = st.slider("Age", 15, 45, 25)
    weight = st.slider("Weight (kg)", 35, 120, 60)
    height = st.slider("Height (cm)", 140, 180, 160)
    sleep = st.slider("Sleep Hours", 4, 10, 7)
    exercise_minutes = st.slider("Exercise Duration", 0, 90, 30)

    family = st.selectbox("Family History of PCOS", ["No", "Yes"])
    menstrual = st.selectbox("Menstrual Irregularity", ["No", "Yes"])
    hormonal = st.selectbox("Hormonal Imbalance", ["No", "Yes"])
    hirsutism = st.selectbox("Hirsutism", ["No", "Yes"])
    mental = st.selectbox("Mental Health Issues", ["No", "Yes"])
    insulin = st.selectbox("Insulin Resistance", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    smoking = st.selectbox("Smoking", ["No", "Yes"])

    if st.button("Predict 🔍"):

        bmi = weight / ((height / 100) ** 2)

        input_df = pd.DataFrame([{
            "Age": age,
            "Weight_kg": weight,
            "Sleep_Hours": sleep,
            "Exercise_Duration": exercise_minutes,
            "Family_History_PCOS": 1 if family == "Yes" else 0,
            "Menstrual_Irregularity": 1 if menstrual == "Yes" else 0,
            "Hormonal_Imbalance": 1 if hormonal == "Yes" else 0,
            "Hirsutism": 1 if hirsutism == "Yes" else 0,
            "Mental_Health": 1 if mental == "Yes" else 0,
            "Insulin_Resistance": 1 if insulin == "Yes" else 0,
            "Diabetes": 1 if diabetes == "Yes" else 0,
            "Smoking": 1 if smoking == "Yes" else 0
        }])[features]

        st.session_state.pred = model.predict(input_df)[0]
        st.session_state.prob = model.predict_proba(input_df)[0][1] * 100
        st.session_state.bmi = bmi
        st.session_state.step = 4
        st.rerun()

# ==================================================
# STEP 4 – RESULT
# ==================================================
elif st.session_state.step == 4:

    if st.session_state.pred == 1:
        st.error(f"⚠️ High PCOS Risk ({st.session_state.prob:.2f}%)")
    else:
        st.success(f"✅ Low PCOS Risk ({st.session_state.prob:.2f}%)")

    # Risk Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.prob,
        title={'text': "PCOS Risk Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#E94560"},
            'steps': [
                {'range': [0, 30], 'color': "#00C49A"},
                {'range': [30, 70], 'color': "#FFA500"},
                {'range': [70, 100], 'color': "#FF4B4B"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # BMI Chart
    st.write(f"Your BMI: {st.session_state.bmi:.2f}")
    bmi = st.session_state.bmi

    bmi_df = pd.DataFrame({
        "Category": ["Underweight", "Normal", "Overweight", "Obese"],
        "BMI Limit": [18.5, 24.9, 29.9, 35]
    })

    fig_bmi = go.Figure(go.Bar(
        x=bmi_df["Category"],
        y=bmi_df["BMI Limit"],
        marker_color="#E94560"
    ))

    fig_bmi.add_shape(
        type="line",
        x0=-0.5,
        x1=3.5,
        y0=bmi,
        y1=bmi,
        line=dict(color="yellow", width=4)
    )

    fig_bmi.update_layout(title="BMI Category Indicator")
    st.plotly_chart(fig_bmi, use_container_width=True)

    if st.button("Next ➡️ Exercise Plan"):
        st.session_state.step = 5
        st.rerun()

# ==================================================
# STEP 5 – EXERCISE
# ==================================================
# ==================================================
# STEP 5 – ADVANCED WEEKLY EXERCISE PLANNER
# ==================================================
elif st.session_state.step == 5:

    st.subheader("🏃 PCOS-Friendly Weekly Exercise Planner")

    level = st.radio("Select Your Fitness Level", ["Beginner", "Intermediate", "Advanced"])

    # PCOS-Focused Exercise Pools
    exercise_pool = {
        "Beginner": {
            "Cardio": [
                "🚶 Brisk Walking (25 mins) – Improves insulin sensitivity",
                "🚴 Light Cycling (20 mins) – Supports weight balance"
            ],
            "Strength": [
                "🧘 Bodyweight Squats (2x12) – Hormonal balance support",
                "🧍 Wall Push-ups (2x10) – Metabolism boost"
            ],
            "Flexibility": [
                "🌸 PCOS Yoga Flow (20 mins) – Stress reduction",
                "🧘 Stretching Routine (15 mins) – Cortisol control"
            ]
        },
        "Intermediate": {
            "Cardio": [
                "🏃 Jogging (30 mins) – Fat metabolism boost",
                "💃 Dance Workout (30 mins) – Hormone regulation"
            ],
            "Strength": [
                "🔥 Lunges (3x12) – Lower body strength",
                "💪 Plank (3x30 sec) – Core stability"
            ],
            "Flexibility": [
                "🧘 Yoga Flow (25 mins) – Stress & cortisol reduction",
                "🌿 Mobility Drills (20 mins)"
            ]
        },
        "Advanced": {
            "Cardio": [
                "⚡ HIIT (30 mins) – Insulin resistance improvement",
                "🏃 Sprint Intervals (25 mins)"
            ],
            "Strength": [
                "🔥 Burpees (3x15) – Fat burning",
                "🏋 Weighted Squats (3x12)"
            ],
            "Flexibility": [
                "🧘 Power Yoga (30 mins)",
                "🌿 Deep Stretching (25 mins)"
            ]
        }
    }

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def generate_week_plan():
        plan = []
        pool = exercise_pool[level]

        for i, day in enumerate(days):
            if day == "Sunday":
                workout = "🛌 Active Recovery – Light Yoga / Meditation"
            elif i % 3 == 0:
                workout = random.choice(pool["Cardio"])
            elif i % 3 == 1:
                workout = random.choice(pool["Strength"])
            else:
                workout = random.choice(pool["Flexibility"])

            plan.append(workout)

        return plan

    # Initialize weekly plan
    if "weekly_plan" not in st.session_state:
        st.session_state.weekly_plan = generate_week_plan()

    if "progress_tracker" not in st.session_state:
        st.session_state.progress_tracker = [False] * 7

    st.markdown("### 🌸 Weekly Workout Cards")

    # CARD LAYOUT
    for i, day in enumerate(days):

        st.markdown(f"""
        <div style="
            background-color:#1C1F26;
            padding:20px;
            border-radius:15px;
            margin-bottom:15px;
            border-left:6px solid #E94560;">
            <h4 style="color:#E94560;">{day}</h4>
            <p>{st.session_state.weekly_plan[i]}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        # Mark complete
        with col1:
            completed = st.checkbox(
                "Mark as Done ✅",
                value=st.session_state.progress_tracker[i],
                key=f"progress_{i}"
            )
            st.session_state.progress_tracker[i] = completed

        # Change exercise
        with col2:
            if st.button("Change Exercise 🔄", key=f"change_{i}"):
                pool = exercise_pool[level]
                category = random.choice(["Cardio", "Strength", "Flexibility"])
                st.session_state.weekly_plan[i] = random.choice(pool[category])
                st.rerun()

    # Weekly Progress Calculation
    completed_days = sum(st.session_state.progress_tracker)
    progress_percent = int((completed_days / 7) * 100)

    st.markdown("### 📊 Weekly Progress Tracker")
    st.progress(progress_percent)
    st.write(f"Completion: {progress_percent}% ({completed_days}/7 days)")

    # Regenerate entire plan
    if st.button("🔄 Regenerate Full Week Plan"):
        st.session_state.weekly_plan = generate_week_plan()
        st.session_state.progress_tracker = [False] * 7
        st.rerun()

    if st.button("Next ➡️ Diet Plan"):
        st.session_state.step = 6
        st.rerun()

# ==================================================
# STEP 6 – DIET
# ==================================================
# ==================================================
# STEP 6 – PCOS WEEKLY FOOD PLANNER
# ==================================================
elif st.session_state.step == 6:

    st.subheader("🥗 PCOS-Friendly Weekly Food Planner")

    preference = st.radio("Select Diet Preference",
                          ["Vegetarian", "Non-Vegetarian"])

    # ----------------------------------------------
    # PCOS-Friendly Meal Database
    # ----------------------------------------------

    food_db = {
        "Vegetarian": {
            "Breakfast": [
                "🥣 Oats with Chia & Nuts – High fiber, controls insulin",
                "🥗 Vegetable Upma – Low GI carbs",
                "🍓 Greek Yogurt + Berries – Protein rich",
                "🥑 Avocado Toast (Multigrain)"
            ],
            "Lunch": [
                "🍛 Brown Rice + Dal + Veggies",
                "🥗 Quinoa Salad + Paneer",
                "🌯 Multigrain Roti + Sabzi + Curd",
                "🥦 Millet Bowl + Stir Fry Veggies"
            ],
            "Dinner": [
                "🥣 Vegetable Soup + Sprouts Salad",
                "🥗 Paneer Stir Fry + Salad",
                "🥑 Grilled Tofu + Sauteed Veggies",
                "🌮 Lettuce Wraps + Beans"
            ],
            "Snacks": [
                "🥜 Almonds & Walnuts",
                "🍎 Apple + Peanut Butter",
                "🥕 Carrot & Hummus",
                "🍵 Green Tea + Roasted Chana"
            ]
        },

        "Non-Vegetarian": {
            "Breakfast": [
                "🍳 Boiled Eggs + Multigrain Toast",
                "🥣 Oats + Nuts + Seeds",
                "🍓 Greek Yogurt + Berries",
                "🥑 Avocado Egg Toast"
            ],
            "Lunch": [
                "🍗 Grilled Chicken + Brown Rice",
                "🐟 Fish Curry + Millet",
                "🥗 Chicken Salad + Olive Oil",
                "🍛 Egg Curry + Multigrain Roti"
            ],
            "Dinner": [
                "🍲 Chicken Soup + Veggies",
                "🐟 Grilled Fish + Salad",
                "🥗 Egg Bhurji + Sauteed Veggies",
                "🍗 Stir Fry Chicken Bowl"
            ],
            "Snacks": [
                "🥜 Mixed Nuts",
                "🥚 Boiled Egg",
                "🍎 Apple + Peanut Butter",
                "🍵 Green Tea + Seeds Mix"
            ]
        }
    }

    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]

    def generate_food_plan():
        weekly_plan = []

        for day in days:
            daily_plan = {
                "Breakfast": random.choice(food_db[preference]["Breakfast"]),
                "Lunch": random.choice(food_db[preference]["Lunch"]),
                "Dinner": random.choice(food_db[preference]["Dinner"]),
                "Snacks": random.choice(food_db[preference]["Snacks"]),
            }
            weekly_plan.append(daily_plan)

        return weekly_plan

    # Initialize plan
    if "weekly_food_plan" not in st.session_state:
        st.session_state.weekly_food_plan = generate_food_plan()

    st.markdown("### 🌸 Weekly Meal Cards")

    # ----------------------------------------------
    # CARD LAYOUT
    # ----------------------------------------------

    for i, day in enumerate(days):

        st.markdown(f"""
        <div style="
            background-color:#1F2937;
            padding:20px;
            border-radius:15px;
            margin-bottom:20px;
            border-left:6px solid #10B981;">
            <h4 style="color:#10B981;">{day}</h4>
        </div>
        """, unsafe_allow_html=True)

        meals = st.session_state.weekly_food_plan[i]

        for meal_type in ["Breakfast", "Lunch", "Dinner", "Snacks"]:

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**{meal_type}:** {meals[meal_type]}")

            with col2:
                if st.button(f"🔄 Change", key=f"{day}_{meal_type}"):

                    new_meal = random.choice(
                        food_db[preference][meal_type]
                    )

                    st.session_state.weekly_food_plan[i][meal_type] = new_meal
                    st.rerun()

        st.markdown("---")

    # Regenerate entire week
    if st.button("🔄 Regenerate Full Week Plan"):
        st.session_state.weekly_food_plan = generate_food_plan()
        st.rerun()

    if st.button("Next ➡️ Summary & Download"):
        st.session_state.step = 7
        st.rerun()

# ==================================================
# STEP 7 – REPORT
# ==================================================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak
from reportlab.platypus import Frame
from reportlab.platypus import PageTemplate
from reportlab.platypus import BaseDocTemplate
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4

# ==================================================
# STEP 7 – SUMMARY & DOWNLOAD
# ==================================================
elif st.session_state.step == 7:

    st.title("📄 PCOS Health Summary Report")

    # Show Summary On Screen
    st.subheader("🧾 Patient Summary")

    st.write(f"**Name:** {st.session_state.user_name}")
    st.write(f"**PCOS Risk Probability:** {st.session_state.prob:.2f}%")
    st.write(f"**BMI:** {st.session_state.bmi:.2f}")

    if st.session_state.pred == 1:
        st.error("⚠️ High PCOS Risk")
    else:
        st.success("✅ Low PCOS Risk")

    st.markdown("---")

    # Exercise Plan Preview
    st.subheader("🏃 Weekly Exercise Plan")

    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]

    if "weekly_plan" in st.session_state:
        for i, day in enumerate(days):
            st.write(f"**{day}:** {st.session_state.weekly_plan[i]}")

    st.markdown("---")

    # Food Plan Preview
    st.subheader("🥗 Weekly Food Plan")

    if "weekly_food_plan" in st.session_state:
        for i, day in enumerate(days):
            st.write(f"### {day}")
            meals = st.session_state.weekly_food_plan[i]
            for meal_type, meal in meals.items():
                st.write(f"- **{meal_type}:** {meal}")

    st.markdown("---")

    # Generate PDF
    pdf_file = generate_pdf(st.session_state.user_name)

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_file,
        file_name="PCOS_Health_Report.pdf",
        mime="application/pdf"
    )

    if st.button("🔄 Start Over"):
        st.session_state.clear()
        st.rerun()
