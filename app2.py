import pandas as pd
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

MODEL_PATH = "models/dt.pkl"  # or your actual path
dt_model = joblib.load(MODEL_PATH)
print("Loaded DT model from", MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Build features DataFrame
        input_df = build_feature_vector(request.form)

        # Get prediction (e.g. Mental_Health_Score)
        prediction = dt_model.predict(input_df)[0]

        print("Input features:\n", input_df)
        print("Model prediction:", prediction)

        return render_template("index.html", submitted=True, prediction=prediction)

    return render_template("index.html", submitted=False, prediction=prediction)


def build_feature_vector(form):
    """
    Build a single-row DataFrame matching your training feature columns.
    Assumes the target was Mental_Health_Score, so it is NOT included.
    """

    # 1. Initialize all features to 0
    features = {
        "Age": 0,
        "Avg_Daily_Usage_Hours": 0.0,
        "Sleep_Hours_Per_Night": 0.0,
        "Conflicts_Over_Social_Media": 0,
        "Gender_Male": 0,
        "Academic_Level_High School": 0,
        "Academic_Level_Undergraduate": 0,
        "Most_Used_Platform_Instagram": 0,
        "Most_Used_Platform_KakaoTalk": 0,
        "Most_Used_Platform_LINE": 0,
        "Most_Used_Platform_LinkedIn": 0,
        "Most_Used_Platform_Snapchat": 0,
        "Most_Used_Platform_TikTok": 0,
        "Most_Used_Platform_Twitter": 0,
        "Most_Used_Platform_VKontakte": 0,
        "Most_Used_Platform_WeChat": 0,
        "Most_Used_Platform_WhatsApp": 0,
        "Most_Used_Platform_YouTube": 0,
        "Affects_Academic_Performance_Yes": 0,
        "Relationship_Status_In Relationship": 0,
        "Relationship_Status_Single": 0,
    }

    # 2. Numeric/base fields
    features["Age"] = form.get("age", type=int)
    features["Avg_Daily_Usage_Hours"] = form.get("avg_daily_usage", type=float)
    features["Sleep_Hours_Per_Night"] = form.get("sleep_hours", type=float)
    features["Conflicts_Over_Social_Media"] = form.get("conflicts_over_social_media", type=int)

    # 3. Gender → Gender_Male (Female is baseline = 0)
    gender = form.get("gender")  # "Male" or "Female"
    if gender == "Male":
        features["Gender_Male"] = 1
    # Female => leave as 0

    # 4. Academic level (Graduate is baseline: both 0)
    academic_level = form.get("academic_level")  # "High School", "Undergrad", "Graduate"
    if academic_level == "High School":
        features["Academic_Level_High School"] = 1
    elif academic_level == "Undergrad" or academic_level == "Undergraduate":
        features["Academic_Level_Undergraduate"] = 1
    # Graduate => both 0

    # 5. Most used platform (others become "baseline" where all dummies are 0)
    platform = form.get("most_used_platform")  # matches select options
    platform_map = {
        "Instagram": "Most_Used_Platform_Instagram",
        "KakaoTalk": "Most_Used_Platform_KakaoTalk",
        "LINE": "Most_Used_Platform_LINE",
        "LinkedIn": "Most_Used_Platform_LinkedIn",
        "Snapchat": "Most_Used_Platform_Snapchat",
        "TikTok": "Most_Used_Platform_TikTok",
        "Twitter": "Most_Used_Platform_Twitter",
        "VKontakte": "Most_Used_Platform_VKontakte",
        "WeChat": "Most_Used_Platform_WeChat",
        "WhatsApp": "Most_Used_Platform_WhatsApp",
        "YouTube": "Most_Used_Platform_YouTube",
    }
    col = platform_map.get(platform)
    if col:
        features[col] = 1
    # If platform is something like Facebook etc., all remain 0 as baseline.

    # 6. Affects academic performance
    affects = form.get("affects_academic_performance")  # "True" or "False"
    if affects == "True":
        features["Affects_Academic_Performance_Yes"] = 1

    # 7. Relationship status (Complicated is baseline)
    relationship = form.get("relationship_status")  # "Single", "In Relationship", "Complicated"
    if relationship == "Single":
        features["Relationship_Status_Single"] = 1
    elif relationship == "In Relationship":
        features["Relationship_Status_In Relationship"] = 1
    # Complicated => both 0

    # 8. Turn into a single-row DataFrame in the correct column order
    column_order = list(features.keys())  # keep explicit order if you want
    df = pd.DataFrame([[features[c] for c in column_order]], columns=column_order)
    return df



if __name__ == "__main__":
    app.run(debug=True)
