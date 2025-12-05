import pandas as pd
import logging
from flask import Flask, render_template, request, redirect, url_for
from groq import Groq
import markdown as md
import joblib

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

MODEL_PATH = "models/dt.pkl"
dt_model = joblib.load(MODEL_PATH)
print("Loaded DT model from", MODEL_PATH)

MODEL_PATH = "models/rf.pkl"
rf_model = joblib.load(MODEL_PATH)
print("Loaded RF model from", MODEL_PATH)

MODEL_PATH = "models/rfreg.pkl"
rfreg_model = joblib.load(MODEL_PATH)
print("Loaded RFREG model from", MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    # Just render the form
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Build features DataFrame
    input_df = build_feature_vector(request.form)

    # Get prediction
    dt_prediction = dt_model.predict(input_df)[0]
    rf_prediction = rf_model.predict(input_df)[0]
    rfreg_prediction = rfreg_model.predict(input_df)[0]

    print("Input features:\n", input_df)
    print("DT Model prediction:", dt_prediction)
    print("RF Model prediction:", rf_prediction)
    print("RFREG Model prediction:", rfreg_prediction)
    print(f"{dict(request.form)}")

    client = Groq() #Need API Key
    completion = client.chat.completions.create(
        model="groq/compound-mini",
        messages=[
        {
            "role": "system",
            "content": "You are a certified doctor who specially diagnoses levels of social media addiction based on the following parameters:\n\n{'age', 'gender', 'academic_level', 'avg_daily_usage', 'most_used_platform', 'affects_academic_performance', 'sleep_hours', 'relationship_status', 'conflicts_over_social_media'}\n\nYou must always generate a response including your diagnosis of social media addiction between 1 (not addicted) to 10 (incredibly addicted). Follow this diagnosis with a brief reasoning in paragraph format for the diagnosis along with possible remediations in paragraph format."
        },
        {
            "role": "user",
            "content": f"{dict(request.form)}"
        }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )

    llm_message = completion.choices[0].message.content

    llm_html = md.markdown(
        llm_message,
        extensions=["extra", "sane_lists", "tables"]  # tables for that score table
    )

    # Redirect to result page with prediction in query string
    # For a demo this is fine. For sensitive predictions, you’d use session or a DB.
    return redirect(url_for("result", prediction1=dt_prediction, prediction2=rf_prediction, prediction3=rfreg_prediction, message=llm_message))


@app.route("/result", methods=["GET"])
def result():
    prediction1 = request.args.get("prediction1")
    prediction2 = request.args.get("prediction2")
    prediction3 = request.args.get("prediction3")
    message = request.args.get("message")

    if prediction1 is None:
        # If someone hits /result directly, send them back to the form
        return redirect(url_for("index"))
    if prediction2 is None:
        # If someone hits /result directly, send them back to the form
        return redirect(url_for("index"))
    if prediction3 is None:
        # If someone hits /result directly, send them back to the form
        return redirect(url_for("index"))
    if message is None:
        # If someone hits /result directly, send them back to the form
        return redirect(url_for("index"))

    llm_html = md.markdown(
        message,
        extensions=["extra", "sane_lists", "tables"]  # tables for that score table
    )


    return render_template("result.html", prediction1=prediction1, prediction2=prediction2, prediction3=prediction3, message=llm_html)



def build_feature_vector(form):
    # ... your existing build_feature_vector function unchanged ...
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

    features["Age"] = form.get("age", type=int)
    features["Avg_Daily_Usage_Hours"] = form.get("avg_daily_usage", type=float)
    features["Sleep_Hours_Per_Night"] = form.get("sleep_hours", type=float)
    features["Conflicts_Over_Social_Media"] = form.get("conflicts_over_social_media", type=int)

    gender = form.get("gender")
    if gender == "Male":
        features["Gender_Male"] = 1

    academic_level = form.get("academic_level")
    if academic_level == "High School":
        features["Academic_Level_High School"] = 1
    elif academic_level in ("Undergrad", "Undergraduate"):
        features["Academic_Level_Undergraduate"] = 1

    platform = form.get("most_used_platform")
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

    affects = form.get("affects_academic_performance")
    if affects == "True":
        features["Affects_Academic_Performance_Yes"] = 1

    relationship = form.get("relationship_status")
    if relationship == "Single":
        features["Relationship_Status_Single"] = 1
    elif relationship == "In Relationship":
        features["Relationship_Status_In Relationship"] = 1

    column_order = list(features.keys())
    df = pd.DataFrame([[features[c] for c in column_order]], columns=column_order)
    return df


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
