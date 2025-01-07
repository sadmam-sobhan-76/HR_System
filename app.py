from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "HR Recruitment Expert System is live!"

@app.route("/process", methods=["POST"])
def process_csv():
    try:
        # Get form data
        file = request.files["file"]
        required_skills = request.form["required_skills"].split(",")
        min_experience = int(request.form["min_experience"])
        min_cgpa = float(request.form["min_cgpa"])

        # Read the uploaded CSV
        df = pd.read_csv(file)

        # Scoring logic
        SKILL_POINTS = 2
        CGPA_SCORES = {
            "2.00-2.49": 0,
            "2.50-2.99": 5,
            "3.00-3.49": 10,
            "3.50-4.00": 15
        }
        EXPERIENCE_SCORES = {
            0: 9,
            6: 18,
            "more": 27
        }
        MAX_TOTAL_SCORE = 70

        # Process candidates
        def calculate_match(row):
            skills = row["Required Skills"].split(", ")
            skill_match_count = len(set(skills) & set(required_skills))
            skill_score = skill_match_count * SKILL_POINTS

            cgpa_range = row["CGPA"]
            cgpa_score = CGPA_SCORES.get(cgpa_range, 0)

            experience_months = row["Experience(In Months)"]
            if experience_months == 0:
                experience_score = EXPERIENCE_SCORES[0]
            elif experience_months == 6:
                experience_score = EXPERIENCE_SCORES[6]
            else:
                experience_score = EXPERIENCE_SCORES["more"]

            total_score = skill_score + cgpa_score + experience_score
            return min(total_score, MAX_TOTAL_SCORE)

        df["Matching Score"] = df.apply(calculate_match, axis=1)
        sorted_df = df.sort_values(by="Matching Score", ascending=False)

        return jsonify(sorted_df[["Candidate Full Name", "Matching Score"]].to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
