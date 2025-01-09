from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://hr-expert-system.netlify.app"]}})

# Scoring criteria
SKILL_POINTS = 2
CGPA_SCORES = {
    "low": 5,
    "medium": 10,
    "high": 15,
    "very_high": 20
}
EXPERIENCE_SCORES = {
    "none": 7,
    "one_year": 14,
    "more_than_one_year": 21
}

# Define skill list
SKILL_LIST = [
    "Test case creation", "Bug tracking", "Automation testing", "Quality standards compliance",
    "Data analysis", "Attention to detail", "Communication", "Problem-solving",
    "Time management", "Team collaboration", "Power Bi Tools", "Documentation",
    "Jira", "MS Office"
]

@app.route("/")
def home():
    return "HR Recruitment Expert System is live!"

@app.route("/process", methods=["POST"])
def process_csv():
    try:
        # Validate file input
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "File upload is required"}), 400
        file = request.files["file"]

        # Validate form data
        required_skills = request.form.get("required_skills")
        if not required_skills:
            return jsonify({"error": "Required skills input is missing"}), 400
        required_skills = [skill.strip() for skill in required_skills.split(",")]

        # Read the CSV file
        df = pd.read_csv(file)
        required_columns = ["Candidate Full Name", "Required Skills", "CGPA", "Experience(In Months)"]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns. Required: {required_columns}"}), 400

        # Define scoring function
        def calculate_match(row):
            # Skills score
            candidate_skills = [skill.strip() for skill in row["Required Skills"].split(",")]
            matched_skills = set(candidate_skills) & set(required_skills)
            skills_score = len(matched_skills) * SKILL_POINTS

            # CGPA score
            cgpa_value = float(row["CGPA"].split("-")[1])  # Use the upper bound of the CGPA range
            if 2.00 <= cgpa_value <= 2.49:
                cgpa_score = CGPA_SCORES["low"]
            elif 2.50 <= cgpa_value <= 2.99:
                cgpa_score = CGPA_SCORES["medium"]
            elif 3.00 <= cgpa_value <= 3.49:
                cgpa_score = CGPA_SCORES["high"]
            elif 3.50 <= cgpa_value <= 4.00:
                cgpa_score = CGPA_SCORES["very_high"]
            else:
                cgpa_score = 0

            # Experience score
            experience_value = row["Experience(In Months)"]
            if experience_value == 0:
                experience_score = EXPERIENCE_SCORES["none"]
            elif experience_value == 12:
                experience_score = EXPERIENCE_SCORES["one_year"]
            else:
                experience_score = EXPERIENCE_SCORES["more_than_one_year"]

            # Total score
            total_score = skills_score + cgpa_score + experience_score
            return total_score

        # Apply scoring to each candidate
        df["Matching Score"] = df.apply(calculate_match, axis=1)
        sorted_df = df.sort_values(by="Matching Score", ascending=False)

        # Return results
        return jsonify(sorted_df[["Candidate Full Name", "Matching Score"]].to_dict(orient="records"))

    except Exception as e:
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print("Error:", error_message)
        return jsonify(error_message), 500

if __name__ == "__main__":
    app.run(debug=True)
