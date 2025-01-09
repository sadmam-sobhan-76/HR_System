from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://hr-job-portal.netlify.app"]}})

# Define fuzzy variables
skills = ctrl.Antecedent(np.arange(0, 29, 1), 'skills')  # Max 14 skills, 2 points each
cgpa = ctrl.Antecedent(np.arange(5, 21, 1), 'cgpa')       # CGPA scoring: 5, 10, 15, 20
experience = ctrl.Antecedent(np.arange(0, 22, 1), 'experience')  # Experience scoring: 0 or >0
match_score = ctrl.Consequent(np.arange(0, 101, 1), 'match_score')  # Total score out of 100

# Define fuzzy membership functions
skills.automf(3)  # Creates poor, average, good
cgpa.automf(3)    # Creates poor, average, good

# Custom membership for experience
experience['low'] = fuzz.trimf(experience.universe, [0, 0, 1])
experience['high'] = fuzz.trimf(experience.universe, [1, 22, 22])

# Define membership for match score
match_score['poor'] = fuzz.trimf(match_score.universe, [0, 25, 50])
match_score['average'] = fuzz.trimf(match_score.universe, [40, 60, 80])
match_score['excellent'] = fuzz.trimf(match_score.universe, [70, 90, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(skills['good'] & cgpa['good'] & experience['high'], match_score['excellent'])
rule2 = ctrl.Rule(skills['average'] & cgpa['average'] & experience['high'], match_score['average'])
rule3 = ctrl.Rule(skills['poor'] | cgpa['poor'] | experience['low'], match_score['poor'])
rule4 = ctrl.Rule(skills['average'] & cgpa['good'] & experience['high'], match_score['excellent'])

# Create the control system
matching_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
matching_simulation = ctrl.ControlSystemSimulation(matching_ctrl)

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
        required_skills = required_skills.split(",")

        # Read the CSV file
        df = pd.read_csv(file)
        required_columns = ["Candidate Full Name", "Required Skills", "CGPA", "Experience(In Months)"]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns. Required: {required_columns}"}), 400

        # Define scoring function
        def calculate_match(row):
            # Skills score
            candidate_skills = row["Required Skills"].split(", ")
            matched_skills = set(candidate_skills) & set(required_skills)
            skills_score = len(matched_skills) * 2  # 2 points per matched skill

            # CGPA score
            cgpa_value = float(row["CGPA"].split("-")[1])  # Use the upper bound of the CGPA range
            if 2.00 <= cgpa_value <= 2.49:
                cgpa_score = 5
            elif 2.50 <= cgpa_value <= 2.99:
                cgpa_score = 10
            elif 3.00 <= cgpa_value <= 3.49:
                cgpa_score = 15
            elif 3.50 <= cgpa_value <= 4.00:
                cgpa_score = 20

            # Experience score
            experience_value = row["Experience(In Months)"]
            experience_score = 21 if experience_value > 0 else 0  # High contribution for >0 months

            # Input values into fuzzy system
            matching_simulation.input['skills'] = skills_score
            matching_simulation.input['cgpa'] = cgpa_score
            matching_simulation.input['experience'] = experience_score

            # Compute fuzzy match score
            matching_simulation.compute()
            return matching_simulation.output['match_score']

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
