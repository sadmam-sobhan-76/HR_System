from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend integration

# Define fuzzy variables
cgpa = ctrl.Antecedent(np.arange(2.0, 4.1, 0.1), 'cgpa')
skills = ctrl.Antecedent(np.arange(0, 101, 1), 'skills')
experience = ctrl.Antecedent(np.arange(0, 61, 1), 'experience')  # Months of experience
match_score = ctrl.Consequent(np.arange(0, 101, 1), 'match_score')

# Define fuzzy membership functions
cgpa['low'] = fuzz.trimf(cgpa.universe, [2.0, 2.5, 3.0])
cgpa['medium'] = fuzz.trimf(cgpa.universe, [2.5, 3.0, 3.5])
cgpa['high'] = fuzz.trimf(cgpa.universe, [3.0, 4.0, 4.0])

skills['low'] = fuzz.trimf(skills.universe, [0, 30, 60])
skills['medium'] = fuzz.trimf(skills.universe, [40, 60, 80])
skills['high'] = fuzz.trimf(skills.universe, [70, 100, 100])

experience['low'] = fuzz.trimf(experience.universe, [0, 12, 24])
experience['medium'] = fuzz.trimf(experience.universe, [18, 36, 48])
experience['high'] = fuzz.trimf(experience.universe, [36, 60, 60])

match_score['poor'] = fuzz.trimf(match_score.universe, [0, 25, 50])
match_score['average'] = fuzz.trimf(match_score.universe, [40, 60, 80])
match_score['excellent'] = fuzz.trimf(match_score.universe, [70, 90, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(cgpa['high'] & skills['high'] & experience['high'], match_score['excellent'])
rule2 = ctrl.Rule(cgpa['medium'] & skills['medium'] & experience['medium'], match_score['average'])
rule3 = ctrl.Rule(cgpa['low'] | skills['low'] | experience['low'], match_score['poor'])
rule4 = ctrl.Rule(cgpa['high'] & skills['medium'] & experience['medium'], match_score['excellent'])

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
        min_experience = request.form.get("min_experience")
        min_cgpa = request.form.get("min_cgpa")

        if not required_skills or not min_experience or not min_cgpa:
            return jsonify({"error": "Missing required fields"}), 400

        required_skills = required_skills.split(",")
        min_experience = int(min_experience)
        min_cgpa = float(min_cgpa)

        # Read the CSV file
        df = pd.read_csv(file)

        # Calculate fuzzy matching scores
        def calculate_match(row):
            # Skills match percentage
            skills_list = row["Required Skills"].split(", ")
            skill_match_count = len(set(skills_list) & set(required_skills))
            skills_score = (skill_match_count / len(required_skills)) * 100

            # Input values into fuzzy system
            matching_simulation.input['cgpa'] = float(row["CGPA"].split("-")[0])  # Lower bound of CGPA range
            matching_simulation.input['skills'] = skills_score
            matching_simulation.input['experience'] = row["Experience(In Months)"]

            # Compute fuzzy score
            matching_simulation.compute()
            return matching_simulation.output['match_score']

        # Apply fuzzy matching to all candidates
        df["Matching Score"] = df.apply(calculate_match, axis=1)

        # Sort candidates by matching score
        sorted_df = df.sort_values(by="Matching Score", ascending=False)

        # Return results as JSON
        return jsonify(sorted_df[["Candidate Full Name", "Matching Score"]].to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
