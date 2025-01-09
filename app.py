from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://hr-expert-system.netlify.app"]}})

# Define fuzzy variables
skills = ctrl.Antecedent(np.arange(0, 29, 1), 'skills')  # Max 14 skills, 2 points each
cgpa = ctrl.Antecedent(np.arange(5, 21, 1), 'cgpa')       # CGPA scoring: 5, 10, 15, 20
experience = ctrl.Antecedent(np.arange(7, 22, 1), 'experience')  # Experience scoring: 7, 14, 21
match_score = ctrl.Consequent(np.arange(0, 101, 1), 'match_score')  # Total score out of 100

# Define fuzzy membership functions
skills['low'] = fuzz.trimf(skills.universe, [0, 7, 14])
skills['medium'] = fuzz.trimf(skills.universe, [10, 14, 21])
skills['high'] = fuzz.trimf(skills.universe, [14, 21, 28])

cgpa['low'] = fuzz.trimf(cgpa.universe, [5, 7.5, 10])
cgpa['medium'] = fuzz.trimf(cgpa.universe, [10, 15, 17.5])
cgpa['high'] = fuzz.trimf(cgpa.universe, [15, 20, 20])

experience['low'] = fuzz.trimf(experience.universe, [7, 10.5, 14])
experience['medium'] = fuzz.trimf(experience.universe, [14, 17.5, 19])
experience['high'] = fuzz.trimf(experience.universe, [18, 21, 21])

match_score['poor'] = fuzz.trimf(match_score.universe, [0, 25, 50])
match_score['average'] = fuzz.trimf(match_score.universe, [40, 60, 80])
match_score['excellent'] = fuzz.trimf(match_score.universe, [70, 90, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(skills['high'] & cgpa['high'] & experience['high'], match_score['excellent'])
rule2 = ctrl.Rule(skills['medium'] & cgpa['medium'] & experience['medium'], match_score['average'])
rule3 = ctrl.Rule(skills['low'] | cgpa['low'] | experience['low'], match_score['poor'])
rule4 = ctrl.Rule(skills['medium'] & cgpa['high'] & experience['medium'], match_score['excellent'])

matching_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
matching_simulation = ctrl.ControlSystemSimulation(matching_ctrl)

@app.route("/")
def home():
    return "HR Recruitment Expert System is live!"

@app.route("/process", methods=["POST"])
def process_csv():
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "File upload is required"}), 400
        file = request.files["file"]

        required_skills = request.form.get("required_skills")
        if not required_skills:
            return jsonify({"error": "Required skills input is missing"}), 400
        required_skills = required_skills.split(",")

        df = pd.read_csv(file)
        required_columns = [
            "Candidate Full Name", "Required Skills", "CGPA", "Experience(In Months)"
        ]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns. Required: {required_columns}"}), 400

        skill_points = {skill.strip(): 2 for skill in [
            "Test case creation", "Bug tracking", "Automation testing", "Quality standards compliance",
            "Data analysis", "Attention to detail", "Communication", "Problem-solving", 
            "Time management", "Team collaboration", "Power Bi Tools", "Documentation", 
            "Jira", "MS Office"
        ]}

        def calculate_match(row):
            try:
                candidate_skills = row["Required Skills"].split(", ")
                matched_skills = set(candidate_skills) & set(skill_points.keys())
                skills_score = len(matched_skills) * 2

                cgpa_value = float(row["CGPA"].split("-")[1])
                if 2.00 <= cgpa_value <= 2.49:
                    cgpa_score = 5
                elif 2.50 <= cgpa_value <= 2.99:
                    cgpa_score = 10
                elif 3.00 <= cgpa_value <= 3.49:
                    cgpa_score = 15
                elif 3.50 <= cgpa_value <= 4.00:
                    cgpa_score = 20

                experience_value = row["Experience(In Months)"]
                if experience_value == 0:
                    experience_score = 7
                elif experience_value == 12:
                    experience_score = 14
                else:
                    experience_score = 21

                matching_simulation.input['skills'] = skills_score
                matching_simulation.input['cgpa'] = cgpa_score
                matching_simulation.input['experience'] = experience_score
                matching_simulation.compute()
                return matching_simulation.output['match_score']

            except Exception as e:
                print(f"Error in calculate_match: {str(e)}")
                raise

        df["Matching Score"] = df.apply(calculate_match, axis=1)
        sorted_df = df.sort_values(by="Matching Score", ascending=False)

        return jsonify(sorted_df[["Candidate Full Name", "Matching Score"]].to_dict(orient="records"))

    except Exception as e:
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        return jsonify(error_message), 500

if __name__ == "__main__":
    app.run(debug=True)
