import joblib  # ← Replace pickle with joblib
import pandas
from typing import Literal

# Model paths
DECISION_TREE_MODEL_5003 = './model/model_dtree_5003.joblib'  # ← Updated extension
DECISION_TREE_MODEL_5003_SCALER = './model/scaler_model_dtree_5003.joblib'  # ← Updated extension
MODEL_NAME = 'Fitness Measure'
MODEL_VERSION = '1.0'

def load_model(scaler_path: str = None):
    # Load model using joblib
    model = joblib.load(DECISION_TREE_MODEL_5003)

    scaler = None
    if scaler_path:
        scaler = joblib.load(scaler_path)

    return model, scaler

class FitnessMeasure:

    def __init__(self):
        self.model, self.scaler = load_model(scaler_path=DECISION_TREE_MODEL_5003_SCALER)

    def predict(
            self, 
            heart_rate: float,
            blood_oxygen_level: float,
            steps_counts: int,
            sleep_duration: float,
            activity_level: Literal[0, 1, 2]
        ) -> int:

        # Build input data
        data: dict = {
            'Heart Rate (BPM)': heart_rate,
            'Blood Oxygen Level (%)': blood_oxygen_level,
            'Step Count': steps_counts,
            'Sleep Duration (hours)': sleep_duration,
            'Activity Level': activity_level
        }

        # Convert to DataFrame
        df = pandas.DataFrame(data=data, index=[0])

        # You can scale here if needed: df = self.scaler.transform(df)

        prediction = self.model.predict(df)[0]
        return int(prediction)

if __name__ == "__main__":
    model = FitnessMeasure()
    prediction = model.predict(
        heart_rate=72,
        blood_oxygen_level=98.0,
        steps_counts=8000,
        sleep_duration=7.5,
        activity_level=1
    )
    print(prediction)
