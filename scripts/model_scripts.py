
import pickle
import pandas 

from typing import Literal




## declaring models path
DECISION_TREE_MODEL_5003 = './model/model_dtree_5003.pkl'
DECISION_TREE_MODEL_5003_SCALER = './model/scaler_model_dtree_5003.pkl'
MODEL_NAME = 'Fitness Measure'
MODEL_VERSION = '1.0'




def load_model(scaler_path: str = None):
    ## import model
    model = pickle.load(open(DECISION_TREE_MODEL_5003, 'rb'))

    if scaler_path:
        scaler = pickle.load(open(scaler_path, 'rb'))

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
            activity_level: Literal[0,1,2]
        ) -> int:

        ## collecting user data
        data: dict = {
            'Heart Rate (BPM)': heart_rate,
            'Blood Oxygen Level (%)': blood_oxygen_level,	
            'Step Count': steps_counts,	
            'Sleep Duration (hours)': sleep_duration,	
            'Activity Level': activity_level
        }

        ## converting as pandas dataframe
        df = pandas.DataFrame(data=data, index=[0])

        prediction = self.model.predict(df)[0]

        return int(prediction)



if __name__ == "__main__":

    model = FitnessMeasure()
    prediction = model.predict(total_steps=8000, very_active_minutes=23, calories=1200)
    print(prediction)



