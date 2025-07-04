# Dependencies
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from scripts.model_scripts import FitnessMeasure, MODEL_NAME, MODEL_VERSION
from schema.main_schema import UserDetails


## setup model
model = FitnessMeasure()



## create an instance of FastAPI class
app = FastAPI()




## creating routes ------------------------------------------------------------------------------------------------------

# general route (/)
@app.get("/")
def general_routes():
    """ this function will display the basic information of the routes """

    routes_basic_info: dict = {
        "message": "welcome in fitness measure model's api",
        "routes": {
            "/predict": "this will used to predict 'Is  person physically fit or not?'",
            "/docs": "at this route you will get documentation for this api"
        }
    }

    return JSONResponse(status_code=200, content=routes_basic_info)








## prediction route
@app.post("/predict")
def predict(details: UserDetails):
    
    try:
        # fetching user details
        heart_rate = details.heart_rate
        blood_oxygen_level = details.blood_oxygen
        steps_counts = details.steps
        sleep_durations = details.sleep_duration
        activity_level = details.activity_level

        prediction = model.predict(
            heart_rate=heart_rate,
            blood_oxygen_level=blood_oxygen_level,
            steps_counts=steps_counts,
            sleep_duration=sleep_durations,
            activity_level=activity_level
        )

        prediction_msg = "Person is physically fit." if prediction else "Person is not physically fit."

    except Exception as e:
        return JSONResponse(status_code = 400, content = {'error occured': e})


    content: dict = {
        'status':'success',
        'model details':{
            'name': MODEL_NAME,
            'version': MODEL_VERSION
        },
        'prediction details':{
            'value': prediction,
            'message': prediction_msg
        },
        'input details': {
            'heart rate in bmp':heart_rate,
            'blood oxygen level':blood_oxygen_level,
            'steps counts':steps_counts,
            'sleep duration in hours':sleep_durations,
            'activity level':activity_level
        }
    }

    return JSONResponse(status_code=200, content=content)



