from pydantic import BaseModel, Field
from typing import Annotated, Literal






class UserDetails(BaseModel):

    #------------------------------------------------------------------------------------
    #                                   Creating variables
    #------------------------------------------------------------------------------------

    heart_rate: Annotated[float, Field(..., 
                                       title = "Heart Rate", 
                                       description = "Heart rate (BPM)",
                                       examples=[78.54, 95.34]
    )]
    
    blood_oxygen: Annotated[float, Field(..., 
                                         title = "Blood Oxygen level", 
                                         description = "Blood oxygen level (in percentage)",
                                         examples = [98.45, 96.99]
    )]

    steps: Annotated[int, Field(..., 
                                title = "Total Steps", 
                                description = "Total Steps coverd in a day (units)",
                                examples = [8792, 12873, 2837]
    )]

    sleep_duration: Annotated[float, Field(..., 
                                           title = "Sleep duration", 
                                           description = "Sleep duration (in hrs)",
                                           examples = [4.5, 7.8, 8,0]
    )]
    
    activity_level: Annotated[Literal[0,1,2], Field(..., 
                                                    title = "Activity Level", 
                                                    description = "Activity levels as: 0 for sedentary, 1 for active, and 2 for very active",
                                                    examples = [0, 1, 2],
                                                    min = 0,
                                                    max = 2
    )]





