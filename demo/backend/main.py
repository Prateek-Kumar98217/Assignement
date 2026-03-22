import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator
    
#only three of the features are allowed to be none, after data analysis of training set and testing set
class PedictionRequest(BaseModel):
    journal_text: str
    ambience_type: str
    time_of_day: str
    duration_min: int
    sleep_hours: float | None
    energy_level: int
    stress_level: int
    previous_day_mood: str | None
    face_emotion_hint: str | None
    reflection_quality: str

    @field_validator("ambience_type", mode="after")
    @classmethod
    def validate_ambience(cls, value: str):
        if value.lower() not in ["cafe", "forest", "mountain", "ocean", "rain"]:
            raise ValueError("Ambience must be one of the following: cafe, forest, mountain, ocean, rain")
        return value

    @field_validator("time_of_day", mode="after")
    @classmethod
    def validate_time(cls, value: str):
        if value.lower() not in ["morning", "afternoon", "evening", "night", "early_morning"]:
            raise ValueError("Time must be one of the following: morning, afternoon, evening, night")
        return value

    @field_validator("reflection_quality", mode="after")
    @classmethod
    def validate_reflection_quality(cls, value: str):
        if value.lower() not in ["clear", "vague", "conflicted"]:
            raise ValueError("Reflection quality must be one of the following: clear, vague, conflicted")
        return value  

    @field_validator("energy_level", "stress_level", mode="after")
    @classmethod
    def validate_energy_level(cls, value: int):
        if value not in range(1, 6):
            raise ValueError("Energy level must be between 1 and 5")
        return value

    @field_validator("sleep_hours", mode="after")
    @classmethod
    def validate_sleep_hours(cls, value: float):
        if value is not None and not (0 <= value <= 24):
            raise ValueError("Sleep hours must be between 0 and 24")
        return value

    @field_validator("previous_day_mood", mode="after")
    @classmethod
    def validate_previous_day_mood(cls, value: str):
        if value and value.lower() not in ["calm", "focused", "mixed", "neutral", "overwhelmed", "restless"]:
            raise ValueError("Previous day mood must be one of the following: calm, focused, mixed, neutral, overwhelmed, restless")
        return value

    @field_validator("face_emotion_hint", mode="after")
    @classmethod
    def validate_face_emotion(cls, value: str):
        if value and value.lower() not in ['calm_face', 'tired_face', 'happy_face', 'tense_face', 'neutral_face', 'none']:
            raise ValueError("Face emotion must be one of the following: calm_face, tired_face, happy_face, tense_face, neutral_face, none")
        return value

# Decision engine: Reasons for such implementation discussed in README.md
def apply_decision_engine(state, intensity, is_uncertain):
    if is_uncertain:
        return pd.Series(["prompt_manual_reflection", "end_of_day"])
    
    if state in ["calm", "focused", "neutral"]:
        if intensity <= 3:
            return pd.Series(["no_action_needed", "none"])
        else:
            return pd.Series(["maintain_momentum", "none"])
            
    if state in ["overwhelmed", "restless"]:
        if intensity >= 4:
            return pd.Series(["suggest_short_break", "immediate"])
        else:
            return pd.Series(["review_daily_schedule", "next_transition"])
            
    if state == "mixed":
        return pd.Series(["log_gratitude_journal", "evening"])
        
    return pd.Series(["no_action_needed", "none"])   

model = {}
MODAL_NAME = "xgboost_clf"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and unload it on shutdown."""
    global model
    model["state"] = joblib.load(f"saved-models/{MODAL_NAME}/state.joblib")
    model["intensity"] = joblib.load(f"saved-models/{MODAL_NAME}/intensity.joblib")
    model["state_encoder"] = joblib.load(f"saved-models/{MODAL_NAME}/state_encoder.joblib")
    model["intensity_encoder"] = joblib.load(f"saved-models/{MODAL_NAME}/intensity_encoder.joblib")
    yield
    model.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status" : "working"}

@app.post("/predict")
async def predict(data: PedictionRequest):
    temp_df = pd.DataFrame([data.dict()])
    state_prediction = model["state_encoder"].inverse_transform(model["state"].predict(temp_df))[0]
    state_prediction_confidence = float(np.max(model["state"].predict_proba(temp_df)[0]))
    intensity_prediction = model["intensity_encoder"].inverse_transform(model["intensity"].predict(temp_df))[0]
    intensity_prediction_confidence = float(np.max(model["intensity"].predict_proba(temp_df)[0]))
    confidence = (state_prediction_confidence + intensity_prediction_confidence) / 2
    is_uncertain = bool(confidence < 0.5)
    decision = apply_decision_engine(
        state_prediction,
        intensity_prediction,
        is_uncertain
    )
    print("recieved: ", data)
    print("sending", {
        "predicted_state": state_prediction,
        "predicted_intensity": int(intensity_prediction),
        "confidence": confidence,
        "uncertain_flag": is_uncertain,
        "what_to_do": decision.iloc[0],
        "when_to_do": decision.iloc[1]
    })
    return {
        "predicted_state": state_prediction,
        "predicted_intensity": int(intensity_prediction),
        "confidence": confidence,
        "uncertain_flag": is_uncertain,
        "what_to_do": decision.iloc[0],
        "when_to_do": decision.iloc[1]
    }