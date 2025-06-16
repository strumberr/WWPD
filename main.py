from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from typing import List

app = FastAPI()

# extra comment because i accidentally merged the pull request

class ModelMetadata(BaseModel):
    """Metadata about the ML model."""

    model_name: str
    version: str
    accuracy: float
    tags: List[str]


class PredictionRequest(BaseModel):
    """Input format for making a prediction."""

    features: List[float]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }
    )


class PredictionResponse(BaseModel):
    """Output format for prediction results."""

    predicted_class: str
    confidence: float


@app.get("/")
def read_root() -> dict:
    """
    Root endpoint.

    Returns:
        dict: A welcome message.
    """

    return {"message": "Welcome to the ML Model API"}


@app.get("/model-info", response_model=ModelMetadata)
def get_model_metadata() -> ModelMetadata:
    """
    Returns metadata about the machine learning model.

    Returns:
        ModelMetadata: Model name, version, accuracy, and tags.
    """

    return ModelMetadata(
        model_name="IrisClassifier",
        version="1.0.0",
        accuracy=0.97,
        tags=["iris", "scikit-learn", "classifier"]
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predicts the class of the input features using a dummy model.

    Args:
        request (PredictionRequest): The input features for prediction.

    Returns:
        PredictionResponse: Predicted class and confidence score.
    """


    pred_class = "setosa" if request.features[0] < 5.0 else "versicolor"
    confidence = 0.92

    return PredictionResponse(
        predicted_class=pred_class,
        confidence=confidence
    )
