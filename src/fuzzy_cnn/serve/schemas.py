from pydantic import BaseModel

class Prediction(BaseModel):
    label: str
    confidence: float
    
    @classmethod
    def from_domain(cls, domain_prediction: dict) -> "Prediction":
        return cls(
            label=str(domain_prediction["label"]),
            confidence=float(domain_prediction["prob"]),
        )

class InferenceResult(BaseModel):
    predictions: list[Prediction]

    @classmethod
    def from_domain(cls, domain_predictions: list[dict]) -> "InferenceResult":
        return cls(
            predictions=[
                Prediction.from_domain(p)
                for p in (domain_predictions or [])
            ]
        )
