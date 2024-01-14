from pydantic import BaseModel


class PredictorConfig(BaseModel):
    predictor_name: str
    predictor_model_config: str


model_config = '{"model_id": "CompVis/stable-diffusion-v1-4", "num_inference_steps": 50}'

predictor_config = PredictorConfig(predictor_name='DiffusionPredictor', predictor_model_config=model_config)
