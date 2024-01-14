from enum import Enum
from fastapi import APIRouter, FastAPI, Query, HTTPException, Path, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from io import StringIO, BytesIO
from pydantic import BaseModel, HttpUrl, EmailStr, validator, ValidationError as PydanticValidationError
from typing import List, Optional

from starlette.responses import JSONResponse

from .utils import BackgroundTaskManager, EmailSender
from template_project.predictor_config import predictor_config
from template_project import predictor
import base64
import os
import PIL

import time


IMG_DIR = os.path.join(os.path.abspath('./'), 'template_project/data/saved_images')
EMAIL_RESPONSE = "Email with the prediction will be sent to {email}"
print(predictor_config.predictor_model_config)

predictor_model = getattr(predictor, predictor_config.predictor_name)(predictor_config.predictor_model_config)
email_sender = EmailSender(sender_email=os.environ['SENDER_EMAIL'],
                           sender_password=os.environ['SENDER_PASSW'])


router = APIRouter()
task_manager = BackgroundTaskManager()


class Prediction(BaseModel):
    name: str
    prediction: int

class Input(BaseModel):
    task_id: str
    data: dict | None = None
    email: EmailStr | None = None


def prepare_input(input: Input) -> dict:
    data = input.data
    return data


async def process_input_file(content: bytes) -> list:
    file_content = content.decode('utf-8')
    file_like_object = StringIO(file_content)
    prompt_list = file_like_object.readlines()
    data = dict(prompt=prompt_list)
    return data


def save_and_return_path(prediction: PIL.Image.Image,
                         image_name: str):
    img_path = os.path.join(IMG_DIR, f"{image_name}.png")
    prediction.save(img_path)
    return img_path


def prepare_single_image_for_response(img_path: str,
                                      img_name: str) -> dict:
    with open(img_path, "rb") as img_file:
        encoded_data = base64.b64encode(img_file.read()).decode("utf-8")
    response_dict = {"filename": f"{img_name}.png", "data": encoded_data}
    return response_dict


def prepare_multiple_images_for_response(img_path_list: list,
                                         img_name_list: list) -> dict:
    response_list = list()
    for img_path, img_name in zip(img_path_list, img_name_list):
        response_dict = prepare_single_image_for_response(img_path=img_path,
                                                          img_name=img_name)
        response_list.append(response_dict)
    return response_list


def make_prediction(input: Input) -> list:
    prepared_input = prepare_input(input)
    prediction = predictor_model.predict(prepared_input)
    return prediction


def make_prediction_background(input: Input):
    task_id = input.task_id
    task_manager.add_task(task_id)
    prediction = make_prediction(input=input)
    img_name_list = [f'{task_id}_{i}' for i in range(len(prediction))]
    img_path_list = [save_and_return_path(prediction=img_pred, image_name=img_name)
                     for img_pred, img_name in zip(prediction, img_name_list)]
    email_sender.send_mult_imgs_via_email(img_path_list=img_path_list,
                                          recipient_email=input.email)
    task_manager.mark_task_completed(task_id)


def run_prediction_process(input: Input) -> JSONResponse:
    task_id = input.task_id
    prediction = make_prediction(input)
    img_name_list = [f'{task_id}_{i}' for i in range(len(prediction))]
    img_path_list = [save_and_return_path(prediction=img_pred, image_name=img_name)
                     for img_pred, img_name in zip(prediction, img_name_list)]
    response = prepare_multiple_images_for_response(img_path_list=img_path_list,
                                                    img_name_list=img_name_list)
    return JSONResponse(content=response)


async def run_prediction_process_background(input: Input, background_tasks: BackgroundTasks) -> Prediction | dict:
    background_tasks.add_task(make_prediction_background, input=input)
    response_message = EMAIL_RESPONSE.format(email=input.email)
    return {"message": response_message}


@router.post("/predict")
async def predict(input: Input, background_tasks: BackgroundTasks) -> JSONResponse:
    try:
        if input.email is not None:
            response = await run_prediction_process_background(input=input,
                                                               background_tasks=background_tasks)
            return response
        response = run_prediction_process(input=input)
        return response
    except Exception as e:
        return {"error": str(e)}


@router.post("/predict_file")
async def predict_file(background_tasks: BackgroundTasks,
                       file: UploadFile = File(...),
                       task_id: str = Form(...),
                       email: str | None = Form(None)) -> JSONResponse:
    try:
        content = await file.read()
        data = await process_input_file(content)
        input = Input(task_id=task_id,
                      email=email,
                      data=data)
        if email is not None:
            response = await run_prediction_process_background(input=input,
                                                               background_tasks=background_tasks)
            return response
        response = run_prediction_process(input=input)
        return response
    except Exception as e:
        return {"error": str(e)}


@router.get("/task-status/{task_id}")
# async def get_task_status(task_id: str = Depends(task_manager.get_task_status)):
def get_task_status(task_id: str):
    return {"task_id": task_id, "status": task_manager.get_task_status(task_id)}
