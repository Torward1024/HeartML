from fastapi import FastAPI, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from heart_model import HeartModel
import uvicorn
import argparse
import logging

app = FastAPI()

app.mount("/output", StaticFiles(directory="output"), name='data')

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict")
def perform_prediction(file: UploadFile, request: Request):
    save_pth = "output/" + file.filename
    app_logger.info(f'processing file - {save_pth}')
    with open(save_pth, "wb") as fid:
        fid.write(file.file.read())
    model = HeartModel('model/model_complete.pkl')
    status, result = model.predict()
    return {"status": status}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
