from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
#from heart_model import HeartModel

import pandas as pd
import uvicorn
import argparse
import logging
import io

app = FastAPI()

app.mount("/output", StaticFiles(directory="output"), name='data')

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

app_logger.info(f'Initializing model...')
#model = HeartModel('model/model_complete.pkl')
#model.load_model()
app_logger.info("Model loaded successfully")
app_logger.info(f"Using treshold={model.get_treshold()}")

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict")
async def perform_prediction(file: UploadFile):
    app_logger.info(f'Processing file - {file.filename}')

    contents = await file.read()
    csv_data = io.StringIO(contents.decode('utf-8'))
    data = pd.read_csv(csv_data)

    app_logger.info(f"Loaded DataFrame with shape: {data.shape}")
<<<<<<< HEAD
    app_logger.info(f"DataFrame dtypes: {data.dtypes}")
    app_logger.info(f"DataFrame sample: {data.head().to_dict()}")
#   status, result = model.predict(data)
=======
    status, result = model.predict(data)
>>>>>>> e580760302233cba7e1d99602582d634b479b73d

#   if status == 'success':
#          # save output result
#          output_path = "output/result.csv"
#          result.to_csv(output_path, index=False)
#          app_logger.info(f"Predictions saved to {output_path}")
#         
#          return {
#              "status": "success", 
#                "message": f"Predictions saved to {output_path}",
#                "predictions_count": len(result)
#          }
#  else:
#          return {"status": "error", "message": result}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="localhost", type=str, dest="host")
    args = vars(parser.parse_args())
    uvicorn.run(app, **args)
