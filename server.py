from fastapi import FastAPI, File, UploadFile
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from fastapi.middleware.cors import CORSMiddleware
from keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

import json
app = FastAPI()
vocab_size = 3000
embedding_dim = 32
max_length = 60
truncation_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
sentence = ["My credit card no is 124345346", "game of thrones season finale showing this sunday night"]
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def predict(image: Image.Image):
    x = np.asarray(image.resize((150, 150)))[..., :3]
    x = np.expand_dims(x, axis=0)
    image = np.vstack([x])  
    model = keras.models.load_model('model/image_model.h5')
    classes = model.predict(image, batch_size=1)
    return str(classes[0])

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.get("/textclass")
async def text_class(needy: str):
    l = []
    l.append(needy)
    model = keras.models.load_model('model/text_model.h5')
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    sequences = tokenizer.texts_to_sequences(l)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)
    
    predictions = model.predict(padded)
    ret = ""
    f = True
    for i in range(len(predictions)):
        if predictions[i][0]>0.5:
            ret += ("Sensitive - "+ str(predictions[i][0]) + "\n")
            f = True
        else:
            ret += ("Non-Sensitive - " + str(predictions[i][0]) + "\n")
            f= False
    return {ret, f}

@app.post("/imageclass")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    sens = predict(image)

    return {sens}