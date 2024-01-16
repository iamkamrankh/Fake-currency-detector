
# To run file, type "uvicorn main:app --reload" in terminal


from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load your model
model = load_model("model.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file as PIL Image
    image = Image.open(io.BytesIO(await file.read()))
    
    # Resize and preprocess the image
    image = image.resize((50, 50))
    image = np.array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(image)
    if prediction[0]<0.1:
        return {"prediction": "real"}
    else:
        return {"prediction": "fake"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import io

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# # Load your model
# model = load_model("saved_model.h5")

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     # Read image file as PIL Image
#     image = Image.open(io.BytesIO(await file.read()))
    
#     # Resize and preprocess the image
#     image = image.resize((224, 224))
#     image = np.array(image)
#     image = np.expand_dims(image, axis=0) / 255.0

#     # Make a prediction
#     prediction = model.predict(image)
    
#     # Extract the predicted class probabilities
#     class_probabilities = prediction[0]

#     # Define a threshold for prediction
#     threshold = 0.1

#     # Check if any class probability is below the threshold
#     if any(prob < threshold for prob in class_probabilities):
#         return {"prediction": "real"}
#     else:
#         return {"prediction": "fake"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import io

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# # Load your model
# model = load_model("model.tflite")

# # Define class labels
# class_labels = [
#     '10_front', '10_back',
#     '20_front', '20_back',
#     '50_front', '50_back',
#     '100_front', '100_back',
#     '500_front', '500_back',
#     '1000_front', '1000_back',
#     '5000_front', '5000_back','others'
# ]
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     # Read image file as PIL Image
#     image = Image.open(io.BytesIO(await file.read()))
    
#     # Resize and preprocess the image
#     image = image.resize((224, 224))
#     image = np.array(image)
#     image = np.expand_dims(image, axis=0) / 255.0

#     # Make a prediction
#     prediction = model.predict(image)

#     if len(prediction) == 0:
#         return {"prediction": "unable to predict"}

#     # Map predicted probabilities to class labels
#     predicted_classes = [class_labels[i] for i in np.argmax(prediction, axis=1)]

#     if len(predicted_classes) == 0:
#         return {"prediction": "class label not found"}

#     # Calculate total amount
#     currency_values = {
#         '10_front': 10, '10_back': 10,
#         '20_front': 20, '20_back': 20,
#         '50_front': 50, '50_back': 50,
#         '100_front': 100, '100_back': 100,
#         '500_front': 500, '500_back': 500,
#         '1000_front': 1000, '1000_back': 1000,
#         '5000_front': 5000, '5000_back': 5000,
#         'others':0
#     }

#     total_amount = sum(currency_values[prediction] for prediction in predicted_classes)

#     # Return predicted classes and total amount
#     return {"prediction": total_amount, "total_amount": total_amount}


# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# # Load your TensorFlow Lite model
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Define class labels
# class_labels = [
#     '10_front', '10_back',
#     '20_front', '20_back',
#     '50_front', '50_back',
#     '100_front', '100_back',
#     '500_front', '500_back',
#     '1000_front', '1000_back',
#     '5000_front', '5000_back', 'others'
# ]

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     # Read image file as PIL Image
#     image = Image.open(io.BytesIO(await file.read()))
    
#     # Resize and preprocess the image
#     image = image.resize((180, 180))  # Resize to match the model's input size
#     image = np.array(image, dtype=np.float32)  # Convert to float32
#     image = np.expand_dims(image, axis=0) / 255.0

#     # Run inference with TensorFlow Lite model
#     interpreter.set_tensor(input_details[0]['index'], image)
#     interpreter.invoke()
#     prediction = interpreter.get_tensor(output_details[0]['index'])

#     # Map predicted probabilities to class labels and calculate total amount
#     predicted_classes = [class_labels[i] for i in np.argmax(prediction, axis=1)]
#     currency_values = {
#         '10_front': 10, '10_back': 10,
#         '20_front': 20, '20_back': 20,
#         '50_front': 50, '50_back': 50,
#         '100_front': 100, '100_back': 100,
#         '500_front': 500, '500_back': 500,
#         '1000_front': 1000, '1000_back': 1000,
#         '5000_front': 5000, '5000_back': 5000,
#         'others': 0
#     }

#     total_amount = sum(currency_values[class_label] for class_label in predicted_classes)

#     # Return predicted classes and total amount
#     return {"prediction": total_amount, "total_amount": total_amount}
