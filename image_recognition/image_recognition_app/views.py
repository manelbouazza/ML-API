
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import joblib
import numpy as np
import base64

class_names = ['Plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def predict_form(request):
    return render(request, 'image_recognition/predict_form.html')
@csrf_exempt
def predict_image(request):
    mdl = joblib.load(r"C:\Users\Mon-PC\Documents\soap_projects\image_recognition\image_recognition_app\models\image_recognition.pkl")

    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.FILES.get('image')

        if image_file:
            # Read the image from the file
            img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.uint8)

            # Assuming mdl is a model with a predict method
            prediction = mdl.predict(np.array([img]) / 255.0)  # Normalize to [0, 1]
            
            index = np.argmax(prediction)
            result = {'prediction': class_names[index]}
            return JsonResponse(result)
        else:
            return JsonResponse({'error': 'No image provided'})
    else:
        return JsonResponse({'error': 'Invalid request method'})
