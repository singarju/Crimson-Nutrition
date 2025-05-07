# app.py
import os
from flask import Flask, render_template, request, jsonify
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from groq import Groq
import os
from dotenv import load_dotenv   # add this
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for model and class names
model = None
idx_to_class = None

def load_model():
    global model, idx_to_class
    
    # Initialize the model architecture
    model = torchvision.models.resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 101)
    )
    
    # Load the saved state
    checkpoint = torch.load('./food101_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    idx_to_class = checkpoint['idx_to_class']  # Dict: class index -> class name
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
def get_calorie_estimate(food_name: str) -> tuple[int,int]:
    """
    Ask the Groq LLM to estimate calories for a single serving of `food_name`.
    Returns (lower, upper).
    """
    # make sure your key is loaded
    assert "GROQ_API_KEY" in os.environ, "GROQ_API_KEY not found in env"
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    
    system_prompt = (
        "You are a nutrition expert. For the food item you are given, "
        "estimate a realistic calorie range (kcal) for a standard serving."
        "Reply only with two integers: lower and upper."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": food_name}
    ]

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=50,
        top_p=1.0
    )

    # dump the raw response so you can inspect any errors or empty content
    print("LLM raw:", completion)

    text = completion.choices[0].message.content or ""
    text = text.strip()
    print("LLM replied:", repr(text))

    # split and parse
    low, high = map(int, text.split())
    return low, high

def predict_image(image):
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        results = []
        for i in range(3):
            idx = top_indices[i].item()
            results.append({
                'class': idx_to_class[idx],
                'probability': f"{top_probs[i].item()*100:.2f}%"
            })
    print(idx_to_class)
    return results

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Save the uploaded image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)
        
        # Open and process the image
        img = Image.open(img_path).convert('RGB')
        
        # Make prediction (top‐3 classes & probs)
        predictions = predict_image(img)
        
        # Calorie estimate for the top‐1 class
        top_food = predictions[0]['class']
        low, high = get_calorie_estimate(top_food)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'calories': {
                'food': top_food,
                'lower': low,
                'upper': high
            },
            'image_path': img_path
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Load model when starting the app
    load_model()
    app.run(debug=True)