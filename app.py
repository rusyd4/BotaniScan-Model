from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import json

# Load the TorchScript model
model = torch.jit.load('best_model_android.pt')
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to model input size if needed
    transforms.ToTensor(),
])

# Load species mapping from JSON file
with open('species.json', 'r') as f:
    idx_to_class = json.load(f)

app = Flask(__name__)

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    
    # Load and preprocess image
    image = Image.open(io.BytesIO(file.read()))
    image = preprocess(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        
        # Convert output to species name and confidence
        species_id = str(predicted.item())  # Convert to string to match JSON keys
        species_name = idx_to_class.get(species_id, "Unknown Species")
        confidence = torch.softmax(output, dim=1)[0, predicted].item()
        
        return jsonify({'species': species_name, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
