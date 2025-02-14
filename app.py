import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path on Hugging Face Hub
model_path = "HaryaniAnjali/Llama_3.2_Trained_Emotion"

# Load the tokenizer with error handling
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token if none exists
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Load the model with error handling
try:
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=7, ignore_mismatched_sizes=True, torch_dtype=torch.float16
    ).to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Emotion classification function
def predict_emotion(text):
    if model is None or tokenizer is None:
        return "Model or tokenizer failed to load."

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

    labels = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]
    return labels[predicted_label]

# Gradio UI
ui = gr.Interface(
    fn=predict_emotion,
    inputs="text",
    outputs="text",
    title="Emotion Classifier",
    description="Enter a text and classify its emotion."
)

ui.launch()
