import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths on Hugging Face Hub
model_paths = {
    "LLaMA-3.2": "HaryaniAnjali/Llama_3.2_Trained_Emotion"
}

# Load tokenizers first with error handling
tokenizers = {}
for name, path in model_paths.items():
    try:
        print(f"🔄 Loading tokenizer for {name}...")
        tokenizer = AutoTokenizer.from_pretrained(path)

        # Ensure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token if none exists
        
        tokenizers[name] = tokenizer
        print(f"Tokenizer loaded for {name}")
    except Exception as e:
        print(f"Error loading tokenizer for {name}: {e}")

# Lazy loading of models to save memory
models = {}

def get_model(model_name):
    if model_name not in models:
        try:
            print(f"Loading model: {model_name}...")
            models[model_name] = AutoModelForSequenceClassification.from_pretrained(
                model_paths[model_name], num_labels=7, ignore_mismatched_sizes=True, torch_dtype=torch.float16
            ).to(device)
            print(f"Model {model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None
    return models[model_name]

# Emotion classification function
def predict_emotion(text, model_name):
    model = get_model(model_name)
    if model is None:
        return f"Model {model_name} failed to load. Check logs."

    tokenizer = tokenizers.get(model_name)
    if tokenizer is None:
        return f"Tokenizer for {model_name} not available."

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

    labels = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]
    return labels[predicted_label]

# Gradio UI
ui = gr.Interface(
    fn=predict_emotion,
    inputs=["text", gr.Radio(list(model_paths.keys()), label="Select Model")],
    outputs="text",
    title="Emotion Classifier",
    description="Enter a text, select a model, and classify its emotion."
)

ui.queue().launch()
