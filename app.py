import gradio as gr
from transformers import pipeline

# Load your model from Hugging Face
model_name = "HaryaniAnjali/Llama_3.2_Trained_Emotion"
classifier = pipeline("Emotion-Classification", model=model_name)

def classify_text(text):
    return classifier(text)

# Create a Gradio interface
interface = gr.Interface(fn=classify_text, inputs="text", outputs="json")
interface.launch()
