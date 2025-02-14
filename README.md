---
title: Emotion Classification
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
short_description: Emotion Classification
---

Emotion Classifier

1. Overview

This project classifies emotions in textual data using advanced Large Language Models (LLMs). Three models have been fine-tuned on a structured emotion dataset to detect seven primary emotions: anger, disgust, fear, guilt, joy, sadness, and shame. The models used are:

* GPT-4

* Llama-3.2

* T5 Base

The models are deployed on Hugging Face Spaces using Gradio, offering a real-time, interactive web-based interface for emotion classification.

2. Live Demo

Interact with the emotion classifier models in real-time by entering a text sample and selecting a model for classification.

Link: https://huggingface.co/spaces/HaryaniAnjali/Emotion_Classification 

3. Models Used

     * GPT-4: A state-of-the-art language model optimized for text understanding.
     * Llama-3.2: A fine-tuned 7B parameter model for emotion classification.
     * T5 Base: A smaller yet efficient model designed for prompt-based emotion detection.


4. Dataset

The dataset used for training consists of text samples labeled with seven emotions. 

5. Implementation Details

   * Data Preprocessing: Text cleaning (alphnumeric code, punctuation removal).


   * Model Fine-Tuning: Fine-tuning was performed using prompt-based training strategies for all models (GPT-4, Llama-3.2, and T5 Base).
GPT-4, Llama-3.2, and T5 Base models were trained with specialized prompts to improve emotion classification.
The training approach included using structured prompts to guide the models in recognizing and classifying emotions in text.
This method helped the models focus on the task of emotion recognition from natural language inputs, improving their ability to understand context and nuances in emotions.

   * Model Deployment: Fine-tuned models were uploaded to Hugging Face Model Hub.

   * Performance Analysis The models were evaluated using accuracy, precision, recall, and F1-score. Here are the results:
  
     

![image/png](https://cdn-uploads.huggingface.co/production/uploads/67aa7cb6c6abc0f7e891d922/F15Sp2zmgNfsxkAczMmQb.png)

   * Key Findings: GPT-4 demonstrated the best performance with the highest precision (0.8006) and accuracy (0.7399). Llama-3.2 showed good performance, but with slightly lower accuracy compared to GPT-4. T5 Base was effective but had slightly lower scores across the board.

6. How to Use

* Enter a text input in the provided text box.
* Click the "Submit" button to classify the emotion.
* The predicted emotion will be displayed on the interface.

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).