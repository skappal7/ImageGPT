import base64
import os
import uuid
import requests
import numpy as np
from PIL import Image
import streamlit as st

MARKDOWN = """
# WebcamGPT 💬 + 📸

webcamGPT is a tool that allows you to chat with video using OpenAI Vision API.

Visit [awesome-openai-vision-api-experiments](https://github.com/roboflow/awesome-openai-vision-api-experiments) 
repository to find more OpenAI Vision API experiments or contribute your own.
"""
AVATARS = (
    "https://media.roboflow.com/spaces/roboflow_raccoon_full.png",
    "https://media.roboflow.com/spaces/openai-white-logomark.png"
)
IMAGE_CACHE_DIRECTORY = "data"
API_URL = "https://api.openai.com/v1/chat/completions"


def preprocess_image(image: Image.Image) -> Image.Image:
    # Flip left-right
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped_image


def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_image


def compose_payload(image: Image.Image, prompt: str) -> dict:
    base64_image = encode_image_to_base64(image)
    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }


def compose_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


def prompt_image(api_key: str, image: Image.Image, prompt: str) -> str:
    headers = compose_headers(api_key=api_key)
    payload = compose_payload(image=image, prompt=prompt)
    response = requests.post(url=API_URL, headers=headers, json=payload).json()

    if 'error' in response:
        raise ValueError(response['error']['message'])
    return response['choices'][0]['message']['content']


def cache_image(image: Image.Image) -> str:
    image_filename = f"{uuid.uuid4()}.jpeg"
    os.makedirs(IMAGE_CACHE_DIRECTORY, exist_ok=True)
    image_path = os.path.join(IMAGE_CACHE_DIRECTORY, image_filename)
    image.save(image_path)
    return image_path


def respond(api_key: str, image: Image.Image, prompt: str, chat_history):
    if not api_key:
        raise ValueError(
            "API_KEY is not set. "
            "Please follow the instructions in the README to set it up.")

    flipped_image = preprocess_image(image=image)
    cached_image_path = cache_image(flipped_image)
    response = prompt_image(api_key=api_key, image=flipped_image, prompt=prompt)
    chat_history.append(((cached_image_path,), None))
    chat_history.append((prompt, response))
    return "", chat_history


def main():
    st.markdown(MARKDOWN)
    api_key = st.text_input("OpenAI API KEY", type="password")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    prompt = st.text_area("Enter your message:")
    chat_history = []

    if st.button("Submit"):
        if not api_key:
            st.error("API_KEY is not set. Please set it up.")
        elif uploaded_image is None:
            st.error("Please upload an image.")
        else:
            image = Image.open(uploaded_image)
            response, chat_history = respond(api_key, image, prompt, chat_history)
            st.success("Response: {}".format(response))

    # Display chat history
    for entry in chat_history:
        if entry[0]:
            for image_path in entry[0]:
                st.image(image_path, use_column_width=True)
        if entry[1]:
            st.text(entry[1])


if __name__ == "__main__":
    main()
