import streamlit as st
from transformers import pipeline

# Load the text classification model pipeline
classifier = pipeline("text-classification",
                      model='isom5240ust/bert-base-uncased-emotion',
                      return_all_scores=True)

# Streamlit application title
st.title("Text Classification for you")
st.write("Classification for 6 emotions: sadness, joy, love, anger, fear, surprise")

# Text input for user to enter the text to classify
text = st.text_area("Enter the text to classify", "")

# Perform text classification when the user clicks the "Classify" button
if st.button("Classify"):
    # Perform text classification on the input text
    results = classifier(text)[0]

    # Display the classification result
    max_score = float('-inf')
    max_label = ''

    for result in results:
        if result['score'] > max_score:
            max_score = result['score']
            max_label = result['label']

    st.write("Text:", text)
    st.write("Label:", max_label)
    st.write("Score:", max_score)

# Program title: Simple Storytelling App (Text to Story + Audio)

import streamlit as st
from transformers import pipeline

# Set up the page
st.set_page_config(page_title="Text to Audio Story", page_icon="🦜")
st.header("Turn Your Text into an Audio Story")

# User enters text
user_text = st.text_area("Enter a prompt or scenario for your story:")

if user_text:
    # Stage 1: Text to Story
    st.text('Generating a story...')
    story_generator = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story = story_generator(user_text)[0]['generated_text']
    st.write(story)

    # Stage 2: Story to Audio
    st.text('Generating audio data...')
    audio_generator = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    speech_output = audio_generator(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = speech_output["audio"]
        sample_rate = speech_output["sampling_rate"]
        # Play audio directly using Streamlit
        st.audio(audio_array,
                 sample_rate=sample_rate)
