import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import tempfile
import os
import time
# Streamlit page configuration
st.set_page_config(page_title="AI Chat App", page_icon="🤖", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
body {
    color: #333;
    background-color: #f0f2f6;
}

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}
.chat-message.user {
    background-color: #2b313e;
    color: white;
}
.chat-message.bot {
    background-color: #475063;
    color: white;
}
.conversation-starter {
    cursor: pointer;
    padding: 0.5rem;
    border-radius: 0.3rem;
    background-color: #e0e0e0;
    margin-bottom: 0.5rem;
    color: #333;
}
.conversation-starter:hover {
    background-color: #d0d0d0;
}
</style>
""", unsafe_allow_html=True)

# API keys
OPENAI_API_KEY = "Your_key"
ELEVENLABS_API_KEY = "Your_key"

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Initialize the ChatOpenAI model
@st.cache_resource
def get_llm():
    return ChatOpenAI(temperature=0.7, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Chat template and chain
persona = "Zoe"
prompt_template = ChatPromptTemplate.from_template(
    "You are {persona}, a friendly and engaging chatbot. "
    "Respond in Hinglish only. The person is romantically interested in you.\n\n"
    "Human: {human_input}\n"
    "{persona}:"
)

@st.cache_resource
def get_chain(_llm):
    return LLMChain(llm=_llm, prompt=prompt_template)

@st.cache_data
def text_to_speech(text: str) -> str:
    with st.spinner("Recording voice..."):
        response = client.text_to_speech.convert(
            voice_id="EXAVITQu4vr4xnSDxMaL",    # Emily   Sarah -> #EXAVITQu4vr4xnSDxMaL
            optimize_streaming_latency=0,
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.3,
                similarity_boost=0.8,
                style=0.2,
                use_speaker_boost=True,
            ),
        )
        
        # Collect the audio data
        audio_data = b''.join(response)
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_data)
            return temp_audio.name

# Main app
st.title("AI Chat App")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Conversation starters
st.sidebar.markdown("## Conversation Starters")
starters = [
    "Hey! Kya chal raha hai aaj kal?",
    "Tumhara favorite movie kaunsa hai?",
    "Kya tum mere saath ek virtual date pe chalna chahogi?"
]

for starter in starters:
    if st.sidebar.button(starter):
        st.session_state.messages.append({"role": "user", "content": starter})
        st.experimental_rerun()

# User input
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    llm = get_llm()
    chain = get_chain(llm)
    response = chain.run(persona=persona, human_input=user_input)
    ai_response = response.strip()

    # Add a 2-second pause
    time.sleep(3)

    # Display AI response and play audio
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        # Create a placeholder for the animated text
        text_placeholder = st.empty()

        # Generate audio
        with st.spinner("Recording voice..."):
            audio_file_path = text_to_speech(ai_response)

        # Animate the text
        displayed_text = ""
        for char in ai_response:
            displayed_text += char
            text_placeholder.markdown(displayed_text + "▌")
            time.sleep(0.05)  # Adjust this value to change the typing speed

        # Display the final text without the cursor
        text_placeholder.markdown(ai_response)

        # Play the audio
        st.audio(audio_file_path, format="audio/mp3", start_time=0, autoplay = True)

    # Clean up the temporary audio file
    os.unlink(audio_file_path)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by OpenAI and ElevenLabs")