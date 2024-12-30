import streamlit as st
from streamlit_chat import message
import random
from src.ai_agent import chat_response


st.set_page_config(page_title='🤖 Hidroelectrica Chat APP', layout='centered', page_icon='🤖')
st.title("🤖 Hidroelectrica Chat AI")

# adding session state to each user session
session_id = random.randint(0, 100000)
# adding session_id to session state
if "session_id" not in st.session_state:
    st.session_state.session_id = session_id

# initial message
INIT_MESSAGE = {"role": "assistant",
                "content": "Hello! I am you Hidroelectrica Chat Agent, I will help answer all questions you might have about Hidroelectrica."}


if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]

def generate_response(input_text):
    output = chat_response(user_query=input_text)
    return output

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input(placeholder="Your message ....", key="input")

# display user input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    user_message = st.chat_message("user")
    user_message.write(user_input)

# Generate response
if st.session_state.messages[-1]["role"] != "assistant":
    response = generate_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    assistant_message = st.chat_message("assistant")
    assistant_message.write(response)