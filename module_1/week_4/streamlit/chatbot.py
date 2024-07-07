import streamlit as st
from hugchat import hugchat
from hugchat.login import Login

# Add title
st.title('Simple Chatbot')

# Hugging Face Account
with st.sidebar:
    st.title('Hugging Face Account')
    hf_email = st.text_input('E-mail')
    hf_password = st.text_input('Password', type='password')

    if not hf_email or not hf_password:
        st.warning('Please enter your Hugging Face Account!')
    else:
        st.success('Hugging Face Account is set!')

# Store LLM generated responses
if 'messages' not in st.session_state.keys():
    st.session_state.messages = [{'role': 'assistant',
                                 'content': 'Hello! How can I help you?'}]
# Display stored messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Function for generating LLM response


def generate_response(prompt_input, email, password):
    # Hugging Face Login
    sign = Login(email, password)
    cookies = sign.login()
    # Create ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)


if prompt := st.chat_input(disabled=not (hf_email or not hf_password)):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.write(prompt)


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            response = generate_response(prompt, hf_email, hf_password)
            st.write(response)
    messages = {'role': 'assistant', 'content': response}
    st.session_state.messages.append(messages)
