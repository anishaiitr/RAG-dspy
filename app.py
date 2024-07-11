import streamlit as st
import sys
sys.path.insert(0, '../')
from rag import prepare_data, get_response

st.title('Insurance product QA')

if 'messages' not in st.session_state:
    prepare_data()
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('Write your query here:'):
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message("assistant"):
        stream = get_response(prompt)
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})


