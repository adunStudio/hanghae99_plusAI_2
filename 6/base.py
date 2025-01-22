import streamlit as st
import time

from langchain_core.messages import AIMessage
from service import ImageChatService


import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    OPENAI_API_SUMMARY_TOKEN_LIMIT = 5000



def main():
    st.session_state.setdefault("chat_service", ImageChatService(Config.OPENAI_API_KEY, Config.OPENAI_API_SUMMARY_TOKEN_LIMIT))
    chat_service = st.session_state.chat_service

    st.title("Image Chat Bot")

    if uploaded_images := st.file_uploader("함께할 이미지를 올려주세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True):

        for uploaded_image in uploaded_images:
            chat_service.add_image(uploaded_image)

        cols = st.columns(len(uploaded_images))

        for idx, col in enumerate(cols):
            with col:
                st.image(uploaded_images[idx], use_container_width=True, caption=uploaded_images[idx].name)

        with st.chat_message('ai'):
            st.markdown('안녕하세요. 무엇을 도와드릴까요?')

        for message in chat_service.chat_histories:
            role = 'ai' if isinstance(message, AIMessage) else 'human'

            with st.chat_message(role):
                st.markdown(message.content)


        selected_prompt = None
        if not chat_service.have_message:
            prepared_prompt = ['주어진 두 사진의 공통점이 뭐야?', '주어진 두 사진의 차이점이 뭐야?']
            columns = st.columns(len(prepared_prompt))
            for idx, colum in enumerate(columns):
                if colum.button(prepared_prompt[idx], use_container_width=True, disabled=chat_service.waiting, on_click=chat_service.set_waiting, args=[True]):
                    selected_prompt = prepared_prompt[idx]


        prompt = st.chat_input("메시지를 입력하세요.", disabled=chat_service.waiting, on_submit=chat_service.set_waiting(True))
        if selected_prompt or prompt:
            with st.chat_message("user"):
                st.markdown(selected_prompt or prompt)

            with st.chat_message("assistant"):
                st.write_stream(chat_service.answer_generate_stream(selected_prompt or prompt))
                #st.markdown(chat_service.answer_generate(selected_prompt or prompt))

                st.toast('답변 완료!', icon='🎉')
                time.sleep(1)

                chat_service.set_waiting(False)
                st.rerun()


if __name__ == "__main__":
    main()