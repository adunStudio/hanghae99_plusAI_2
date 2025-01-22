import streamlit as st
import time

from service import ImageChatService

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    SUMMARY_TOKEN_LIMIT = 10000
    CHAT_BUFFER_COUNT = 10


def main():
    # 🔹 이미지 채팅 서비스 객체 세션
    st.session_state.setdefault("chat_service", ImageChatService(Config.OPENAI_API_KEY, Config.SUMMARY_TOKEN_LIMIT, Config.CHAT_BUFFER_COUNT))
    chat_service = st.session_state.chat_service

    # 🔹 서비스 타이틀
    st.title('💬✨ **이미지 톡톡** ️💫')

    if uploaded_images := st.file_uploader("✨ 함께할 이미지를 올려볼까요? 🌈", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True):

        # 🔹 1. 이미지 업로드
        for uploaded_image in uploaded_images:
            chat_service.add_image(uploaded_image)

        cols = st.columns(len(uploaded_images))

        # 🔹 2. 이미지 노출
        for idx, col in enumerate(cols):
            with col:
                st.image(uploaded_images[idx], use_container_width=True, caption=uploaded_images[idx].name)

        # 🔹 3. 시작 메시지
        with st.chat_message('ai'):
            st.markdown('💡 **안녕하세요!** 이미지를 보면서 이야기를 나눠봐요. 궁금한 점을 물어보세요! 😊')

        # 🔹 4. 채팅 메시지 히스토리
        for message in chat_service.chat_histories:
            with st.chat_message(message.role):
                st.markdown(message.content)

        # 🔹5. 기존에 메시지 없다면 -> 준비된 목록 선택 가능
        selected_prompt = None
        if not chat_service.have_message:
            prepared_title = ['🤔 **두 사진의 공통점은?**', '🔍 **두 사진의 차이점은?**']
            prepared_prompt = ['두 사진의 공통점은?', '두 사진의 차이점은?']
            columns = st.columns(len(prepared_title))
            for idx, colum in enumerate(columns):
                if colum.button(prepared_title[idx], use_container_width=True, disabled=chat_service.waiting,
                                on_click=chat_service.set_waiting, args=[True]):
                    selected_prompt = prepared_prompt[idx]

        # 🔹 6. 메시지 직접 입력
        prompt = st.chat_input("📝 메시지를 입력해보세요! 😊", disabled=chat_service.waiting, on_submit=chat_service.set_waiting(True))
        if selected_prompt or prompt:

            with st.chat_message("user"):
                st.markdown(selected_prompt or prompt)

            # 🔹 7. 메시지 요청 & 답변
            with st.chat_message("assistant"):
                st.write_stream(chat_service.answer_generate_stream(selected_prompt or prompt))
                # st.markdown(chat_service.answer_generate(selected_prompt or prompt))

                st.toast('✨ **답변 완료!** 🎉', icon='🎉')
                time.sleep(1)

                chat_service.set_waiting(False)
                st.rerun()


if __name__ == "__main__":
    main()
