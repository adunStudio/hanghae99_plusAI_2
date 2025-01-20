import os
import base64
import streamlit as st
import time

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

class Config:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

class ImageChatService:
    def __init__(self, st, api_key):
        self.st = st

        self.st.session_state.setdefault("waiting", False)
        self.st.session_state.setdefault("messages", [
            SystemMessage(content='당신은 주어지는 이미지를 참고해서 응답하는 챗봇입니다.'),
            AIMessage(content='안녕하세요! 무엇을 도와드릴까요?')
        ])

        self.messages = self.st.session_state.messages

        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    @property
    def waiting(self):
        return self.st.session_state.waiting

    def set_waiting(self, waiting: bool):
        self.st.session_state.waiting = waiting



    def add_image(self, base64_image):
        image_message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        )
        self.messages.append(image_message)

    def answer_generate(self, prompt):
        self.messages.append(HumanMessage(content=prompt))

        result = self.llm.invoke(self.messages)

        response = result.content
        self.messages.append(AIMessage(content=response))
        return response

    def answer_generate_stream(self, prompt):
        self.messages.append(HumanMessage(content=prompt))

        result_stream = self.llm.stream(self.messages)

        response = ""
        for chunk in result_stream:
            response += chunk.content
            yield chunk.content

        self.messages.append(AIMessage(content=response))


def main():
    chat_service = ImageChatService(st, Config.OPENAI_API_KEY)

    have_human_message = False

    st.title("Image Chat Bot")

    if uploaded_images := st.file_uploader("함께할 이미지를 올려주세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True):
        # 업로드된 파일 처리
        for uploaded_image in uploaded_images:
            base64_image = base64.b64encode(uploaded_image.read()).decode("utf-8")
            chat_service.add_image(base64_image)

        image_count = len(uploaded_images)
        cols = st.columns(image_count)

        for idx, col in enumerate(cols):
            with col:
                st.image(uploaded_images[idx], use_container_width=True, caption=uploaded_images[idx].name)

        for message in chat_service.messages:
            role = 'system' if isinstance(message, SystemMessage) else \
                    'ai' if isinstance(message, AIMessage) else \
                    'human'

            if role == 'system':
                # 시스템 메시지는 무시
                continue

            if isinstance(message.content, list):
                # 이미지 메시지는 무시
                continue

            if role == 'human':
                have_human_message = True

            with st.chat_message(role):
                st.markdown(message.content)


        selected_prompt = None

        if have_human_message == False:
            left, right = st.columns(2)

            if left.button("주어진 두 사진의 공통점이 뭐야?", use_container_width=True, disabled=chat_service.waiting, on_click=chat_service.set_waiting, args=[True]):
                selected_prompt = '주어진 두 사진의 공통점이 뭐야?'
            if right.button("주어진 두 사진의 차이점이 뭐야?", icon=":material/mood:", use_container_width=True, disabled=chat_service.waiting, on_click=chat_service.set_waiting,  args=[True]):
                selected_prompt = '주어진 두 사진의 차이점이 뭐야?'


        prompt = st.chat_input("메시지를 입력하세요.", disabled=chat_service.waiting, on_submit=chat_service.set_waiting(True))

        if selected_prompt or prompt:
            with st.chat_message("user"):
                st.markdown(selected_prompt or prompt)

            with st.chat_message("assistant"):
                st.write_stream(chat_service.answer_generate_stream(selected_prompt or prompt))
                st.toast('답변 완료!', icon='🎉')
                time.sleep(1)

                chat_service.set_waiting(False)
                st.rerun()

if __name__ == "__main__":
    main()