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
        self.st.session_state.setdefault("image_messages", [])
        self.st.session_state.setdefault("common_messages", [])

        self.system_messages = [SystemMessage(content='당신은 주어지는 이미지를 참고해서 응답하는 챗봇입니다.')]
        self.image_messages  = self.st.session_state.image_messages
        self.common_messages = self.st.session_state.common_messages


        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    @property
    def waiting(self):
        return self.st.session_state.waiting

    def set_waiting(self, waiting: bool):
        self.st.session_state.waiting = waiting

    @property
    def have_message(self):
        return len(self.common_messages) > 0

    def add_image(self, image):
        base64_image = base64.b64encode(image.read()).decode("utf-8")
        self._add_image_message(base64_image)


    def _add_image_message(self, base64_image):
        image_message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        )
        self.image_messages.append(image_message)

    def _add_human_message(self, prompt):
        self.common_messages.append(HumanMessage(content=prompt))

    def _add_ai_message(self, prompt):
        self.common_messages.append(AIMessage(content=prompt))

    @property
    def _messages(self):
        return self.system_messages + self.image_messages + self.common_messages

    def answer_generate(self, prompt):
        self._add_human_message(prompt)

        result = self.llm.invoke(self._messages)
        response = result.content

        self._add_ai_message(result)

        return response

    def answer_generate_stream(self, prompt):
        self._add_human_message(prompt)

        result_stream = self.llm.stream(self._messages)

        response = ""
        for chunk in result_stream:
            response += chunk.content
            yield chunk.content

        self._add_ai_message(response)


def main():
    chat_service = ImageChatService(st, Config.OPENAI_API_KEY)

    st.title("Image Chat Bot")

    if uploaded_images := st.file_uploader("함께할 이미지를 올려주세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True):

        for uploaded_image in uploaded_images:
            chat_service.add_image(uploaded_image)

        image_count = len(uploaded_images)
        cols = st.columns(image_count)

        for idx, col in enumerate(cols):
            with col:
                st.image(uploaded_images[idx], use_container_width=True, caption=uploaded_images[idx].name)

        with st.chat_message('ai'):
            st.markdown('안녕하세요. 무엇을 도와드릴까요?')

        for message in chat_service.common_messages:
            role = 'ai' if isinstance(message, AIMessage) else 'human'

            with st.chat_message(role):
                st.markdown(message.content)

        selected_prompt = None
        if not chat_service.have_message:
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