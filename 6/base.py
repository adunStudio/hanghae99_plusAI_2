import os
import base64
import streamlit as st
import time

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

load_dotenv()

class Config:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    OPENAI_API_SUMMARY_TOKEN_LIMIT = 3000


class ImageChatService:
    def __init__(self, st, api_key, summary_max_token):
        self.st = st

        self.st.session_state.setdefault("waiting", False)
        self.st.session_state.setdefault("image_messages", [])
        self.st.session_state.setdefault("common_messages", [])

        # 한글 요약 프롬프트 정의
        summary_prompt = PromptTemplate(
            input_variables=["summary", "new_lines"],
            template=(
                "다음은 지금까지의 대화 요약입니다:\n"
                "{summary}\n\n"
                "다음은 새로 추가된 대화입니다:\n"
                "{new_lines}\n\n"
                "이 대화를 기반으로 대화 요약을 한글로 업데이트하세요:"
            ),
        )

        self.st.session_state.setdefault("common_memory", ConversationSummaryMemory(
            llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key),
            max_token_limit=summary_max_token,
            prompt=summary_prompt,
            verbose=False
        ))

        self.st.session_state.setdefault("llm",ChatOpenAI(model="gpt-4o-mini", api_key=api_key))


        self._system_messages = [SystemMessage(content='당신은 주어지는 이미지를 참고해서 응답하는 챗봇입니다.')]
        self._image_messages  = self.st.session_state.image_messages
        self._common_messages = self.st.session_state.common_messages
        self._common_memory  = self.st.session_state.common_memory

        self._llm = self.st.session_state.llm


    @property
    def waiting(self):
        return self.st.session_state.waiting

    def set_waiting(self, waiting: bool):
        self.st.session_state.waiting = waiting

    @property
    def have_message(self):
        return len(self._common_messages) > 0

    @property
    def chat_histories(self):
        return self._common_messages

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
        self._image_messages.append(image_message)

    def _add_human_message(self, prompt):
        self._common_messages.append(HumanMessage(content=prompt))

    def _add_ai_message(self, prompt):
        self._common_messages.append(AIMessage(content=prompt))

    @property
    def _last_human_content(self):
        return self._common_messages[-1].content

    @property
    def _messages(self):
        return self._system_messages + self._image_messages + self._common_messages

    def answer_generate(self, prompt):
        self._add_human_message(prompt)

        result = self._llm.invoke(self._messages)
        print(type(result))
        print(result)
        response = result.content

        self._add_ai_message(result)

        #print(self._common_memory.load_memory_variables({})["history"])
        return response

    def answer_generate_stream(self, prompt):
        self._add_human_message(prompt)

        result_stream = self._llm.stream(self._messages)

        response = ""
        for chunk in result_stream:
            response += chunk.content
            yield chunk.content

        self._add_ai_message(response)


def main():
    chat_service = ImageChatService(st, Config.OPENAI_API_KEY, Config.OPENAI_API_SUMMARY_TOKEN_LIMIT)

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