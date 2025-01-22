import base64
import hashlib

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from message import AdvancedAIMessage, AdvancedHumanMessage
from conversation import SummaryBufferConversation


class ImageChatService:
    def __init__(self, api_key, max_token, buffer_count):

        self._api_key = api_key

        self._llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

        self._image_hashes = []

        self._system_messages = [SystemMessage(content='당신은 주어지는 이미지를 참고해서 응답하는 챗봇입니다.')]
        self._image_messages  = []
        self._chat_messages   = [] # 전체 대화 기록
        self._summary_messages = SummaryBufferConversation(api_key, max_token, buffer_count) # 자동 요약

        self._waiting = False

    ####################################################################################################################
    # Public Method
    ####################################################################################################################
    def add_image(self, image):
        image_data = image.read()
        image_hash = hashlib.md5(image_data).hexdigest()  # 해시값 계산
        if image_hash in self._image_hashes:
            return

        self._image_hashes.append(image_hash)
        base64_image = base64.b64encode(image_data).decode("utf-8")
        self._add_image_message(base64_image)

    def answer_generate(self, prompt):
        self._add_human_message(prompt)

        result = self._llm.invoke(self._all_messages)
        response = result.content

        self._add_ai_message(response)

        return response

    def answer_generate_stream(self, prompt):
        self._add_human_message(prompt)

        result_stream = self._llm.stream(self._all_messages)

        response = ''
        for chunk in result_stream:
            response += chunk.content
            yield chunk.content

        self._add_ai_message(response)

    def set_waiting(self, waiting: bool):
        self._waiting = waiting

    ####################################################################################################################
    # Property
    ####################################################################################################################
    @property
    def waiting(self):
        return self._waiting

    @property
    def have_message(self):
        return len(self._chat_messages) > 0

    @property
    def chat_histories(self):
        return self._chat_messages

    @property
    def _all_messages(self):
        return self._system_messages + self._image_messages + self._summary_messages.get()

    ####################################################################################################################
    # Private Method
    ####################################################################################################################
    def _add_image_message(self, base64_image):
        image_message = AdvancedHumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
            check_tokens=False
        )
        self._image_messages.append(image_message)

    def _add_human_message(self, prompt):
        human_message = AdvancedHumanMessage(content=prompt)
        self._chat_messages.append(human_message)
        self._summary_messages.append(human_message)

    def _add_ai_message(self, prompt):
        ai_message = AdvancedAIMessage(content=prompt)
        self._chat_messages.append(ai_message)
        self._summary_messages.append(ai_message)