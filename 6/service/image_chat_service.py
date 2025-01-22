import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from messages import AdvancedAIMessage, AdvancedHumanMessage



class ImageChatService:
    def __init__(self, api_key, summary_max_token):

        self._api_key = api_key
        self._summary_max_token = summary_max_token

        self._llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

        self._system_messages = [SystemMessage(content='당신은 주어지는 이미지를 참고해서 응답하는 챗봇입니다.')]
        self._image_messages  = []
        self._common_messages = []

        self._waiting = False

    ####################################################################################################################
    # Public Method
    ####################################################################################################################
    def add_image(self, image):
        base64_image = base64.b64encode(image.read()).decode("utf-8")
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

    @property
    def waiting(self):
        return self._waiting

    ####################################################################################################################
    # Property
    ####################################################################################################################
    @property
    def have_message(self):
        return len(self._common_messages) > 0

    @property
    def chat_histories(self):
        return self._common_messages

    @property
    def _all_messages(self):
        return self._system_messages + self._image_messages + self._common_messages

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
        self._common_messages.append(human_message)

    def _add_ai_message(self, prompt):
        ai_message = AdvancedAIMessage(content=prompt)
        self._common_messages.append(ai_message)