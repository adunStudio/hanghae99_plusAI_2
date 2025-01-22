import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate

class ImageChatService:
    def __init__(self, api_key, summary_max_token):

        self._api_key = api_key
        self._summary_max_token = summary_max_token

        # 한글 요약 프롬프트 정의d
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

        '''
        self.st.session_state.setdefault("common_memory", ConversationSummaryBufferMemory(
            llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key),
            max_token_limit=self._summary_max_token,
            #prompt=summary_prompt,
            verbose=False
        ))
        '''

        self._llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

        self._system_messages    = [SystemMessage(content='당신은 주어지는 이미지를 참고해서 응답하는 챗봇입니다.')]
        self._image_messages     = []
        self._common_messages    = []
        self._common_memory      = []
        self._summary_messages   = []
        self._truncated_messages = []

        self._waiting = False

    def answer_generate(self, prompt):
        self._add_human_message(prompt)

        result = self._llm.invoke(self._all_messages)
        response = result.content

        self._add_ai_message(response)

        #self._on_tokens_changed(callback.total_tokens)

        return response


    def answer_generate_stream(self, prompt):
        self._add_human_message(prompt)

        result_stream = self._llm.stream(self._all_messages)

        response = ""
        for chunk in result_stream:
            response += chunk.content
            yield chunk.content

        self._add_ai_message(response)

        # self._on_tokens_changed(callback.total_tokens)

    @property
    def waiting(self):
        return self._waiting

    def set_waiting(self, waiting: bool):
        self._waiting = waiting

    @property
    def have_message(self):
        return len(self._common_messages) > 0

    @property
    def chat_histories(self):
        return self._common_messages

    @property
    def _all_messages(self):
        return self._system_messages + self._image_messages + self._summary_messages + self._truncated_messages

    def add_image(self, image):
        base64_image = base64.b64encode(image.read()).decode("utf-8")
        self._add_image_message(base64_image)

    ####################################################################################################################
    # Private Method
    ####################################################################################################################
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
        human_message = HumanMessage(content=prompt)
        self._common_messages.append(human_message)
        self._truncated_messages.append(human_message)

    def _add_ai_message(self, prompt):
        ai_message = AIMessage(content=prompt)
        self._common_messages.append(ai_message)
        self._truncated_messages.append(ai_message)


    def _on_tokens_changed(self, tokens):
        return
        print(tokens)

        # 토큰 수가 임계값을 넘지 않으면 요약하지 않음
        if tokens < self._summary_max_token:
            return

        # _truncated_messages가 비어 있으면 요약하지 않음
        if len(self._truncated_messages) == 0:
            return


        print('요약 시작')
        previous_summary = ''
        if len(self._summary_messages) != 0:
            print("기존 요약 있다.")
            previous_summary = self._summary_messages[0].content

        new_summary_message = self._common_memory.predict_new_summary(self._truncated_messages, previous_summary)

        self._truncated_messages.clear()

        self._summary_messages = [AIMessage(content=f'{new_summary_message}')]

        print(f'이전 대화 요약: {new_summary_message}')





