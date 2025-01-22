from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from message import AdvancedAIMessage, AdvancedHumanMessage


# 일정 개수(버퍼) 이상 지난 대화 메세지들의 토큰 합이 max_token보다 커지면 자동 요약해주는 자료구조
class SummaryBufferConversation:
    def __init__(self, api_key, max_token, buffer_count=5):
        self._api_key = api_key
        self._max_token = max_token        # 맥스 토큰
        self._buffer_count = buffer_count  # 최소 유지 개수

        self._summary  = []
        self._messages = []

        self.__init_chain()

    def __init_chain(self):
        summary_prompt = PromptTemplate(
            input_variables=["previous_summary", "messages", "max_token"],
            template=(
                "다음은 지금까지의 대화 요약입니다:\n"
                "{previous_summary}\n\n"
                "다음은 새로 추가된 대화들입니다:\n"
                "{messages}\n\n"
                "이전 대화와 추가된 대화를 요약해 한글로 업데이트하세요:"
                "결과는 {max_token} 이내로 해주세요."
            ),
        )

        summary_llm = ChatOpenAI(model="gpt-4o-mini", api_key=self._api_key, max_tokens=self._max_token)

        self._chain = summary_prompt | summary_llm

    ####################################################################################################################
    # Public Method
    ####################################################################################################################

    def append(self, message):
        self._messages.append(message)

    def get(self):
        print(f'현재 토큰: {self._total_tokens }')

        self._call_summary()

        return self._summary + self._messages

    ####################################################################################################################
    # Private Method
    ####################################################################################################################

    def _call_summary(self):
        if len(self._messages) <= self._buffer_count:
            return

        total_token = self._get_total_token()
        if total_token <= self._max_token:
            return

        print(f'요약 시작({total_token})')

        messages = self._messages[:-self._buffer_count]
        self._messages = self._messages[-self._buffer_count:]

        previous_summary = '' if len(self._summary) == 0 else self._summary[0].content

        result = self._chain.invoke({'previous_summary': previous_summary, 'messages': messages, 'max_token': self._max_token})

        self._summary = [AdvancedAIMessage(content=result.content)]

        print(f'요약 완료({self._summary[0].tokens})')
        print(self._summary[0].content)

    def _get_total_token(self):
        if len(self._messages) <= self._buffer_count:
            return 0

        return sum(message.tokens for message in self._messages[:-self._buffer_count])
