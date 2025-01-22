from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from messages import AdvancedAIMessage, AdvancedHumanMessage


class SummaryMessages:
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
                "요약 결과는 최대 {max_token}이하이어야 합니다."
            ),
        )

        self._llm = ChatOpenAI(model="gpt-4o-mini", api_key=self._api_key)

        self._chain = summary_prompt | self._llm

    def append(self, message):
        self._messages.append(message)

    def get(self):
        print(f'현재 토큰: {self._total_tokens }')
        if self._total_tokens <= self._max_token:
            return self._summary + self._messages

        self._call_summary()

        return self._summary + self._messages

    def _call_summary(self):
        if len(self._messages) <= self._buffer_count:
            return

        print(f'요약 시작({self._total_tokens})')

        messages = self._messages[:-self._buffer_count]
        self._messages = self._messages[-self._buffer_count:]

        previous_summary = '' if len(self._summary) == 0 else self._summary[0].content

        result = self._chain.invoke({'previous_summary': previous_summary, 'messages': messages, 'max_token': self._max_token})

        self._summary = [AdvancedAIMessage(content=result.content)]

        print(f'요약 완료({self._summary[0].tokens})')

    @property
    def _total_tokens(self):
        if len(self._messages) <= self._buffer_count:
            return 0

        return sum(message.tokens for message in self._messages[:-self._buffer_count])
