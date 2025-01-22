from langchain_core.messages import HumanMessage
import tiktoken


class AdvancedHumanMessage(HumanMessage):
    _enc: tiktoken.Encoding = None

    def __init__(self, content: str, check_tokens: bool = True):
        super().__init__(content=content)

        self._role = 'human'
        self._tokens = 0

        if not isinstance(AdvancedHumanMessage._enc, tiktoken.Encoding):
            AdvancedHumanMessage._enc = tiktoken.encoding_for_model("gpt-4o-mini")

        if check_tokens:
            content = self.content if isinstance(self.content, str) else f'{content}'
            self._tokens = len(AdvancedHumanMessage._enc.encode(content))

    @property
    def role(self) -> str:
        return self._role

    @property
    def tokens(self) -> int:
        return self._tokens
