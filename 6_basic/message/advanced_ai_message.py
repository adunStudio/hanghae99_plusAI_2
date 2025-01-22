from langchain_core.messages import AIMessage
import tiktoken


class AdvancedAIMessage(AIMessage):
    _enc: tiktoken.Encoding = None

    def __init__(self, content: str, check_tokens: bool = True):
        super().__init__(content=content)

        self._tokens = 0

        if not isinstance(AdvancedAIMessage._enc, tiktoken.Encoding):
            AdvancedAIMessage._enc = tiktoken.encoding_for_model("gpt-4o-mini")

        if check_tokens:
            content = self.content if isinstance(self.content, str) else f'{content}'
            self._tokens = len(AdvancedAIMessage._enc.encode(content))

    @property
    def role(self) -> str:
        return 'ai'

    @property
    def tokens(self) -> int:
        return self._tokens
