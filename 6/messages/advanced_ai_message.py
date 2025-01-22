from langchain_core.messages import AIMessage


class AdvancedAIMessage(AIMessage):
    def __init__(self, content):
        super().__init__(content=content)
        self._role = 'ai'

    @property
    def role(self):
        return self._role
