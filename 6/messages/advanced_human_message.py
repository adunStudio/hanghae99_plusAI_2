from langchain_core.messages import HumanMessage


class AdvancedHumanMessage(HumanMessage):
    def __init__(self, content):
        super().__init__(content=content)
        self._role = 'human'

    @property
    def role(self):
        return self._role
