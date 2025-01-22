"""


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
"""