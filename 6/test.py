import base64
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


st.title("Fashion Recommendation Bot")
model = ChatOpenAI(model="gpt-4o-mini")
if image := st.file_uploader("본인의 전신이 보이는 사진을 올려주세요!", type=['png', 'jpg', 'jpeg']):
    st.image(image)
    image = base64.b64encode(image.read()).decode("utf-8")
    with st.chat_message("assistant"):
        message = HumanMessage(
            content=[
                {"type": "text", "text": "사람의 전신이 찍혀있는 사진이 한 장 주어집니다. 이 때, 사진 속 사람과 어울리는 옷 및 패션 스타일을 추천해주세요."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ],
        )
        result = model.invoke([message])
        response = result.content
        st.markdown(response)