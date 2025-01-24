# [6주차] 기본과제: 이미지를 가지고 질의응답을 진행할 수 있는 챗봇 구현

### 시연 영상
![시연](./image/demo.gif)

### 프로젝트 구조
```
─── main.py (Streamlit 앱의 진입점)

└── service
    └── ImageChatService.py (LLM 채팅 서비스)
    
└── conversation
    └── SummaryBufferConversation.py (자동 메시지 요약 자료구조)
    
└── message (토큰 수 계산 기능 추가된 메시지 클래스)
    ├── AdvancedAImessage.py    
    └── AdvancedHumanMessage.py

└── test
    └── 테스트할 때 사용한 파일들
    
└── image
    └── 테스트에 사용한 이미지들
```