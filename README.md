# 🎬 AI 동영상 검색 시스템

AWS Bedrock + Rekognition을 활용한 스마트 비디오 분석 및 자연어 검색 시스템

## 🚀 주요 기능

### 📤 비디오 업로드 & AI 분석
- 드래그 앤 드롭으로 간편한 비디오 업로드
- 스마트 프레임 추출 (장면 변화 감지 알고리즘)
- AWS Rekognition을 통한 자동 AI 분석
- 실시간 처리 상태 모니터링

### 🔍 자연어 기반 장면 검색
- **AWS Bedrock (Claude)** 활용한 질의 분석
- 한국어 검색 지원: "칼싸움하는 장면", "웃는 사람", "자동차가 나오는 부분" 등
- 키워드 자동 확장 및 매핑
- 점수 기반 결과 랭킹 시스템

### 📚 비디오 라이브러리
- 업로드된 모든 비디오 통합 관리
- 상세 분석 결과 확인
- 프레임별 AI 분석 데이터 시각화

## 🤖 AI 기술 스택

- **AWS Bedrock**: 자연어 처리 및 질의 분석
- **AWS Rekognition**: 객체/얼굴/텍스트/콘텐츠 감지
- **OpenCV**: 스마트 프레임 추출 및 장면 변화 감지
- **S3**: 비디오 파일 및 메타데이터 저장

## 💡 스마트 기능들

- **장면 변화 감지**: 히스토그램 비교를 통한 효율적 프레임 추출
- **병렬 처리**: ThreadPoolExecutor를 활용한 빠른 AI 분석
- **한국어-영어 키워드 매핑**: 자동 번역 및 확장
- **검색 결과 하이라이팅**: 매칭된 프레임 강조 표시

## 🛠️ 설치 및 실행

### 필수 요구사항
```bash
# Python 패키지
pip install flask boto3 opencv-python-headless pillow anthropic

# 시스템 패키지 (Ubuntu)
sudo apt install ffmpeg
```

### AWS 설정
1. AWS CLI 설정 및 인증
2. S3 버킷 생성
3. IAM 역할에 다음 권한 추가:
   - Rekognition: DetectLabels, DetectFaces, DetectText, DetectModerationLabels
   - Bedrock: InvokeModel
   - S3: GetObject, PutObject, ListBucket

### 실행
```bash
python ai_video_search_app.py
```

웹 브라우저에서 `http://localhost:8082` 접속

## 📊 사용 예시

### 검색 질의 예시
- "칼싸움하는 장면을 찾아줘"
- "웃는 사람이 나오는 부분"
- "자동차가 등장하는 장면"
- "음식을 먹는 장면"
- "춤추는 사람"

### 분석 결과
각 프레임별로 다음 정보를 제공:
- **객체/장면 라벨**: 감지된 물체나 상황 (신뢰도 포함)
- **얼굴 분석**: 나이, 성별, 감정 상태
- **텍스트 감지**: 영상 내 텍스트 추출
- **콘텐츠 조정**: 부적절한 콘텐츠 감지

## 🏗️ 시스템 아키텍처

```
[비디오 업로드] 
    ↓
[스마트 프레임 추출] (OpenCV)
    ↓
[병렬 AI 분석] (Rekognition)
    ↓
[메타데이터 저장] (S3)
    ↓
[자연어 질의] (사용자)
    ↓
[질의 분석] (Bedrock)
    ↓
[장면 검색] (매칭 알고리즘)
    ↓
[결과 반환] (점수 기반 랭킹)
```

## 💰 예상 비용 (월 기준)

### 소규모 사용 (월 20개 비디오)
- EC2 (t3.large): $60
- Bedrock: $3
- Rekognition: $42
- S3: $1
- **총 비용: ~$106/월**

### 중규모 사용 (월 100개 비디오)
- EC2 (t3.large): $60
- Bedrock: $7
- Rekognition: $210
- S3: $1
- **총 비용: ~$278/월**

## 🔧 커스터마이징

### 키워드 매핑 추가
`ai_video_search_app.py`의 `keyword_mapping` 딕셔너리에 새로운 매핑 추가:

```python
self.keyword_mapping = {
    "새로운_한국어": ["new", "english", "keywords"],
    # ...
}
```

### 분석 파라미터 조정
- 프레임 추출 빈도: `fps` 파라미터 조정
- 신뢰도 임계값: `MinConfidence` 값 변경
- 최대 프레임 수: `extracted_count` 제한 조정

## 📝 라이선스

MIT License

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

---

**🎬 AI 동영상 검색 시스템으로 비디오 콘텐츠를 더 스마트하게 관리하세요!**
