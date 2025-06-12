# FOODING-ai

## AI 이용 기능
1. 이미지/영수증 인식 (YOLO + OCR)
2. 메뉴 추천 (재고 기반 GPT 프롬프트 엔지니어링)
3. 식재료 조언 (주간 소비 + 메뉴 분석 → GPT 기반 CSV 리포트)
4. 구매 주기 예측 (RandomForest 머신러닝 모델)

## 폴더 구조
- `OCR/`              : 영수증 OCR → 텍스트 추출 코드  
- `YOLO_train_test/`  : YOLOv8 모델 학습/테스트 코드  
- `YOLO_result/`      : 학습된 모델 결과 
- `filtering/`        : 식재료 이미지 정제  
- `menu_recomm/`      : 보유 재고 기준 메뉴 추천 모듈  
- `ingredient_recomm/`: 주간 소비 분석 → 식재료 조언 모듈  
- `purchase_prediction.ipynb`: 구매 주기 및 수량 예측 실험 코드
- `web_crawling/`     : 식재료 이미지 크롤링 스크립트  
- `my_flask_app/`     : Flask 서버  

## How to Build
1. OCR (영수증 텍스트 추출)
- 별도 빌드 필요 없음
- API Key, 이미지 파일, 경로를 코드 상에서 직접 지정 필요

2. YOLOv8 (식재료 이미지 인식)
- YOLOv8 모델 학습에는 커스텀 데이터셋(data.yaml 기반)과 ultralytics 라이브러리를 사용
- 학습된 weight는 YOLO_result/ 폴더에 best.pt로 저장

3. OpenAI 기반 GPT4o (메뉴 추천 / 식재료 조언)
- GPT 모델은 OpenAI API(gpt-4o)를 통해 사용
- 사용자의 보유 식재료 또는 식단 소비 기록 데이터를 기반으로 프롬프트를 구성하고 GPT에게 JSON/CSV 형식으로 직접 응답을 요청
- 메뉴 추천 기능
: /inventory API를 통해 사용자 보유 식재료 목록을 받아온 후, GPT에게 "가장 오래된 식재료부터 최대한 많이 소진할 수 있는 메뉴"를 추천받음
- 식재료 조언 기능
: /purchases, /menu_history API를 통해 사용자의 식단 소비 내역을 받아온 후, 영양소 부족/과잉 여부를 분석하고 보충이 필요한 식재료를 추천받음

4. 구매 주기 예측


## How to Install
1. OCR (영수증 텍스트 추출)
   pip install requests
   : Google Colab 환경 기준이며, 추가적으로 google.colab, base64, uuid, re, json, time 모듈 사용

2. YOLOv8 (식재료 이미지 인식)
   pip install -U ultralytics==8.2.103 torch numpy
   + Colab 환경에서는 nvidia-smi로 GPU 상태를 확인할 수 있음

3. OpenAI 기반 GPT4o (메뉴 추천 / 식재료 조언)
   pip install openai pandas requests
   + .env 또는 환경변수에 다음과 같은 형식으로 OpenAI API 키 등록
   export OPENAI_API_KEY=sk-xxxxxx
   openai.api_key = "sk-xxxxxx"

4. 구매 주기 예측
   pip install scikit-learn xgboost lightgbm catboost joblib pandas numpy flask flask-cors prophet

## How to Test
1. OCR (영수증 텍스트 추출)
- Colab 환경에서 OCR 코드 실행
- 영수증 이미지를 업로드 후, 이미지 경로 지정
- 추출된 inferText들을 기반으로 필터링 조건에 따라 이름, 수량 추출

2. YOLOv8 (식재료 이미지 인식)
  from ultralytics import YOLO
  model = YOLO("path/to/best.pt")  # 학습된 모델 불러오기
  model.predict(source="path/to/test/images", save=True) # 테스트 이미지 예측

3. OpenAI 기반 GPT4o (메뉴 추천 / 식재료 조언)
: 코드 참고

4. 구매 주기 예측


## Sample Data
2. YOLOv8 (식재료 이미지 인식)
  train/validation/test 이미지가 포함된 커스텀 데이터셋 zip: fooding.v1i.yolov8.zip
