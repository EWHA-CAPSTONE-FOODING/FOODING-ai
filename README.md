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
