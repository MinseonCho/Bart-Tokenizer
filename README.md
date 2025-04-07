# 🧠 BART Tokenizer & Summarization on Android (Kotlin)

이 프로젝트는 Hugging Face의 BART Tokenizer를 **Kotlin으로 직접 구현**하고,  
ONNX 포맷의 `DistilBART` 모델을 사용해 **텍스트 요약 기능을 Android 앱에서 수행**하는 데모입니다.

## 📌 주요 내용

- 🧩 `BartTokenizer`를 Kotlin으로 재구현 (Byte-level BPE 방식)
- 🧠 ONNX 모델(`encoder_model_q4.onnx`, `decoder_model_q4.onnx`)을 Android에서 실행
- 📱 Compose UI로 입력 → 요약 결과 다이얼로그로 표시
- 📝 요약 로직 직접 구현 (Greedy decoding 기반)

## 💡 사용 모델

- Tokenizer 기준: `facebook/bart-large`
- ONNX 모델: [`onnx-community/distilbart-cnn-12-6-ONNX`](https://huggingface.co/onnx-community/distilbart-cnn-12-6-ONNX)
  - `encoder_model_q4.onnx`
  - `decoder_model_q4.onnx`

## 🎓 학습 내용

- Tokenizer 내부 동작 이해 (Byte-Encoding → BPE → Vocab 매핑 → Special Token 처리)
- Transformer 모델의 Encoder/Decoder 구조
- ONNX Runtime을 통한 모델 추론 처리
- Android On-Device 요약 시도

## 📷 스크린샷
> 입력 문장 → 요약 결과
![미디어](https://github.com/user-attachments/assets/36511c0b-c35b-40ca-a37e-4c767cb757df)
