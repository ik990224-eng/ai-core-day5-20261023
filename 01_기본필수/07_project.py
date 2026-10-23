"""07~08 종합: CSV 읽기 -> 분할 -> X/y 선택 -> 전처리와 학습 -> 검증 예측 -> 저장."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

# 제공: 이 Python 파일과 같은 폴더에 있는 CSV를 읽습니다. 작업 폴더를 바꾸어도 됩니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "requests.csv")
# 제공: 먼저 20%를 test로 보관하고, 남은 80% 중 25%를 valid로 나눕니다.
# random_state는 재현용 고정값입니다. stratify는 성공/실패 비율을 고려합니다.
# 전체 20행에서 Test 4행을 먼저 보관하고, 남은 16행에서 Valid 4행을 나눈다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])
# test는 개수 확인 외에 사용하지 않습니다.

# 제공: 앞에서 배운 입력 선택과 전처리 구조를 그대로 연결합니다.
features = ["input_chars", "task_type"]
X_train = train[features]
y_train = train["failed"]
X_valid = valid[features]
# 검증 정답은 아래 채점 단계에만 사용한다. predict에는 X_valid만 전달한다.
# 숫자 경로와 문자 경로를 각각 처리해 [표준화 글자 수, qa 여부, summary 여부]로 합친다.
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])
# DummyClassifier는 y_train의 최빈값만 기억하고 모든 요청에 같은 답을 내는 기준선이다.
# 이번에는 성공(0)이 더 많아 모두 0을 예측한다. 입력과 실패 사이의 관계를 학습하지 않는다.
# 따라서 숫자 표준화가 이 Dummy의 정확도를 높이는 것은 아니다. 여기서는 처리 흐름을 배운다.
# ColumnTransformer가 열별 처리를 합친다면, Pipeline은 전처리 -> 모델을 순서대로 연결한다.
pipeline = Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])
# B07-1: 내부 흐름은 전처리 fit/transform(X_train) -> 모델 fit(변환된 X_train, y_train)이다.
# 전처리 기준과 모델 상태 모두 Train에서만 배운다.
pipeline.fit(X_train, y_train)
# B07-2: 내부 흐름은 전처리 transform(X_valid) -> 모델 predict이다. Valid로 다시 fit하지 않는다.
prediction = pipeline.predict(X_valid)

# 제공: 예측을 원래 ID·정답과 나란히 놓고, 이 파일 옆에 저장합니다.
result = valid[["request_id", "failed"]].copy()
result["predicted"] = prediction
result.to_csv(HERE / "valid_predictions.csv", index=False, encoding="utf-8-sig")
accuracy = (result["failed"] == result["predicted"]).mean()
# 비교 결과 True/False의 평균은 맞힌 비율이다: True=1, False=0으로 계산되어 3/4=0.75이다.
print("분할:", len(train), len(valid), len(test))
print(result.to_string(index=False))
print("valid 정확도:", round(accuracy, 2))
print("저장: valid_predictions.csv / test는 사용하지 않았습니다.")

# 결과 설명: 실제 Valid는 성공 3건, 실패 1건인데 예측은 모두 성공(0)이다.
# 따라서 정확도는 75%지만 실패 1건 중 발견한 실패는 0건이다. 실패 탐지에 유용하다고 할 수 없다.
# Train=학습, Valid=개발 중 점검, Test=선택 후 최종 평가용이며 이번 실습에서는 Test를 보관한다.
# CSV의 request_id는 같은 요청의 실제값/예측값을 연결해 확인하기 위한 식별자이다.
