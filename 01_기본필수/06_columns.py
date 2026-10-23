"""06 기본필수: 숫자 1열과 문자 1열을 함께 변환합니다."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

train = pd.DataFrame({"input_chars": [100, 200, 300],
                      "task_type": ["qa", "summary", "qa"]})
valid = pd.DataFrame({"input_chars": [400, 500],
                      "task_type": ["summary", "qa"]})
# ColumnTransformer는 열마다 다른 도구를 적용한 뒤 결과를 옆으로 합친다.
# 각 튜플은 (경로 이름, 변환 도구, 대상 열 목록)이다. num/cat은 경로를 구별하는 이름이다.
# input_chars -> 표준화 1열, task_type -> qa/summary 2열, 합계 3열이 된다.
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])
# fit 한 번으로 숫자 경로는 평균/표준편차, 문자 경로는 범주 목록을 Train에서 배운다.
preprocessor.fit(train)
# B06-1: 두 경로 모두 이미 배운 기준으로 Valid를 바꾼 뒤 결합한다.
# 첫 행 [400, summary] -> [약 2.449, 0, 1]. 행 수는 유지되고 열 수만 2 -> 3으로 변한다.
valid_ready = preprocessor.transform(valid)

print("변환 전:", valid.shape)
print("변환 후:", valid_ready.shape)
print(valid_ready.round(3))
