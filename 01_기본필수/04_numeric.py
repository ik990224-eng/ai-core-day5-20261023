"""04 기본필수: 결측이 없는 숫자를 train 기준으로 표준화합니다."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 제공: 정상 숫자만 사용합니다. 결측 대체는 선택 단계에서 배웁니다.
train = pd.DataFrame({"input_chars": [100, 200, 300]})
valid = pd.DataFrame({"input_chars": [400, 500]})
scaler = StandardScaler()
# fit은 Train 평균 200과 표준편차 약 81.65를 계산해 scaler 안에 기억한다.
# fit 자체가 원래 train 표를 변환된 값으로 덮어쓰지는 않는다.
scaler.fit(train)

# transform은 이미 배운 평균과 크기를 사용합니다. 다시 fit하지 않습니다.
# B04-1: transform은 (원래 값 - Train 평균) / Train 표준편차를 적용한다.
# 400 -> 약 2.449, 500 -> 약 3.674. 반환한 새 배열을 valid_ready에 담는다.
# 여기서 fit(valid)를 하면 평균이 450으로 바뀌어 Train과 서로 다른 기준이 된다.
valid_ready = scaler.transform(valid)
print("train에서 배운 평균:", scaler.mean_.tolist())
print("valid 변환 결과:")
print(valid_ready.round(3))
