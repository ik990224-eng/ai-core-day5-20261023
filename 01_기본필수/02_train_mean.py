"""02 기본필수: 평균을 어디에서 계산해야 하는지 비교합니다."""
train_values = [10, 20]
valid_values = [100]

# 진짜 학습 기준으로 사용할 수 있는 것은 train의 평균입니다.
# B02-1: sum은 합(30), len은 개수(2)이므로 Train 평균은 15이다.
# 전처리도 자료에서 기준을 배우는 과정이다. 검증 자료는 기준 계산에서 제외한다.
train_mean = sum(train_values) / len(train_values)
# 아래 전체 평균은 '잘못된 기준과의 비교'를 위한 시연일 뿐입니다.
all_values = train_values + valid_values
# 리스트의 +는 [10, 20, 100]으로 이어 붙인다. 전체 평균은 약 43.33이다.
# Valid의 100을 섞으면 Train을 표현하는 기준까지 달라진다. 정답 y 없이도 누수가 가능하다.
# 이 파일은 두 평균을 비교하는 예제이며 실제 데이터 변환이나 모델 학습은 하지 않는다.
all_mean = sum(all_values) / len(all_values)

print("train 평균:", round(train_mean, 2))
print("전체 평균(비교만):", round(all_mean, 2))
print("실제 전처리에는 train 평균만 사용합니다.")
