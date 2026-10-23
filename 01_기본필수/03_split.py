"""03 기본필수: 20행을 train 12행, valid 4행, test 4행으로 나눕니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 Python 파일과 같은 폴더에 있는 CSV를 읽습니다. 작업 폴더를 바꾸어도 됩니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "requests.csv")

# test_size=0.2는 전체 20행 중 4행을 따로 보관한다는 뜻입니다.
# B03-1: 전체 20행 × 0.2 = Test 4행. 남는 train_valid는 16행이다.
test_ratio = 0.2
# 표 전체를 함께 나누므로 입력과 정답의 행 대응이 유지된다.
# random_state=42는 분할을 재현하기 위한 값이며 성능을 높이는 설정이 아니다.
# stratify는 각 묶음에 성공/실패 비율이 가능한 한 유지되도록 한다.
train_valid, test = train_test_split(
    data, test_size=test_ratio, random_state=42, stratify=data["failed"])
# 두 번째 test_size는 전체가 아닌 남은 16행에 적용된다: 16 × 0.25 = Valid 4행.
# 따라서 Train=12, Valid=4, Test=4이다. 여기서 test_size라는 인자명이 Test 역할을 뜻하지는 않는다.
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])

# Train은 학습, Valid는 개발 중 점검, Test는 선택 완료 후 최종 평가용이다.
# 이번 수업에서는 Test를 보관만 한다. 실제 최종 Test 평가를 수행한 것은 아니다.
# Train/Test로 나누고 CV를 했다면, 각 fold의 검증 부분이 여기서 말하는 Valid 역할을 한다.
print("train:", len(train))
print("valid:", len(valid))
print("test:", len(test))
print("valid ID:", valid["request_id"].tolist())
print("test는 오늘 학습과 예측에 사용하지 않습니다.")
