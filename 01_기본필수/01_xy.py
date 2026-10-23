"""01 기본필수: CSV에서 입력과 정답을 각각 꺼냅니다."""
from pathlib import Path
import pandas as pd

# 제공: 이 Python 파일과 같은 폴더에 있는 CSV를 읽습니다. 작업 폴더를 바꾸어도 됩니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "requests.csv")

# X는 예측 전에 알 수 있는 정보입니다. ID와 정답은 입력에 넣지 않습니다.
# B01-1: 예측 시점은 요청 실행 전이다. 그때 아는 글자 수와 작업 종류를 X로 고른다.
# 이중 대괄호는 열 이름 목록으로 표(DataFrame)를 선택한다. 20행 × 2열이다.
# request_id는 식별용이고, failed는 예측할 정답이므로 입력에서 제외한다.
X = data[["input_chars", "task_type"]]
# B01-2: 같은 요청 행의 실제 결과를 y로 선택한다. 0=성공, 1=실패이다.
# 지도학습에서는 y를 학습에 사용하지만, 새 요청을 predict할 때는 X만 전달한다.
# failed를 X에 넣으면 실행 전에 모르는 답을 미리 보여 주는 데이터 누수가 된다.
y = data["failed"]

# shape는 크기만 알려 준다. (20,)이어도 정답 열을 잘못 골랐을 수 있으므로 값도 확인한다.
print("입력 열:", X.columns.tolist())
print("X 모양:", X.shape)
print("y 모양:", y.shape)
print("처음 3개 정답:", y.head(3).tolist())
