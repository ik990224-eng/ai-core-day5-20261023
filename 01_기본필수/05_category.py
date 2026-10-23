"""05 기본필수: qa와 summary를 서로 다른 0/1 열로 표현합니다."""
from sklearn.preprocessing import OneHotEncoder

train = [["qa"], ["summary"], ["qa"]]
valid = [["summary"], ["qa"]]
# sparse_output=False는 결과를 바로 볼 수 있는 숫자 표로 받는 설정입니다.
encoder = OneHotEncoder(sparse_output=False)
# fit은 평균이 아니라 종류 목록과 열 배치를 배운다. 이 예제의 열 순서는 [qa, summary]이다.
encoder.fit(train)

# B05-1: Train에서 배운 열 배치를 그대로 적용한다.
# summary -> [0, 1], qa -> [1, 0]. 1은 해당 종류라는 뜻이지 실패(y=1)라는 뜻이 아니다.
# 종류에 순위를 붙이지 않고 별도 열로 표시한다. 입력 문자 1열이 숫자 2열로 바뀐다.
# tolist()는 변환된 배열을 Python 리스트로 표현할 뿐, 새로 학습하지 않는다.
valid_ready = encoder.transform(valid).tolist()
# Valid로 다시 fit하면 종류와 열 구성이 바뀔 수 있으므로 transform만 사용한다.
# 이 기본 실습의 Valid에는 알려진 종류만 있다. 새 종류 처리는 응용 실습에서 다룬다.
print("열 순서:", encoder.categories_[0].tolist())
print("valid 변환:", valid_ready)
