### [데이콘] 감귤 착과량 예측 AI 경진대회
### Predicting_amount_of_tangerines
<img src="https://user-images.githubusercontent.com/83712521/209248378-71c3846a-2959-48ab-be05-3f68b2a84280.png" width="500" height="250">


<img src="https://user-images.githubusercontent.com/83712521/209248500-25c1d6e9-c295-45f4-baf3-f001e6960b23.png" width="500" height="300">

#### public : 6등
#### private : 상위 7%

---

### 0. 코드 정리
<table> 
    <thead>
        <tr>
            <th>구    분</th>
            <th>설    명</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> 01 </td>
            <td> 시도 모델, 시도 및 착안 아이디어 </td>
        </tr>
    </tbody>
</table>

### 1. 대회 목적
- 협업 및 의사소통 능력 향상
- 데이터 분석 능력 배양
- 머신러닝 및 딥러닝 회귀 모델 숙달
- 

### 2. 대회 개요
#### (1) 내용
<img src="https://user-images.githubusercontent.com/83712521/209249341-87230027-5a63-44be-92de-eb1db6d81b40.png" width="600" height="400">

#### (2) 평가 산식 : NMAE

```python
import numpy as np
def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
```


### 3. 데이터 및 모델

#### (1) 데이터
#### (2) 모델

### 4. 과정
#### (1) 데이터 핸들링
#### (2) 모델 핸들링

### 5. 결과 및 성과
#### (1) 느낀 점
#### (2) 참고 사진

### 6. 추후 개선 방향
### 7. 기타 참고 자료
 
