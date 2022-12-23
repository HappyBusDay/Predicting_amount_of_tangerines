### [데이콘] 감귤 착과량 예측 AI 경진대회
### Predicting_amount_of_tangerines
<img src="https://user-images.githubusercontent.com/83712521/209253559-18bc6b5e-57c7-4eb7-9ca0-def609d0626f.png" width="500" height="250">

<img src="https://user-images.githubusercontent.com/83712521/209253528-76191999-6f1f-4665-a236-47b82eab6cb3.png" width="500" height="300">



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
        <tr>
            <td> 02 </td>
            <td> 제출 PPT </td>
        </tr>
    </tbody>
</table>

### 1. 대회 목적

    1. 협업 및 의사소통 능력 향상
    2. 데이터 분석 능력 배양
    3. 머신러닝 및 딥러닝 회귀 모델 숙달

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


### 3. 데이터 및 모델 핸들링

#### (1) 데이터

<img src="https://user-images.githubusercontent.com/83712521/209250837-1b9547f0-4364-4ca2-9b24-d727324b0eae.png" width="700" height="300">

    sourece : 수고, 수관폭, 2022-09-01 ~ 2022-11-30 기간의 새순 및 엽록소
    target : 착과량 (int)

#### (2) 모델


하나. 머신 러닝 (Regressor)

    1. Decision Tree 
    2. Logistic 
    3. RandomForest 
    4. Support Vector  
    5. KNeighbors Regressor
    6. Gaussian NB
    7. Extra Tree
    8. AdaBoost
    9. Gradient Boost
    10. XGBoost (eXtreme Gradient Boosting)
    11. Huber
    12. Theil-Sen 
    13. polynomial
    
둘. 딥러닝

    1. DNN
        * layer : 1, 2, 3
        * dropout : 0.1, 0.3, 0.5, 0.9
        * optimizer : adam, adamW
        * activation : relu, leaky relu
        
셋. 앙상블
    
    1. Voting Regressor
    2. Mean Score

### 4. 시도 및 착안 아이디어

#### (1) Correlation 활용 Feature Select

![image](https://user-images.githubusercontent.com/83712521/209253166-dc28bf6a-1ae4-45c3-838c-2b02d6894592.png)

```python
from category_encoders import OneHotEncoder
from sklearn.feature_selection import f_regression, SelectKBest

encoder = OneHotEncoder(use_cat_names = True)
selector = SelectKBest(score_func=f_regression, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)

selected_names = all_names[selected_mask]
```
![image](https://user-images.githubusercontent.com/83712521/209252535-a2eec236-1d69-4d0f-bb22-7d94ba7bf23d.png)


#### (2) 새로운 column 생성

![image](https://user-images.githubusercontent.com/83712521/209252898-12880366-614b-4dc0-881d-9293c94bc867.png)
 


#### (3) voting regressor

![image](https://user-images.githubusercontent.com/83712521/209252978-cfe2d523-26c4-4479-93b5-f11e9e54bf7c.png)

#### (4) 중간 값 이용 앙상블

![image](https://user-images.githubusercontent.com/83712521/209253032-e1b76b8e-8b4d-4594-b03f-a8e619e5440c.png)


#### (5) grid search

![image](https://user-images.githubusercontent.com/83712521/209253115-9402cd20-eb69-4e32-ace9-29f3cd016ba8.png)

```python
estimator = RandomForestRegressor()
param_grid = {'criterion':['mae','mse’], 'max_depth':[5]} 
grid = GridSearchCV(estimator, param_grid=param_grid) 
grid.fit(X_train, y_train)

estimator = RandomForestRegressor()
estimator.set_params(**grid.best_params_)
estimator.fit(X_train, y_train)
```


#### (6) optuna

![image](https://user-images.githubusercontent.com/83712521/209253213-dc1f925e-6b27-410c-bf1c-b77c14308347.png)


#### (7) 반올림, 내림 (데이터 후처리)

![image](https://user-images.githubusercontent.com/83712521/209253276-27b0bded-0db4-436e-8d23-0d04ddb5331a.png)

```python
sample_submission['착과량(int)'].map(lambda x: round(x)) 
```

### 5. 기타 참고 자료

    제출 PPT 
 
