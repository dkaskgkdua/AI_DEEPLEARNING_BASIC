# Unsupervised Learning
비지도 학습은 입력 데이터에 대한 레이블 없이 모델이 데이터의 패턴을 스스로 찾아내는 방식입니다. 데이터의 구조를 이해하거나 데이터 내에서 유사한 항목을 그룹화하는 데 주로 사용됩니다.
예를 들어, 고객의 구매 데이터를 사용해 특정 행동 패턴을 가진 그룹을 찾아내는 것이 가능합니다.
<img src="./images/unsupervised/unsupervised.jpg" width="300" />

## Supervised Learning
이전 목차까지 배운건 지도학습이다. 지도 학습은 모델이 학습할 때 입력 데이터와 이에 대한 정답(레이블)을 함께 제공하는 방식입니다. 이를 통해 모델은 주어진 입력에 대해 정답을 예측하는 방법을 학습하게 됩니다.
예를 들어, 사진을 입력으로 주고 사진 속 인물이 누구인지에 대한 레이블을 제공하면, 모델은 학습을 통해 새로운 사진에 대해서도 인물을 예측할 수 있게 됩니다.

### Category & Example
- Classification
<img src="./images/unsupervised/classification.jpg" width="500" />
- Regression
<img src="./images/unsupervised/regression.jpg" width="500" />

## Unsupervised Learning
- 레이블 불필요: 정답이 없으므로 데이터 준비에 드는 비용이 적습니다.
- 패턴 탐색: 데이터의 분포나 패턴을 찾는 데 효과적이며, 데이터 군집화나 차원 축소 등에 적합합니다.
- example : Clustering, Compression(용량 줄여야 하는데 방법을 모를 때), Feature & Representation(중요한 데이터 뽑기), Dimensionality reduction, Generative model..

### Principle Component Analysis(PCA - 주성분 분석)
고차원의 데이터를 저차원으로 축소하는 차원 축소(dimensionality reduction) 기법입니다. 
여러 변수 간의 상관관계를 분석해 데이터의 중요한 특징을 유지하면서 데이터를 단순화하는 방식으로 작동합니다. 이를 통해 데이터의 주요 패턴을 더 명확하게 파악할 수 있으며, 노이즈 제거 및 시각화 용이성의 장점이 있습니다.

- 특징
  - 중요한 데이터만 남기면서 효율성 올리고 overfitting 완화
  - 2,3차원으로 시각화가 가능해져 패턴 분석이 쉬워짐
  - 전처리 용도로 사용
- 한계
  - 선형성 가정(비선형 관계를 설명하기 어려움)
  - 차원 축소하면서 정보 손실 가능
  - feature scale에 민감하기 때문에 정규화가 필요함
- 절차
  1. d-dimensional 데이터 정규화
     - 평균은 0, 표준 편차는 1로 변환
     - <img src="./images/unsupervised/standardization.jpg" width="150" />
  2. convariance matrix(공분산 행렬) 계산
     - 공분산은 두 개 이상의 변수 사이의 상관관계를 나타냄
     - x1과 x2 사이의 공분산이 0보다 크면, x1이 커질 때 x2도 같이 커지는 경향. 0보다 작으면 반대 경향.
     - 공분산이 0이면 두 변수 사이에 상관 관계가 없음
     - <img src="./images/unsupervised/공분산매트릭.jpg" width="250" />
  3. Egienvector와 eigenvalue를 구하기 위한 Eigenvalue Decomposition
     - v는 eigenvector, λ는 eigenvalue, 13차원 feature = 13개 eigenvalue
     - <img src="./images/unsupervised/eigenvalue_decomposition.jpg" width="100" />
  4. Eigenvalue 정렬
     - 전체 eigenvalue 합 대비 eigenvalue 값의 비율
  5. Top-k Eigenvector와 eigenvalue 선택
     - 아래 이미지처럼 주성분이 어느정도 비율인지 확인
     - <img src="./images/unsupervised/explained_variance.jpg" width="300" />
  6. 선택된 K개의 eigenvector로 부터 projection matrix W구하기
     - decomposition 수행 후 얻은 eigenvalue 값을 이용하여 정보량이 많은 eigenvector 선택, projection matrix 만듬
  7. W를 이용해 d-dimensional 데이터를 k-dimensional 데이터로 변환
     - projection matrix를 이용하여 feature transform 수행
     - <img src="./images/unsupervised/transformation.jpg" width="100" />
- 예시(13차원 데이터 > 2차원 projection)
  - <img src="./images/unsupervised/PCA예시.jpg" width="300" />

### Linear Discriminant Analysis(LDA)
데이터의 분류(classification) 를 목적으로 하는 지도 학습 기반의 차원 축소 기법입니다. 데이터의 클래스 간 분산을 극대화하고, 클래스 내 분산을 최소화하도록 차원 축소를 수행하여 분류 성능을 높이는 데 주로 사용됩니다. 주성분 분석(PCA)과는 다르게, 데이터의 클래스 정보를 활용해 변환을 수행합니다.

- 특징
  - 차원 축소: 입력 데이터의 차원을 줄여 처리 속도를 높이고 데이터 구조를 단순화함
  - 분류 성능 개선: 클래스 간 구분을 극대화하는 방향으로 데이터를 변환해 분류 성능 향상
- 한계
  - 선형성 가정(비선형 관계를 설명하기 어려움)
  - 정규분포 가정(클래스 간 데이터가 정규 분포를 따른다는 가정이 필요)
- 절차
  1. d-dimensional 데이터 정규화
  2. 각 class별로 d-dimensional mean vector 계산
  3. Between-class scatter matrix, within-class scatter matrix 계산(S_b, S_w)
     - S_w : 각 class별로 평균과 얼마나 멀리 퍼져있는지를 표현
     - <img src="./images/unsupervised/S_w.jpg" width="100" />
     - S_b : class간 상관 관계를 표현
     - <img src="./images/unsupervised/s_b.jpg" width="200" />
  4. Eigenvalue, eigenvector 계산
     - 역행렬 구하고 고유값 decomposition
  5. Top-k Eigenvector, eigenvalue 선택
  6. 선택된 K개의 eigenvector로 부터 projection matrix W구하기
  7. W를 이용해 d-dimensional 데이터를 k-dimensional 데이터로 변환

### Manifold Learning
실제 세계에서는 데이터가 nonlinear하게 분포된 경우가 많아 PCA, LDA는 적합하지 않음. Manifold는 비선형 구조를 찾아내어 차원을 축소하는 기법이다.

## Clustring
대표적인 비지도 학습 기법 중 하나이며, 같은 cluster에 속한 데이터 샘플 간의 유사도는 높이고 다른 cluster에 있는 샘플과의 유사도는 낮추는 방법이다.

### K-means
비슷한 sample끼리 grouping하는 간단한 알고리즘

- 특징
  - 각 군집을 대표 prototype으로 표현(클러스터 중심-centroid을 기준으로 데이터 배정)
  - K가 clustering 개수(최적의 k 결정이 중요)
- 절차
  1. Random으로 K개의 sample을 초기 centroids로서 선택
  2. 모든 sample에 대해 K개의 centroids와 거리를 측정하여, 제일 가까운 centroid의 군집으로 할당
     - 대표적인 측정 방법 : Squared Euclidean distance
     - <img src="./images/unsupervised/squared_euclidean_distance.jpg" width="200" />
     - Sum of squared error(SSE)를 측정해서 응집도 확인
     - <img src="./images/unsupervised/sse.jpg" width="200" />
  3. 같은 cluster로 할당된 sample들의 평균을 구하여 centroid를 갱신
  4. step2, step3의 특정 조건이 만족될 때 까지 반복
- example
  - 음악, 영화 카테고리화
  - 구매자 행동을 기반으로 비슷한 관심사 추천
  - 2차원 feature 기준의 150개 샘플, 3개의 centroid 데이터 예시
    - <img src="./images/unsupervised/clustering.jpg" width="200" />
- 한계
  - K를 사전에 정해야함
  - Overlap 허용 안함, 하나의 sample은 하나의 cluster에 속함
  - 계층적 분류 불가
  - outlier에 민감
  - 초기 centroid에 영향 많이 받음
    - 개선을 위해 여러번 학습시켜 보고 가장 좋은 성능 모델을 선택

### K-means plusplus
K-means의 개선된 버전. centroid를 더 효과적으로 초기화하여 효율적으로 만든다.

- 절차
  1. K개의 centroid를 저장할 빈 matrix M 생성
  2. 첫번째 centroid는 random하게 생성 후, M에 저장
  3. M에 소속되지 않은 남아있는 sample에 대해, M에 있는 centroid들과의 distance 중 최소값을 구함
  4. step3에서 구한 값에 기반하여, 기존 centroid들과의 최소거리가 가장 먼 sample을 다음 centroid로 선택 후 M에 추가
  5. step3, 4를 k개의 centroid가 선택될때까지 진행
  6. 앞서 선택된 K개의 centroid를 이용하여 Classic K-means clustering 수행

### Fuzzy C-Means(FCM)
K-means는 하나의 sample에 하나의 cluster를 할당하는 hard clustering인데 여러개의 cluster에 할당이 필요한 경우가 있다.
이때 사용하는 방법이 Soft Clustering이며 FCM이 있다. 0,1 할당이 아닌 0~1 사이 값으로 추론.
<img src="./images/unsupervised/fcm.jpg" width="300" />

- 절차
  1. K개의 centroid를 random하게 설정하고 각 sample에 class membership 부여
  2. 각 cluster의 centroid 계산
  3. 각 sample의 membership을 update
  4. step2, 3을 특정 조건이 만족 될 때 까지 반복

> Elbow Method
> 클러스터링에서 최적의 클러스터 개수를 선택하는 데 사용되는 기법입니다. K-Means와 같은 클러스터링 알고리즘에서 클러스터 개수
> K를 사전에 지정해야 하는데, 이때 엘보우 기법을 통해 데이터에 적합한 클러스터 개수를 찾을 수 있습니다.
> 1. SSE 계산
> 2. SSE와 클러스터 개수 K의 관계 그래프 생성(특정 시점 이후로 감소 폭이 줄어들음)
> 3. 엘보우 지점 찾기(SSE가 급격히 감소하다가 완만해지는 시점-팔굼치처럼 꺾이는 지점을 찾는다. 이 지점이 최적의 K 값)
> <img src="./images/unsupervised/elbow.jpg" width="200" />
 

### Hierarchical Tree



### Density

## Autoencoder(AE)

### Convolutional AE

### Regularization: Sparse

### Denoising AE

### Stacked AE