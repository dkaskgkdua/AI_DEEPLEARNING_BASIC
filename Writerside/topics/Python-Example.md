# Python Example

## Multi Layer Perceptron(MLP)
```Python
class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier
    
    Parameters
    ---------------
    n_hidden : int (default: 30)
        히든 레이어 1개의 노드 수
    l2 : float (default: 0.)
        Lambda value for L2-regularization
        No regularization if L2=0.
    epochs : int (default: 100)
        Number of passes over the raining set.
    eta : float (default: 0.001)
        Learning rate
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training examples per minibatch
    seed : int (default: None)
        Random seed for initailizing weights and shuffling.
        
    Attributes
    --------------
    eval_ : dict
        Dictionary collecting the cost, training accuracy.
        and validation accuracy for each epoch during training.
    
    """
    
    # 생성자 함수
    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001
                , shuffle=True, minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        
    # Forward 함수
    def _forward(self, X):
        """ Compute forward propagation step """
        
        # hidden layer의 net input을 구한다 : 입력값과 weight를 dot product하고 bias 더함
        z_h = np.dot(X, self.w_h) + self.b_h
        
        # activation func 적용
        a_h = self._sigmoid(z_h)
        
        # output layer의 net input을 구한다.
        z_out = np.dot(a_h, self.w_out) + self.b_out
        
        # activation func 적용
        a_out = self._sigmoid(z_out)
        
        return z_h, a_h, z_out, a_out
        
    # Loss 계산
    def _compute_cost(self, y_enc, ouput):
        """
        Parameters
        -------------
        y_enc : array, shape = (n_examples, n_labels)
            실제 레이블(one-hot 인코딩 형태)
        output : array, shape = [n_examples, n_output_units]
            예측 확률 분포 - Activation of the output layer (forward propagation)
        
        Returns
        ------------
        cost : float
            Regularized cost
        """
        # 아래부터 cross entropy loss에 L2 loss가 융합된 형태로 계산
        
        # L2 정규화 : self.l2는 정규화 계수(정규화 강도 조절 하이퍼 파라미터)
        # 정규화 계수 * 가중치 제곱의 합
        L2_term = (self.l2 * (
                    np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)
                   ))
        
        # 식 1 > 라벨값 * log(예측값)
        term1 = -y_enc * (np.log(output))
        
        # 식 2 > (1 - 라벨값) * log(1 - 예측값)
        term2 = (1. - y_enc) * np.log(1. - output)
        
        # 원래 손실함수 L + L2 정규화 값
        cost = np.sum(term1 - term2) + L2_term
        
        return cost
    
    def predict(self, X):
        """
        Parameters
        ------------
        X : array, shape = [n_examples, n_features]
            Input layer with original features
            
        Returns
        -----------
        y_pred : array, shape = [n_examples]
            예측값 결과
        """
        z_h, a_h, z_out, a_out = self._forward(X)
        
        # net input의 첫 행의 최대값을 예측값으로 활용
        y_pred = np.argmax(z_out, axis=1)
    
    # 학습용 함수    
    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        weight 초기화, 데이터 순회, backpropagation, evaluation
        Parameters
        -------------
        X_train : array, shape = [n_examples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_examples]
            Target class labels.
        X_valid : array, shape = [n_examples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_example]
            Sample labels for validation during training
        
        Return
        --------
        self
        """
        
        # output, feature 수 계산
        n_output = np.unique(y_train).shape[ 0]
        n_features = X_train.shape[ 1]
        
        # hidden layer 학습 weight, bias 랜덤 초기화
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                   size=(n_features, self.n_hidden))
                                   
        # output layer 학습 weight, bias 랜덤 초기화
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                    size=(self.n_hidden, n_output))
        
        # 진행율
        epoch_strlen = len(str(self.epochs))
        
        # 비용, 정확도 평가
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        
        y_train_enc = self._onehot(y_train, n_output)
        
        # 학습 시작
        for i in range(self.epochs):
            indices = np.arange(X_train.shape[ 0])
            
            if self.shuffle:
                self.random.shuffle(indices)
            
            for start_idx in range(0, indices.shape[0] 
                    - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx 
                                    + self.minibatch_size]
                                    
                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])
                
                # ===================
                # Back propagation
                # ===================
                # weight, bias에 대한 gradient를 계산하고, update값을 구하는 부분 포함
                
                # [n_examples, n_classlabels]
                delta_out = a_out - y_train_enc[batch_idx]
                
                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)
                
                # [n_examples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_examples, n_hidden]
                delta_h = (np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h)
                
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)
                
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)
                
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h
                
                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b.out
                
            # Evaluation
            
            # Evaluation after each epoch during tarining
            z_h, a_h, z_out, a_out = self._forward(X_train)
            
            cost = self._coupute_cost(y_enc=y_train_enc, output=a_out)
            
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) / X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) / X_valid.shape[0])
            
            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()
            
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
            
        return self
                                   
```

### One Hot
```Python
def _onehot(self, y, n_classes):
    one hot = np.zeros((n_classes, y.shape[ 0]))
    for idx, val in enumerate(y.astype(int)):
        onehot[val, idx] = 1.
    return onehot.T
```

### Sigmoid
```Python
def _sigmoid(self, z):
    return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
```

