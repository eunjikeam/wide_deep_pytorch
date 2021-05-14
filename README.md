# WDL

wide and deep with pytorch 구축
해당 모델은 Binary Classification 용으로만 구축되어 있음

# widendeep folder
## Preprocessor(preprocessor.py)

- BasePreprocessor, WidePreprocessor, DeepPreprocessor
- WidePreprocessor
    - widepart에 들어갈 변수들을 지정해주고, crossed column을 생성해서 label encoding하는 class
    - Wide Part에는 categorical한 변수만 들어가야 함
- DeepPreprocessor
    - deep part에 들어갈 변수들을 지정해주고, scaling & embedding 할 컬럼과 차원 등을 성정해주는 class
- 이 두가지 Preprocessor는 fit한 Preprocessor를 joblib으로 저장해야함 ( test할 데이터에 동일하게 적용하기 위해서)

## Dataset(datareader.py)

- input type : Dictionary(wide, deep_dense, target)
- attribute : ( X_wide, X_deep_dence, Y)

## WDL Architecture(widendeep.py)

- Wide
    - Linear(wide_dim, output_dim)
- Deep
    - Embedding(embedding_cols.nunique(), embed_dim)  + FC
- WideDeep
    - Wide + Deep

## Train(train.py)
- 현재 binary classification 만 가능하게 구축해놓음
- train/eval/pred/retrain 함수 존재
- config 파일(dictionary)를 만들어서 입력하고, 학습 및 예측하게 하기

## Reference code
[widedeep shebburs](https://github.com/shebburs/Wide-and-Deep-PyTorch)  
[widedeep jrzaurin](https://github.com/jrzaurin/pytorch-widedeep)  
[widedeep zenwan](https://github.com/zenwan/Wide-and-Deep-PyTorch)  
