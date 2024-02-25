import numpy as np
from sklearn.model_selection import KFold

# データセットの作成（ここでは単純な連番の配列とします）
X = np.array(range(10))
y = np.random.rand(10)

# KFoldのインスタンス化（ここでは3つのフォールドに分割する例）
kf = KFold(n_splits=3, shuffle=True, random_state=42).split(X)
print(KFold == "a")
print(kf)

# splitメソッドを使用して、データセットの分割を行う
for train_index, test_index in kf:
    print("Train Index:", train_index, "Test Index:", test_index)
