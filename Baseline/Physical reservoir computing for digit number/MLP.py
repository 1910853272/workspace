import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from IPython.display import display
import matplotlib.pyplot as plt

# 加载数据
train_data = pd.read_csv("./train_data_01.csv")
test_data = pd.read_csv("./test_data_01.csv")

# 打乱数据
train_data = shuffle(train_data)
test_data = shuffle(test_data)

# 提取训练集和测试集的特征和标签
X_train, y_train = train_data.iloc[:,1:6], train_data.loc[:,"LETTER"]
X_test, y_test = test_data.iloc[:,1:6], test_data.loc[:,"LETTER"]

# 创建模型
model = tf.keras.models.Sequential([
    # 添加全连接层，包含5个神经元，激活函数为softmax
    tf.keras.layers.Dense(5, activation='softmax', input_shape=(5,))
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型并记录训练过程
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=16,
                    validation_data=(X_test, y_test))

# 可视化训练过程
# 绘制训练和验证的损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证的准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()

# 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# 对测试集进行预测
predictions = model.predict(X_test)

# 可视化预测结果与实际标签
for i in range(10):  # 显示前10个预测结果
    print(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test.iloc[i]}")
