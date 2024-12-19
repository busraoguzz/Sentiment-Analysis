#!/usr/bin/env python
# coding: utf-8

# In[486]:


import numpy as np
import pandas as pd
import tensorflow as tf
import tkinter as tk
import matplotlib.pyplot as plt


# In[487]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tkinter import messagebox
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[488]:


dataset = pd.read_csv("C:/Users/busra/OneDrive/Masaüstü/kullanıcıyorumları.csv")


# In[489]:


import os
print(os.getcwd()) 


# In[490]:


dataset


# In[491]:


target = dataset['Rating'].values.tolist()
data = dataset['Review'].values.tolist()


# In[492]:


cutoff = int(len(data) * 0.80)
x_train, x_test = data[:cutoff], data[cutoff:]
y_train, y_test = target[:cutoff], target[cutoff:]


# In[493]:


x_train[500]


# In[494]:


x_train[800]


# In[495]:


y_train[800]


# In[496]:


num_words = 10000
tokenizer = Tokenizer(num_words=num_words)


# In[497]:


tokenizer.fit_on_texts(data)


# In[498]:


tokenizer.word_index


# In[499]:


x_train_tokens = tokenizer.texts_to_sequences(x_train)


# In[500]:


x_train[800]


# In[501]:


print(x_train_tokens[800])


# In[502]:


x_test_tokens = tokenizer.texts_to_sequences(x_test)


# In[503]:


num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)


# In[504]:


print(num_tokens)


# In[505]:


x_train[2]


# In[506]:


np.mean(num_tokens)


# In[507]:


np.max(num_tokens)


# In[508]:


np.argmax(num_tokens)


# In[509]:


x_train[21941]


# In[510]:


max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens


# In[511]:


np.sum(num_tokens < max_tokens) / len(num_tokens)


# In[512]:


x_train_pad = pad_sequences(x_train_tokens, maxlen = max_tokens)


# In[513]:


x_test_pad = pad_sequences(x_test_tokens, maxlen = max_tokens)


# In[514]:


y_train = np.array(y_train)


# In[515]:


y_test = np.array(y_test)


# In[516]:


x_train_pad.shape


# In[517]:


x_test_pad.shape


# In[518]:


np.array(x_train_tokens[800])


# In[519]:


x_train_pad[800]


# In[520]:


idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))


# In[521]:


def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token!=0]
    text = ' '.join(words)
    return text


# In[522]:


x_train[800]


# In[523]:


tokens_to_string(x_train_tokens[800])


# In[524]:


model = Sequential()


# In[525]:


embedding_size = 50


# In[526]:


model.add(Embedding(input_dim=num_words,
                   output_dim=embedding_size,
                   input_length=max_tokens,
                   name='embedding_layer'))


# In[527]:


model.add(GRU(units=32, return_sequences=True))
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))


# In[528]:


model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[529]:


model.summary()


# In[530]:


history = model.fit(x_train_pad, y_train, epochs=8, batch_size=128, validation_data=(x_test_pad, y_test))


# In[531]:


import seaborn as sns
from tensorflow.keras.callbacks import History

if isinstance(history, dict):
    history_obj = History()
    history_obj.history = history
    history = history_obj
history_data = history.history


# In[532]:


print(type(history))


# In[533]:


print(history_data.keys()) 


# In[534]:


result = model.evaluate(x_test_pad, y_test)


# In[535]:


result[1]


# In[536]:


y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]


# In[537]:


cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])


# In[538]:


cls_true = np.array(y_test[0:1000])


# In[539]:


incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]


# In[540]:


len(incorrect)


# In[541]:


idx = incorrect[0]
idx


# In[542]:


text = x_test[idx]
text


# In[543]:


y_pred[idx]


# In[544]:


cls_true[idx]


# In[545]:


text1 = "bu ürün çok iyi herkese tavsiye ederim"
text2 = "kargo çok hızlı aynı gün elime geçti"
text3 = "büyük bir hayal kırıklığı yaşadım bu ürün bu markaya yakışmamış"
text4 = "mükemmel"
text5 = "tasarımı harika ancak kargo çok geç geldi ve ürün açılmıştı tavsiye etmem"
text6 = "hiç resimde gösterildiği gibi değil"
text7 = "kötü yorumlar gözümü korkutmuştu ancak hiçbir sorun yaşamadım teşekkürler"
text8 = "hiç bu kadar kötü bir satıcıya denk gelmemiştim ürünü geri iade ediyorum"
text9 = "tam bir fiyat performans ürünü"
text10 = "beklediğim gibi çıkmadı"
texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10]


# In[546]:


tokens = tokenizer.texts_to_sequences(texts)


# In[547]:


tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
tokens_pad.shape


# In[548]:


model.predict(tokens_pad)


# In[549]:


def predict_sentiment():
    
    user_text = text_input.get("1.0", "end-1c")
    if not user_text.strip():
        messagebox.showwarning("Uyarı", "Lütfen bir metin giriniz!")
        return

    
    tokens = tokenizer.texts_to_sequences([user_text])
    tokens_pad = pad_sequences(tokens, maxlen=max_tokens)

    
    prediction = model.predict(tokens_pad)
    if prediction[0][0] > 0.5:
        sentiment = "Pozitif ✔"
        sentiment_color = "#28a745"  
    else:
        sentiment = "Negatif ✘"
        sentiment_color = "#dc3545"  

    
    result_label.config(text=sentiment, fg=sentiment_color)


# In[550]:


def create_interface():
   
    root = tk.Tk()
    root.title("Duygu Analizi")
    root.geometry("500x400")
    root.configure(bg="#f8f9fa") 

    
    tk.Label(root, text="Duygu Analizi Uygulaması", font=("Helvetica", 16, "bold"), bg="#f8f9fa", fg="#343a40").pack(pady=10)

    
    tk.Label(root, text="Metni Giriniz:", font=("Helvetica", 12), bg="#f8f9fa", fg="#343a40").pack(pady=5)
    global text_input
    text_input = tk.Text(root, height=5, width=50, font=("Helvetica", 11), bd=2, relief="solid")
    text_input.pack(pady=5)

    
    predict_button = tk.Button(root, text="Tahmin Et", font=("Helvetica", 12, "bold"), bg="#007bff", fg="white",
                               activebackground="#0056b3", activeforeground="white", bd=0, relief="flat", padx=10, pady=5,
                               command=predict_sentiment)
    predict_button.pack(pady=10)

    
    global result_label
    result_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#f8f9fa")
    result_label.pack(pady=20)

    
    root.mainloop()


create_interface()


# In[551]:


plt.figure(figsize=(8, 6))
plt.scatter(cls_true[:1000], y_pred[:1000], alpha=0.5, c='blue', label='Predictions')
plt.plot([0, 1], [0, 1], 'r--', label='İdeal')
plt.title("Scatterplot: True vs. Prediction Values")
plt.xlabel("True Values")
plt.ylabel("Prediction Values")
plt.legend()
plt.show()


# In[552]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)), 
    Dense(1, activation='sigmoid')  
])


model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])


import numpy as np
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))

x_val = np.random.rand(20, 10)
y_val = np.random.randint(0, 2, size=(20,))


history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)


plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




# In[553]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])


x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))
x_val = np.random.rand(20, 10)
y_val = np.random.randint(0, 2, size=(20,))


history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


# In[554]:


print(history.history.keys())


# In[555]:


print(f"Model Accuracy: {result[1] * 100:.2f}%")


# In[556]:


conf_matrix = confusion_matrix(cls_true, cls_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Negatif", "Pozitif"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




