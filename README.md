Duygu Analizi Projesi

Duygu analizi, metinlerdeki duygu durumlarını pozitif ve negatif olarak belirlemek amacıyla kullanılan bir metin madenciliği ve doğal dil işleme (NLP) uygulamasıdır. Bu proje, e-ticaret sitelerindeki kullanıcı yorumlarını sınıflandırmak için geliştirilmiştir ve GRU (Gated Recurrent Unit) tabanlı bir derin öğrenme modeli kullanılmıştır.

Proje Detayları

1. Doğal Dil İşleme (NLP):
   - Kullanıcı yorumları kelimelere ayrılarak sayısal formatta temsil edilmiştir.
   - Metinler sabit bir uzunluğa padding işlemiyle getirildi.

2. GRU Tabanlı Model:
   - GRU, ardışık verilerdeki bağımlılıkları etkili bir şekilde işlemek için kullanılır.
   - Model mimarisi 4 GRU katmanı ve bir sigmoid aktivasyon fonksiyonuna sahip çıkış katmanından oluşmaktadır.

3. Performans:
   - Test setindeki doğruluk: %95.12.

4. Kullanıcı Arayüzü:
   - Tkinter tabanlı bir grafiksel arayüz, kullanıcılara metinlerini girerek tahmin alabilme olanağı sağlamaktadır.

Gerçek vs. tahmin edilen değerler için scatter plot.

![image](https://github.com/user-attachments/assets/1c8e68a1-53c2-46fd-b645-633e8ab2b27b)



Performansı değerlendirmek için karışıklık matrisi.

![image](https://github.com/user-attachments/assets/b21e5780-8966-4886-abeb-3c17257eae2c)



