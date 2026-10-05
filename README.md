# Hava Kalitesi Tahmini

## Servisi çalıştırma

Proje klasöründe bağımlılıkları yükleyin ve veri hazırlama/eğitim adımlarını çalıştırın:

```bash
python -m pip install -r requirements.txt
python preprocessing.py
python train.py
```

Eğitim `lstm_model.pth` dosyasını oluşturduktan sonra API'yi başlatın:

```bash
uvicorn api:app --reload
```

API dokümantasyonu `http://127.0.0.1:8000/docs` adresindedir. `GET /health`
servis ve model dosyalarının durumunu, `POST /predict` ise sonraki saatin
`CO(GT)` tahminini döndürür. Model henüz eğitilmediyse tahmin isteği `503`
yanıtı verir.

## Tahmin isteği

`observations`, kronolojik sıradaki son 72 saatlik gözlemleri içerir. Her satırın
özellik sırası aşağıdaki gibi olmalıdır; değerler ölçeklenmemiş/orijinal ölçüm
değerleridir:

1. `CO(GT)`
2. `PT08.S1(CO)`
3. `NMHC(GT)`
4. `C6H6(GT)`
5. `PT08.S2(NMHC)`
6. `NOx(GT)`
7. `PT08.S3(NOx)`
8. `NO2(GT)`
9. `PT08.S4(NO2)`
10. `PT08.S5(O3)`
11. `T`
12. `RH`
13. `AH`
14. `hour`
15. `month`

```json
{
  "observations": [
    [2.6, 1360, 150, 11.9, 1046, 166, 1056, 113, 1692, 1268, 13.6, 48.9, 0.76, 0, 3]
  ]
}
```

Örnek yalnızca gövde biçimini gösterir; gerçek istekte tam olarak 72 gözlem
satırı gönderilmelidir. Başarılı yanıt `{"target":"CO(GT)","prediction":...}`
biçimindedir.
