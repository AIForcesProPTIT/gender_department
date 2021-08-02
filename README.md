# GENDER-DEPARTMENT

## Dữ liệu

Dữ liệu training/testing có tỉ lệ 7992/2224 câu văn.

Số lượng nhãn gender/department là 7000/12000.

## Mô hình 

Dùng 2 layer của phobert + 1 layer LSTM làm backbone.

Top layer là linear + CRF làm head.

## Kết quả 

F1-SCORE: GENDER = 0.9

## Predict 

```python3

from model.model import BertCRF
model = BertCRF(n_layer_bert=2)
text = 'chị gái của nguyễn văn nam xin thông tin khám bệnh ở khoa cấp cứu phía miền nam'
predict, sentence=model.predict(text,device='cpu')
print(list(zip(sentence,predict)))
