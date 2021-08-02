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
import torch
from model.model import BertCRF
model = BertCRF(n_layer_bert=2)
model.load_state_dict(torch.load("/home/tuenguyen/Desktop/24mar2021/task_nlp/join_task_gender_department/checkpoints/vner_model.bin"))
text = 'cho anh nguyễn   văn nam ~  xin thông tin khám bệnh ở khoa cấp cứu cơ sở miền nam'
a=model.predict(text,device='cpu')
print(a)
>>>[('O', 'cho'), ['GENDER', ' anh'], ('O', 'nguyễn_văn'), ('O', 'nam'), ('O', '~'), ('O', 'xin'), ('O', 'thông_tin'), ('O', 'khám'), ('O', 'bệnh'), ('O', 'ở'), ['O', ' khoa cấp_cứu_cơ_sở miền nam']]
```