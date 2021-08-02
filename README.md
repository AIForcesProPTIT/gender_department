# GENDER-DEPARTMENT

## Dữ liệu

Dữ liệu training/testing có tỉ lệ 7992/2224 câu văn.

Số lượng nhãn gender/department là 7000/12000.

![alt text](https://github.com/AIForcesProPTIT/gender_department/blob/main/plot.png?raw=true)
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
text = 'cho anh nguyễn   văn nam ~  xin thông tin khám bệnh ở khoa cấp cứu, có trự sở ở miền nam'
a=model.predict(text,device='cpu')
print(a)
>>>[('O', 'cho'), ['male', ' anh'], ('O', 'nguyễn_văn'), ('O', 'nam'), ('O', '~'), ('O', 'xin'), ('O', 'thông_tin'), ('O', 'khám'), ('O', 'bệnh'), ('O', 'ở'), ['department', ' khoa cấp_cứu'], ('O', ','), ('O', 'có'), ('O', 'trự'), ('O', 'sở'), ('O', 'ở'), ('O', 'miền'), ('O', 'nam')]


text = 'cho chị đặt lịch hẹn lúc 13h tại khoa xương khớp'
a=model.predict(text,device='cpu')
print(a)
>>>[('O', 'cho'), ['female', ' chị'], ('O', 'đặt'), ('O', 'lịch'), ('O', 'hẹn'), ('O', 'lúc'), ('O', '13'), ('O', 'h'), ('O', 'tại'), ['department', ' khoa xương khớp']]

text = 'ông muốn bỏ lịch lúc 14h, cô muốn thay chồng gặp chị gái, anh trai lúc 16h'
a=model.predict(text,device='cpu')
print(a)
>>>[['male', ' ông'], ('O', 'muốn'), ('O', 'bỏ'), ('O', 'lịch'), ('O', 'lúc'), ('O', '14'), ('O', 'h'), ('O', ','), ['female', ' cô'], ('O', 'muốn'), ('O', 'thay'), ['male', ' chồng'], ('O', 'gặp'), ('female', 'chị_gái'), ('O', ','), ('male', 'anh_trai'), ('O', 'lúc'), ('O', '16'), ('O', 'h')]

```

