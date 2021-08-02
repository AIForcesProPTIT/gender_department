from json import encoder
import re
import json
import unicodedata
from datetime import date
import datetime
from typing import Text
import json

class GenderMatcher(object):
    def __init__(self, dict_offset=None) -> None:
        super().__init__()
        if dict_offset is not None:
            with open(dict_offset,"r") as f:
                self.dict_matcher  = json.load(f)
        self.template = {
            'male':['con trai','nam','nam giới','đàn ông','chú','ông','bố','bô'\
                    'cậu','anh','chồng','anh rể','dượng','ba','bác trai','tía'],
            'female':['con gái','trai','nữ','nữ giới','phụ nữ','cô','bà','mẹ','gì','thím','chị','vợ',\
                     'má','chị dâu','bác gái'],
        }

    def iter_pattern(self):
        for k in self.template:
            for i in self.template[k]:
                yield (unicodedata.normalize("NFC",i),k)

    def norm_sentence(self, sentence: Text)->Text:
        sentence_copy = sentence.replace(","," , ")
        for char in '.,;!?':
            
            sentence_copy = sentence_copy.replace(char," " + char + " ")
        sentence_copy = sentence_copy.replace("  "," ")
        return sentence_copy
        # return unicodedata.normalize('NFC',sentence_copy)# sentence_copy.normalize('NFC')
    def nms(self, entities):
        matches = entities['entities']
        matches_nsm=sorted(matches,key=lambda x:(x['start'],-x['end']))
        # matches_nsm  =[matches[0]]
        selects = []
        while len(matches_nsm):
            top = matches_nsm[0]
            matches_nsm.pop(0)
            selects.append(top)
            v=[]
            for i in matches_nsm:
                st = i['start']
                end = i['end']
                st_top = top['start']
                end_top = top['end']
                if (st >= st_top and end <= end_top):
                    continue
                else:
                    v.append(i)
            matches_nsm=v
        entities['entities'] = selects
        return entities

    def match(self, sentence:Text)->Text:
        sentence_copy = sentence
        # sentence_copy = sentence_copy.replace("_"," ")
        entities = {
            'entities':[]
        }
        for pattern,value in self.iter_pattern():
            matches = [match.span() for match in re.finditer(pattern, sentence_copy) if match.group() == pattern]
            # print(matches)
            if len(matches):
                entities['entities'].extend(
                    [
                        {
                            'start':i[0],
                            'end':i[1],
                            'value':value,
                            'extractor':'template_matching_gender',
                            'confidence':1.0
                        }
                        for i in matches
                    ]
                )

        return self.nms(entities)

class DepartmentMatcher(object):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__()

        self.template={
            'department':[
                r'\bhồi sức.{0,1}cấp cứu\b',
                r'\bhồi sức\b',
                r'\bcấp cứu\b',
                r'\bkhám sức khỏe tổng quát\b',
                r'\btổng quát\b',
                r'\btim mạch\b',
                r'\btim\b',
                r'\bung bướu\b',
                r'\bxạ trị\b',
                r'\bung bưới.{0,1}xạ trị\b',
                r'\bsản khoa\b',
                r'\bkhoa sản\b',
                r'\bphụ khoa\b',
                r'\bnam khoa\b',
                r'\bhỗ trợ sinh sản\b',
                r'\bsinh sản\b',
                r'\bnhi\b',
                r'\bchẩn đoán hình ảnh\b',
                r'\bxét nghiệm\b',
                r'\bdược\b',
                r'\bhồi sức, cấp cứu\b',
                r'\btrung tâm ung bướu\b'
                r'\bnhi\b',
                r'\bsơ sinh\b',
                r'\bnhi .{0,1} sơ sinh',
                r'\bsản phụ\b',
                r'\bxét nghiệm\b',
                r'\bkhám bệnh và nội khoa',
                r'\bđơn nguyên khám bệnh',
                r'\bđơn nguyên khám sức khỏe tổng quát\b',
                r'\bđơn nguyên nội tiêu hóa\b',
                r'\bgan mật\b',
                r'\bnội soi\b',
                r'\bngoại tổng hợp\b',
                r'\bkhoa gây mê hồi sức\b',
            ]
        }
    def iter_pattern(self):
        for k in self.template:
            for i in self.template[k]:
                yield (unicodedata.normalize("NFC",i),k)

    def nms(self, entities):
        matches = entities['entities']
        matches_nsm=sorted(matches,key=lambda x:(x['start'],-x['end']))
        # matches_nsm  =[matches[0]]
        selects = []
        while len(matches_nsm):
            top = matches_nsm[0]
            matches_nsm.pop(0)
            selects.append(top)
            v=[]
            for i in matches_nsm:
                st = i['start']
                end = i['end']
                st_top = top['start']
                end_top = top['end']
                if (st >= st_top and end <= end_top):
                    continue
                else:
                    v.append(i)
            matches_nsm=v
        entities['entities'] = selects
        return entities

    def match(self, sentence:Text)->Text:
        sentence_copy = sentence
        # sentence_copy = sentence_copy.replace("_"," ")
        entities = {
            'entities':[]
        }
        for pattern,value in self.iter_pattern():
            matches = [match.span() for match in re.finditer(pattern, sentence_copy)]
            # print(matches)
            if len(matches):
                entities['entities'].extend(
                    [
                        {
                            'start':i[0],
                            'end':i[1],
                            'value':value,
                            'extractor':'template_matching_department',
                            'confidence':1.0
                        }
                        for i in matches
                    ]
                )

        return self.nms(entities)


text = 'chị muốn hủy lịch khám với khoa tim mạch'
matcher = DepartmentMatcher()
print(matcher.match(text))
for i in matcher.match(text)['entities']:
    print(text[i['start']:i['end']])