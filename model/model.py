from logging import NOTSET
from torch import nn

from model.layer import WordRepresentation, FeedforwardLayer, BiaffineLayer
from transformers import AutoConfig
import os
from pathlib import Path
from preprocess.static_features import Feature
from model.layer.featureEmbed import FeatureRep
import torch
from transformers import AutoModel, AutoTokenizer
from model.layer.crf import CRF
from model.layer.utils import get_extended_attention_mask
from logging import NOTSET
from torch import nn

from model.layer import WordRepresentation, FeedforwardLayer, BiaffineLayer
from transformers import AutoConfig
import os
from pathlib import Path
from preprocess.static_features import Feature,FeatureExtractor
from preprocess.processcer_join_bert import NERProcessor
from model.layer.featureEmbed import FeatureRep
import torch
from transformers import AutoModel, AutoTokenizer
from model.layer.crf import CRF
from model.layer.utils import get_extended_attention_mask
from underthesea import word_tokenize
# text=word_tokenize(text, format="text")
class BertCRF(nn.Module):
    def __init__(self, n_layer_bert = -1, cfg_feat=None, pretrained_bert='vinai/phobert-base'):
        super().__init__()
        if cfg_feat is None:
            path = Path(__file__).parent.absolute()
            path = Path(path).parent.absolute()
            cfg_feat = os.path.join(path,"resources/features/feature_config.json")
        self.feats=Feature(cfg_feat)
        self.word_net  = FeatureRep(self.feats)
        path = Path(__file__).parent.absolute()
        path = Path(path).parent.absolute()
        self.fe = FeatureExtractor(dict_dir=os.path.join(path,'resources/features'))
        
        b = AutoModel.from_pretrained(pretrained_bert).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        #----------------
        processor = NERProcessor("./dataset", self.tokenizer)
        processor.labels=["O",'B-GENDER','I-GENDER','B-LOC','I-LOC']
        processor.label_map= {label: i for i, label in enumerate(processor.labels, 1)}
        self.processor=processor
        #-----------------------------------
        self.bert_embed =b.base_model.embeddings 
        
        self.bert_layers = nn.ModuleList([b.base_model.encoder.layer[i] for i in range(n_layer_bert)])
        
        self.lstm = nn.LSTM(768 + 23, 768//2, batch_first=True, num_layers=2,bidirectional=True)
        
        self.crf = CRF(num_tags=6, batch_first=True)
        
        self.slot_classifier  = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(768, 6)
        )
    def forward(self, input_ids, attention_mask,  feat_rep):
        
        bert_output = self.bert_embed(input_ids)
        
        attention_mask = get_extended_attention_mask(attention_mask,input_ids.shape,input_ids.device)
        
        outputs = self.word_net(feat_rep)

        for layer in self.bert_layers:
        
            bert_output = layer(bert_output, attention_mask=attention_mask)[0]

        token_reps = torch.cat([bert_output,outputs],dim=-1)
        
        lstm_outs, _ = self.lstm(token_reps)
        
        lstm_outs=self.slot_classifier(lstm_outs)
        
        return lstm_outs    
    def calculate_loss(self,token_ids, attention_masks, token_mask, segment_ids, label_ids,
                                                      label_masks, feats ):
        
        out = self(token_ids, attention_masks, feats)
        
        slot_loss = self.crf(out, label_ids, mask=attention_masks.byte(), reduction='mean') * -1.
        
        return slot_loss,(out, label_ids)
    
    def predict(self, sentence, device='cpu'):
        sentence =word_tokenize(sentence, format="text")
#         fe = FeatureExtractor(dict_dir='resources/features')
        feats_extracted=self.fe.extract_feature(sentence, ner_labels=None, is_segmentation=False)
        fake_feats = [(0,feats_extracted[0], feats_extracted[1])]
        features = self.processor.convert_sentences_to_features(fake_feats, 100, self.feats)
        
        token_ids = features[0].token_ids
        attention_masks = features[0].attention_masks
        feats = features[0].feats
        
        token_id_tensors = torch.tensor(token_ids, dtype=torch.long).to(device=device)
        attention_mask_tensors = torch.tensor(attention_masks, dtype=torch.long).to(device=device)
        
        feat_tensors = {}
        for feat_key, feat_value in feats.items():
            feat_tensors[feat_key] = torch.tensor(feat_value, dtype=torch.long).to(device=device)
        feat_tensors = {k:v[None,...] for k,v in feat_tensors.items()}
        
        self.eval()
        with torch.no_grad():
            out = self(token_id_tensors[None,...], attention_mask_tensors[None,...], feat_tensors)
            out1 = self.crf.decode(out)[0]
        return [self.processor.labels[max(i-1,0)] for i in out1[features[0].token_mask==1]] #out1[features[0].token_mask==1],sentence
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        
        self.use_pos = args.use_pos

        self.num_labels = args.num_labels
        
        if args.bert_embed_only:
            self.lstm_input_size = config.hidden_size
        else:
            self.lstm_input_size = args.num_layer_bert * config.hidden_size

        if args.use_char:
            self.lstm_input_size = self.lstm_input_size + 2 * args.char_hidden_dim
        
        if args.use_pos:
            self.lstm_input_size = self.lstm_input_size + args.feature_embed_dim

        if args.use_fasttext:
            self.lstm_input_size = self.lstm_input_size + 300
        
        self.word_rep = WordRepresentation(args)
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=args.hidden_dim // 2,
                            num_layers=args.rnn_num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.feedStart = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        self.feedEnd = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        self.biaffine = BiaffineLayer(inSize1=args.hidden_dim, inSize2=args.hidden_dim, classSize=self.num_labels)

    def forward(self, input_ids=None, char_ids=None, fasttext_embs=None, first_subword=None, attention_mask=None, pos_ids=None,train=False):

        word_features = self.word_rep(input_ids=input_ids, 
                                    attention_mask=attention_mask,
                                    first_subword=first_subword,
                                    char_ids=char_ids,
                                    pos_ids=pos_ids,
                                    train=train)

        x, _ = self.bilstm(word_features)

        start = self.feedStart(x)
        end = self.feedEnd(x)

        score = self.biaffine(start, end)

        return score

        