import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import json
import string

class VQADataset(Dataset):
    def __init__(self, annotation_path, transform = None):
        super(VQADataset).__init__()
        with open(".\\Dataset\\VQA_Dataset\\class_to_label.json", 'r') as f:
            answer_dict = json.load(f)
        with open(annotation_path, 'r') as f:
            qa_pairs = json.load(f)
        with open(".\\Dataset\\VQA_Dataset\\word_to_token.json", 'r') as f:
            tokenizer = json.load(f)
        self.answer_dic = answer_dict
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.transform = transform
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, index):
        sample = self.qa_pairs[index]
        img = Image.open(sample['Image_dir'])
        if self.transform is not None:
            img = self.transform(img)
        question = sample['Question'].lower()
        question = question.translate(str.maketrans('', '', string.punctuation))
        question_words = question.split()
        q_tokens = [self.tokenizer.get(word, self.tokenizer["unk"]) for word in question_words]
        if len(q_tokens) < 20:
            q_tokens += [0] * (20 - len(q_tokens)) 
        else:
            q_tokens = q_tokens[:20]
        q_tokens = torch.tensor(q_tokens, dtype=torch.long)
        answer = sample['Ground_Truth']
        if answer not in self.answer_dic:
            raise ValueError(f"Answer '{answer}' not found in class_to_label.json")
        answer = self.answer_dic[answer]
        answer = torch.tensor(answer,dtype=torch.long)
        return img, q_tokens, answer