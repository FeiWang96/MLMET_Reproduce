import os
import json
import gzip
import re
import xml.etree.ElementTree as ET
import time

#macros
pronouns = {'i', 'me', 'myself', 'we', 'us', 'ourselves', 'he', 'him', 'himself', 'she', 'her', 'herself',
            'it', 'they', 'them', 'themselves', 'you'}
nn_tags = {'NN', 'NNP', 'NNS', 'NNPS'}

def prepare_data(file_input = './data/crowd/train.json', file_output = './data/processed/train.txt'):
    '''
    @param file_input (str): input filename
    @param file_output (str): output text file, model inputs
    '''
    f_in = open(file_input, 'r', encoding='utf-8')
    f_out = open(file_output, 'w', encoding='utf-8')
    sentence_left = []
    mention = []
    sentence_right = []
    text_output = []
    for i, line in enumerate(f_in):
        x = json.loads(line)
        sentence = ''
        y_list = x.get('y_str', None)
        sentence_left = x.get('left_context_token', None)
        mention = x.get('mention_span', None)
        sentence_right = x.get('right_context_token', None)
        if y_list is not None:
            for word in y_list:
                sentence += word + ' '
            sentence = sentence[:-1]
            sentence = sentence + '\t'
            for word in sentence_left:
                sentence += word + ' '
            sentence += mention
            for word in sentence_right:
                sentence += word + ' '
            sentence += '[SEP] '
            sentence += mention + '\n'
            text_output.append(sentence)

    for line in text_output:
        f_out.writelines(line)
    f_in.close()
    f_out.close()

if __name__ == '__main__':
    print('Reformatting data...')
    start = time.time()
    prepare_data('./data/crowd/train.json', './data/processed/train.txt')
    prepare_data('./data/crowd/dev.json', './data/processed/dev.txt')
    prepare_data('./data/crowd/test.json', './data/processed/test.txt')
    end = time.time()
    print('Loading finished:\t', end - start)
    
