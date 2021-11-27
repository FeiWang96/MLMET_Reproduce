import json

import time
import argparse
from tqdm import tqdm


def prepare_train_data(file_input='./data/crowd/train.json', file_output='./data/processed/train.txt'):
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
    for i, line in tqdm(enumerate(f_in)):
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


# create 6 different patterns of masked data
def prepare_masked_data(file_input='./data/crowd/train.json', file_output='./data/processed/train.txt'):
    '''
    @param file_input (str): input filename
    @param file_output (str): output text file, model inputs
    '''

    f_in = open(file_input, 'r', encoding='utf-8')
    sentence_left = []
    mention = []
    sentence_right = []
    text_output = []
    patterns = [
        'and any other [MASK] ',
        # 'and some other [MASK] ',
        # '[MASK] such as ',
        # 'such [MASK] as ',
        # '[MASK] including ',
        # '[MASK] especially '
    ]
    f_out = []
    iter = len(patterns)
    for i in range(iter):
        f_out.append(open(file_output + '.' + 'pattern_' + str(i), 'w', encoding='utf-8'))

    for i, line in tqdm(enumerate(f_in)):
        x = json.loads(line)
        y_list = x.get('y_str', None)
        sentence_left = x.get('left_context_token', None)
        mention = x.get('mention_span', None)
        sentence_right = x.get('right_context_token', None)
        # write
        for p in range(iter):
            sentence = ''
            if y_list is not None:
                for word in y_list:
                    sentence += word + ' '
                sentence = sentence[:-1]
                sentence = sentence + '\t'
                for word in sentence_left:
                    sentence += word + ' '
                if p == 0 or p == 1:
                    sentence += mention + ' '
                    sentence += patterns[p]
                else:
                    sentence += patterns[p]
                    sentence += mention + ' '
                for word in sentence_right:
                    sentence += word + ' '
                sentence = sentence[:-1]
                sentence += '\n'
                f_out[p].writelines(sentence)

    for i in range(iter):
        f_out[i].close()
    f_in.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for preparing data.')
    parser.add_argument('-i', type=str, default='./data/distant.json', help='input file')
    parser.add_argument('-o', type=str, default='./data/train_distant.txt', help='output file')
    parser.add_argument('-m', type=int, default=0,
                        help='generation mode: 0 - for training data; 1 - for masked training data')
    args = parser.parse_args()
    print('Reformatting data...')
    start = time.time()
    if args.m != '':
        if args.m == 0:
            prepare_train_data(args.i, args.o)
        elif args.m == 1:
            prepare_masked_data(args.i, args.o)
        else:
            print('Arg error.')
    else:
        print('Arg error.')

    # debug
    # prepare_data('./data/crowd/dev.json', './data/processed/dev.txt')
    # prepare_masked_data('./data/crowd/dev.json', './data/processed/dev.txt')

    end = time.time()
    print('Loading finished:\t', end - start)
