import pyximport; pyximport.install()

import os
import re
import json
import jieba
import pickle
import logging
from bs4 import BeautifulSoup
from collections import Counter

def cut_sent(para):
    ''' Cut Chinese string into multiple sentences. '''
    para = re.sub('([，；：。！？\?])([^』」”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^』」”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^』」”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][』」”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split('\n')


if __name__ == '__main__':
    assert not os.path.exists('finished_files')
    cmd = os.popen('mkdir finished_files/')
    cmd = os.popen('mkdir finished_files/train')
    cmd = os.popen('mkdir finished_files/val')
    cmd = os.popen('mkdir finished_files/test')
    jieba.setLogLevel(logging.ERROR)
    jieba.initialize()
    
    # Count entries in LCSTS
    filename = 'LCSTS/DATA/PART_I.txt'
    cmd = os.popen('cat %s | grep -c "</doc>"' % filename)
    data_count = int(cmd.read().strip())

    # Data counter
    data_iter = 0
    train_iter = 0
    val_iter = 0
    test_iter = 0
    vocab_counter = Counter()

    # Collect lines for soup
    print('Preparing LCSTS for fast_abs_rl...')
    line_buffer = ''
    for line in open(filename):
        strip_line = line.strip()
        line_buffer += strip_line

        # Soup entry
        if strip_line == '</doc>':
            try:
                soup = BeautifulSoup(line_buffer, features='html.parser')
                line_buffer = ''

                # Process data as json
                doc_id = soup.find('doc').attrs['id']
                document = soup.find('short_text').contents[0]
                summary = soup.find('summary').contents[0]
                doc_sents = cut_sent(document)
                summ_sents = cut_sent(summary)
                article = [' '.join(jieba.cut(sent)) for sent in doc_sents]
                abstract = [' '.join(jieba.cut(sent)) for sent in summ_sents]
                data = {'id': doc_id, 'article': article, 'abstract': abstract}

                # Store data as json
                cursor = data_iter % 10
                if cursor == 0:     # Val set
                    json.dump(data, open('finished_files/val/%d.json' % val_iter, 'w'))
                    val_iter += 1
                elif cursor == 1:   # Test set
                    json.dump(data, open('finished_files/test/%d.json' % test_iter, 'w'))
                    test_iter += 1
                else:   # Train set
                    json.dump(data, open('finished_files/train/%d.json' % train_iter, 'w'))
                    train_iter += 1
                
                    # Vocab counter for train set
                    art_tokens = ' '.join(article).split()
                    abs_tokens = ' '.join(abstract).split()
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens] # strip
                    tokens = [t for t in tokens if t != ""] # remove empty
                    vocab_counter.update(tokens)
            except:
                line_buffer = ''
                print('\n---- Error found in soup ----')
                print(soup.prettify())
                print('-----------------------------')
            
            # Progress bar :)
            data_iter += 1
            percentage = 100 * data_iter / data_count
            print('%d/%d (%.2f%%)\r' % (data_iter, data_count, percentage), end='')
            
    print('\nWriting vocab file...')
    pickle.dump(vocab_counter, open('finished_files/vocab_cnt.pkl', 'wb'))
    print('Script ends successfully.')
