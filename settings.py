import sys
import os

USE_JIEBA = False
if USE_JIEBA:
    import jieba
    import jieba.posseg as pseg


'''
Supported summary method
'''
SUMMARY_METHOD = ['pagerank', 'leader']
SUMMARY_DOC_FORMAT = ['xml', 'plain']
SUMMARY_DOC_GENRE = ['product', 'news']
SUMMARY_LANG = ['EN', 'CH']

'''
Polarity 
'''
POSITIVE_SEN = 2
NEGATIVE_SEN = 1
NEUTRAL_SEN = 0

'''
Parameters to control sentence-pair similarity & summary
'''
MAX_SUMMARY_LEN = 10
USE_ASPECT_SELECT = False#True #if True, post-filter summary sentence from community using aspect
MIN_COMMUNITY_SIZE = 1 #if community size is less or equal than it, sentene in the community is excluded from summary
MAX_STD_RATIO = 0. #it control the density of sentence graph contruction. Assume similarity score is Gaussian distributon, if pairwise sentence similarity is more than < mean + ratio * std >, then sentence pair has a link

'''
For frequenct pattern mining
'''
MIN_SUPPORT_ASPECT_FREQ = 1
NGRAM_PATTERN = 2
TOPN_ASPECT = 10

'''
Constant settings for English
'''
STOPWORD_EN = 'data/stopwords_Linhong.txt'
POSITIVE_LEXICON_EN = 'data/positivewords.txt'
NEGATIVE_LEXICON_EN = 'data/negativewords.txt'


'''
Constant settings for Chinese
'''
STOPWORD_CH = 'data/ch_stopwords.txt'
POSITIVE_LEXICON_CH = 'data/ch_positive.txt'
NEGATIVE_LEXICON_CH = 'data/ch_negative.txt'

'''
Constant names for NLP tools
'''
SENTENCE_SPLITTER = 'split'
SENTENCE_TOKENIZER = 'tokenizer'
SENTENCE_POSTAG = 'postag'
SENTENCE_NER = 'ner'
REMOVE_STOPWORD = 'stopword'
SENTENCE_SUBJECT = 'subject'
TOKEN_STEM = 'stem'

'''
Aspect mining configure
'''
ASPECT_CONFIG = {
    'EN':{ 
        'noun_tag':['NN', 'NNS'],
        'constrain_window': 1,
        'adj_tag': ['JJ', 'JJR', 'JJS'],
        'ner': ['person', 'organization', 'location']
    },
    'CH':{
        'noun_tag':['n', 'ni', 'nl', 'ns'],
        'constrain_window': 1,
        'adj_tag': ['a'],     
        'ner': ['nh', 'ni', 'ns']
    }
}
USE_ASPECT_SELECT = True #True: show aspect summary, one aspect one sentence, plus polarity distribution
SCORE_FUSE_WEIGHT = {'content':0.5, 'aspect': 0.5}

'''
'''
#for Chinese
from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

#hold global instance of Chinese NLP tool
SEGMENTOR_CH = None
POSTAG_CH = None
NER_CH = None
PARSER_CH = None
SRL_CH = None

LTP_DATA_PATH = '/media/gao/OS/WORK/tools/ltp_data'#'/Users/shenggao/dev/ltp_data_v3.4.0' #

def getSentenceCh(contents):
    '''
    split contents into sentences
    return list of string, element is sentence
    '''
    sentences = SentenceSplitter.split(contents)
    return sentences

def getTokenCh(sentence):
    '''
    for Chinese, it is word segmentation
    input: a sentence string
    return: list of words
    '''
    if USE_JIEBA:
        words = [wd.encode('utf-8') for wd in jieba.cut(sentence, cut_all=False)]
    else:
        global SEGMENTOR_CH
        if SEGMENTOR_CH == None:
            SEGMENTOR_CH = Segmentor()
            SEGMENTOR_CH.load(os.path.join(LTP_DATA_PATH, "cws.model"))
        words = SEGMENTOR_CH.segment(sentence)

    return words

def getPostagCh(tokenized_sentence):
    '''
    From word segmentation output, i.e. getTokenCh, tag word using Part-of-Speech
    '''
    if USE_JIEBA:
        postag_words = pseg.cut(tokenized_sentence)
        words = []
        ptag = []
        for w, t in postag_words:
            if isinstance(w, unicode):
                words.append(w.encode('utf-8'))
            else:
                words.append(w)
            ptag.append(t)
        return words, ptag
    else:    
        global POSTAG_CH
        if POSTAG_CH == None:
            POSTAG_CH = Postagger()
            POSTAG_CH.load(os.path.join(LTP_DATA_PATH, "pos.model"))
        postag_words = POSTAG_CH.postag(tokenized_sentence)
        ptag = [w for w in postag_words]
    return ptag

def getNameEntityCh(tokenized_sentence, postag_words):
    '''
    From pos-tag output, i.e. getPostagCh, extract name entity
    '''
    global NER_CH
    if NER_CH == None:
        NER_CH = NamedEntityRecognizer()
        NER_CH.load(os.path.join(LTP_DATA_PATH, "ner.model"))
    nertag = NER_CH.recognize(tokenized_sentence, postag_words)
    return [w.lower().split('-')[-1] for w in nertag]

def getParserCh(tokenized_sentence, postag_words):
    '''
    Parse sentence based on tokenized words & pos-tag
    '''
    global PARSER_CH
    if PARSER_CH == None:
        PARSER_CH = Parser()
        PARSER_CH.load(os.path.join(LTP_DATA_PATH, "parser.model"))
    arcs = parser.parse(tokenized_sentence, postag_words)
    return arcs

def getSrlCh(tokenized_sentence, postag_words, arcs):
    global SRL_CH
    if SRL_CH == None:
        SRL_CH = SementicRoleLabeller()
        SRL_CH.load(os.path.join(LTP_DATA_PATH, "pisrl.model"))
    roles = SRL_CH.label(tokenized_sentence, postag_words, arcs)
    return roles

def releaseToolCh():
    '''
    release resources allocated for Chinese tools
    '''
    global SEGMENTOR_CH, POSTAG_CH, NER_CH, PARSER_CH, SRL_CH
    if SEGMENTOR_CH:
        SEGMENTOR_CH.release()
    if POSTAG_CH:
        POSTAG_CH.release()
    if NER_CH:
        NER_CH.release()
    if PARSER_CH:
        PARSER_CH.relesae()
    if SRL_CH:
        SRL_CH.release()

'''
NLP for English
'''
from nltk.tokenize import sent_tokenize
from nltk import pos_tag, ne_chunk, word_tokenize
from nltk.chunk import tree2conlltags
from nltk.stem.snowball import SnowballStemmer

def getSentenceEn(contents):
    '''
    split contents into sentences
    return list of string, element is sentence
    '''
    sentences = sent_tokenize(contents)
    return sentences

def getTokenEn(sentence):
    '''
    for Chinese, it is word segmentation
    input: a sentence string
    return: list of words
    '''
    words = word_tokenize(sentence)
    return words

def getPostagEn(tokenized_sentence):
    '''
    From word segmentation output, i.e. getTokenEn, tag word using Part-of-Speech
    '''
    postag_words = [tag for _, tag in pos_tag(tokenized_sentence)]
    return postag_words

def getNameEntityEn(tokenized_sentence, postag_words):
    '''
    From pos-tag output, i.e. getPostagCh, extract name entity
    '''
    sent = [(w,tg) for w, tg in zip(tokenized_sentence, postag_words)]
    nertag = tree2conlltags(ne_chunk(sent))
    return [tg.lower().split('-')[-1] for _, _, tg in nertag]

def getParserEn(tokenized_sentence, postag_words):
    '''
    Parse sentence based on tokenized words & pos-tag
    '''
    pass

def getSrlEn(tokenized_sentence, postag_words, arcs):
    pass

def getStem(sent):
    o_sent = map(SnowballStemmer('english').stem, sent)
    return o_sent
