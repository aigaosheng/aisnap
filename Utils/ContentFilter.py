# -*- coding: utf-8 -*-
from xml.etree import ElementTree as ET
import sys
sys.path.append('...')
import settings

class Filter(object):
    def __init__(self, lang = 'EN'):
        self.lang = lang
        self.stopwords = set()
        self.posword = set()
        self.negword = set()
        if lang == 'EN':
            self.stopwords = self.__readStopwords__(settings.STOPWORD_EN)
            self.posword = self.__readSentimentWords__(settings.POSITIVE_LEXICON_EN)
            self.negword = self.__readSentimentWords__(settings.NEGATIVE_LEXICON_EN)
        elif lang == 'CH':
            self.stopwords = self.__readStopwords__(settings.STOPWORD_CH)
            self.posword = self.__readSentimentWords__(settings.POSITIVE_LEXICON_CH)
            self.negword = self.__readSentimentWords__(settings.NEGATIVE_LEXICON_CH)
        else:
            pass
        
    #read stopword list
    def __readStopwords__(self, file_name):
        ''' load stop-words '''
        words = set()
        with open(file_name, 'r') as fin:
            print('Load stopwords %s' % file_name)
            for ln in fin:
                wd = ln.strip('\n\t\r ')
                if not isinstance(wd, unicode):
                    wd = wd.decode('utf-8')
                words.add(wd.lower())
        return words

    #read sentiment words
    def __readSentimentWords__(self, file_name):
        ''' load sentiment-words '''
        words = set()
        with open(file_name, 'r') as fin:
            print('Load sentiment words %s' % file_name)
            for ln in fin:
                wd = ln.strip('\n\t\r ').lower()
                if not isinstance(wd, unicode):
                    wd = wd.decode('utf-8')
                words.add(wd)
        return words
    
    def isStopword(self, sent):
        o_sent = filter(lambda word: word not in self.stopwords and word.isalnum(), sent)
        return o_sent

    def isPositive(self, word):
        if not isinstance(word, unicode):
            word = word.decode('utf-8')
        return word in self.posword

    def isNegative(self, word):
        if not isinstance(word, unicode):
            word = word.decode('utf-8')
        return word in self.negword
    
    def isSubject(self, sent):
        #is_subjective_sen = 0.0#False
        pos_score, neg_score = 0., 0.
        for wd in sent:
            if self.isPositive(wd):
                pos_score += 1.
            if self.isNegative(wd):
                neg_score += 1.
        score = pos_score - neg_score
        if score > 0.:
            is_subjective_sen = settings.POSITIVE_SEN
        elif score < 0.:
            is_subjective_sen = settings.NEGATIVE_SEN
        else:
            is_subjective_sen = settings.NEUTRAL_SEN
            
        return is_subjective_sen