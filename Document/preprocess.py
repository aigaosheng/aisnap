# -*- coding: utf-8 -*-
"""
Load documents, and extract informative sentences for document summarization
"""
import logging
import sys
sys.path.append('...')
from gensim import corpora, models, similarities
from gensim.test.utils import get_tmpfile
import networkx as nx
import string
import re
import numpy as np
from Utils import DocumentLoader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk 
from LeaderDetector import leaderDetect
from aspect import MineFact
import jieba
from collections import namedtuple
import codecs, regex
import pyltp
import settings
from Utils import ContentFilter


class Documents(object):
    '''
    Document class to process input PLAIN text or XML file, build dictionary, vector space model, and content-based sentence-sentence similarity
    Input:
        doc_files: file list
        doc_format: support "xml" or "plain" text
        doc_genre: support "news", "product". For news, name-entity default applied; for review, POS-tag & aspect mining & sentiment analysis is default applied
        pipeline: dictionary to list available tool to process sentences, 'ner', 'split' to segment sentence, 'pos' for pos tag, 'cleaner' to remove stopwords, 'stem' to stemming words

    '''
    def __init__(self, input_str, doc_format='xml', doc_genre = 'news', content_filter = None): 
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info('Preprocess')

        #record the document meta-information
        self.sentence_graph = None
        self.content_filter = content_filter
        self._doc_info_ = {
            'file': None, 
            'format': doc_format, 
            'genre': doc_genre, 
            'lang': self.content_filter.lang
            }
        if doc_format == 'stream':
            #input is content stream
            if isinstance(input_str, unicode):
                self.contents = input_str.encode('utf-8')
            else:
                self.contents = input_str
            self._doc_info_['file'] = 'stream'
        else:
            #input is file
            if not isinstance(input_str, list):
                self.doc_files = [input_str]
            else:
                self.doc_files = input_str
            self._doc_info_['file'] = '; '.join(input_str) 
            contents = ''
            for flname in self.doc_files:
                contents += DocumentLoader.readDocument(flname, self._doc_info_['format']) + ' '
            self.contents = contents#contents.decode('utf-8')

        if self.content_filter.lang == 'EN':
            if self._doc_info_['genre'] == 'product':
                self.pipeline = {
                    settings.SENTENCE_SPLITTER: settings.getSentenceEn,
                    settings.SENTENCE_TOKENIZER: settings.getTokenEn,
                    settings.SENTENCE_POSTAG: settings.getPostagEn,
                    settings.REMOVE_STOPWORD: content_filter.isStopword,
                    settings.TOKEN_STEM: settings.getStem,
                    settings.SENTENCE_SUBJECT: content_filter.isSubject,
                    settings.SENTENCE_NER: settings.getNameEntityEn
                }
            elif self._doc_info_['genre'] == 'news':
                self.pipeline = {
                    settings.SENTENCE_SPLITTER: settings.getSentenceEn,
                    settings.SENTENCE_TOKENIZER: settings.getTokenEn,
                    settings.SENTENCE_POSTAG: settings.getPostagEn,
                    settings.REMOVE_STOPWORD: content_filter.isStopword,
                    settings.TOKEN_STEM: settings.getStem,
                    settings.SENTENCE_NER: settings.getNameEntityEn,
                    #settings.SENTENCE_SUBJECT: content_filter.isSubject
                }
            else:
                self.pipeline = {}                
                
        elif self.content_filter.lang == 'CH':
            if self._doc_info_['genre'] == 'product':
                self.pipeline = {
                    settings.SENTENCE_SPLITTER: settings.getSentenceCh,
                    settings.SENTENCE_TOKENIZER: settings.getTokenCh,
                    settings.SENTENCE_POSTAG: settings.getPostagCh,
                    settings.REMOVE_STOPWORD: content_filter.isStopword,
                    settings.SENTENCE_SUBJECT: content_filter.isSubject,
                    settings.SENTENCE_NER: settings.getNameEntityCh
            }
            elif self._doc_info_['genre'] == 'news':
                self.pipeline = {
                    settings.SENTENCE_SPLITTER: settings.getSentenceCh,
                    settings.SENTENCE_TOKENIZER: settings.getTokenCh,
                    settings.SENTENCE_POSTAG: settings.getPostagCh,
                    settings.REMOVE_STOPWORD: content_filter.isStopword,
                    #settings.SENTENCE_SUBJECT: content_filter.isSubject
                    settings.SENTENCE_NER: settings.getNameEntityCh
                }
            else:
                self.pipeline = {
                    settings.SENTENCE_SPLITTER: settings.getSentenceCh,
                    settings.SENTENCE_TOKENIZER: settings.getTokenCh,
                    #settings.SENTENCE_POSTAG: settings.getPostagCh,
                    settings.REMOVE_STOPWORD: content_filter.isStopword,
                    #settings.SENTENCE_SUBJECT: content_filter.isSubject
                    #settings.SENTENCE_NER: settings.getNameEntityCh
                }
                
        else:
            self.pipeline = {}    

        self.index(self.content_filter.lang)

    #create sentence graph
    def index(self, lang = 'EN'):
        self.sentence_graph = nx.Graph()
        #sentence segmentation
        if self.pipeline[settings.SENTENCE_SPLITTER]:
            self.sentences = filter(lambda ss: len(ss) > 0, self.pipeline[settings.SENTENCE_SPLITTER](self.contents))
        else:
            raise ValueError('Sentence segmemtor must be set in pipeline["split"]')
        
        self.sent_tokenized = None
        if self.pipeline[settings.SENTENCE_TOKENIZER]:
            if lang == 'CH':
                if not settings.USE_JIEBA:
                    self.sent_tokenized = [self.pipeline[settings.SENTENCE_TOKENIZER](ss) for ss in self.sentences]
                #else:
                #    self.sent_tokenized = [self.pipeline[settings.SENTENCE_TOKENIZER](ss) for ss in self.sentences]

            elif lang == 'EN':
                self.sent_tokenized = [self.pipeline[settings.SENTENCE_TOKENIZER](ss) for ss in self.sentences]
            else:
                raise ValueError('language only support EN, CH')
        else:
            raise ValueError('Word tokenizer must be set in pipeline["tokenizer"]')

        self.sent_postag = None
        try:
            #self.pipeline[settings.SENTENCE_POSTAG]:
            if settings.USE_JIEBA:
                self.sent_postag = []
                self.sent_tokenized = []
                for ss in self.sentences:#self.sent_tokenized:
                    words, tags = self.pipeline[settings.SENTENCE_POSTAG](ss)
                    self.sent_tokenized.append(words)
                    self.sent_postag.append(tags)

            else:
                self.sent_postag = [self.pipeline[settings.SENTENCE_POSTAG](ss) for ss in self.sent_tokenized]
        except:
            pass
        
        self.sent_ner = None
        self.ner_words = []
        #if self.pipeline[settings.SENTENCE_NER]:
        try:
            name_cand_tag = settings.ASPECT_CONFIG['EN']['ner']
            self.sent_ner = []
            for ss, sstag in zip(self.sent_tokenized, self.sent_postag):
                self.sent_ner.append(self.pipeline[settings.SENTENCE_NER](ss, sstag))
            name_words = set()
            for sstag, ssword in zip(self.sent_ner, self.sent_tokenized):
                for tg, wd in zip(sstag, ssword):
                    if tg in settings.ASPECT_CONFIG[self._doc_info_['lang']]['ner']:
                        if not isinstance(wd, unicode):
                            wd = wd.decode('utf-8')
                        name_words.add(wd.lower())
            self.ner_words = [[wd] for wd in name_words]
        except:
            pass

        #convert sentence string & words into unicode
        if not isinstance(self.sentences[0], unicode):
            ss2 = []
            for kk, _ in enumerate(self.sentences):
                ss2.append(self.sentences[kk].decode('utf-8'))
            self.sentences= ss2
            for kk, ss in enumerate(self.sent_tokenized):
                ss2 = []
                for jj, wd in enumerate(ss):
                    ss2.append(wd.decode('utf-8').lower())
                self.sent_tokenized[kk] = ss2
        
        self.sent_subject = None
        #if self.pipeline[settings.SENTENCE_SUBJECT]:
        try:
            #positive > 0., negative < 0., neutral = 0.0
            self.sent_subject = [self.pipeline[settings.SENTENCE_SUBJECT](ss) for ss in self.sent_tokenized]
        except:
            self.sent_subject = [settings.POSITIVE_SEN for ss in self.sent_tokenized] #all sentence used

        self.sent_stem = None
        #if self.pipeline[settings.TOKEN_STEM]:
        try:
            self.sent_stem = [self.pipeline[settings.TOKEN_STEM](ss) for ss in self.sent_tokenized]  
        except:
            self.sent_stem = self.sent_tokenized

        self.sent_clean = None
        #if self.pipeline[settings.REMOVE_STOPWORD]:
        try:
            self.sent_clean = [self.pipeline[settings.REMOVE_STOPWORD](ss) for ss in self.sent_stem]  
        except:
            self.sent_clean = self.sent_stem


        #Filter sentences using subjective detector 
        s_count = 0
        self.senIdMap = {} #index new pos, value: raw pos
        self.senIdMap2clean = {} #index is raw sentence position, value: new pos
        self.sent_for_summary = []
        used_sen_id = []
        if True:#self._doc_info_['genre'] == 'product':
            for k in range(len(self.sent_clean)):
                if self.sent_subject[k] != settings.NEUTRAL_SEN: 
                    #filter sentence by subjective detector
                    self.senIdMap[s_count] = k
                    self.sent_for_summary.append(self.sent_clean[k])
                    used_sen_id.append(k)
                    self.senIdMap2clean[k] = s_count
                    s_count += 1
        else:
            for k in range(len(self.sent_clean)):
                if True: 
                    #filter sentence by subjective detector
                    self.senIdMap[s_count] = k
                    self.sent_for_summary.append(self.sent_clean[k])
                    used_sen_id.append(k)
                    self.senIdMap2clean[k] = s_count
                    s_count += 1
            

        #extract frequenct noun/noun-phrase as aspects, particullarly for review data
        self.aspects = []
        aspect_score = np.zeros(shape=(len(used_sen_id), len(used_sen_id)), dtype = float)
        if True:#self._doc_info_['genre'] == 'product':
            self.noun_sen = self.findTransaction()
            self.aspects = MineFact(self.noun_sen, min_support_v = settings.MIN_SUPPORT_ASPECT_FREQ, pattern_len = settings.NGRAM_PATTERN, topn = settings.TOPN_ASPECT)
            print('--------------')

        if len(self.ner_words):
            for w in self.ner_words:
                if w not in self.aspects:
                    self.aspects.append(w)
        self.topics = dict()
        for tid, tp in enumerate(self.aspects):
            self.topics[' '.join(tp)] = tid
    
        """
        print('------- Aspect found -----')
        for asp in self.aspects:
            print(''.join(asp))
        """

        if len(self.aspects):
            # ---------- Aspect based score ----------
            #extract aspect, pattern_len<=2 is fully debuged
            #aspect based similarity score
            self.label_sents, self.sen_label, self.topic_sen_index = self.AspectLabel(lambda x: '%s'%x)

            for kk, _ in enumerate(used_sen_id): #self.label_sents):
                ss = self.label_sents[used_sen_id[kk]]
                for kk2a, _ in enumerate(used_sen_id[kk+1:]):#self.label_sents[kk+1:]):                    
                    kk2 = kk2a + kk + 1
                    ss2 = self.label_sents[used_sen_id[kk2]]
                    aspect_score[kk, kk2] = self.AspectSimilarity(ss, ss2)

        dct = corpora.Dictionary(self.sent_for_summary)
        bow_corpus = [dct.doc2bow(ss) for ss in self.sent_for_summary]
        #model = models.TfidfModel(bow_corpus)
        #feat = []
        #vector = model[bow_corpus]
        #content based similarity
        index_temp = 'cache/docindex'#get_tmpfile("index")
        index = similarities.Similarity(index_temp, bow_corpus, num_features=len(dct.token2id), num_best = None, norm='l2')  # create index
        cscore = np.zeros(shape=(len(bow_corpus), len(bow_corpus)), dtype = float)
        for k, dv in enumerate(bow_corpus):
            cscore[k,:] = index[dv]
            cscore[k, k] = 0.
        #other feature based similarity, TO DO
        self.score = settings.SCORE_FUSE_WEIGHT['content'] * cscore + settings.SCORE_FUSE_WEIGHT['aspect'] * aspect_score

        #find threshold based on score distribution
        ssc = self.score[self.score > 0.]
        ssc = np.log(ssc)
        sc_mean = np.mean(ssc)
        sc_std = np.std(ssc)
        self.knn_threshold = sc_mean + sc_std * settings.MAX_STD_RATIO
        self.knn_threshold = np.exp(self.knn_threshold)

        #build a non-weighted graph from kNN, i.e. symetric 
        isknn = np.zeros(self.score.shape, dtype = bool)
        for k in range(self.score.shape[0]):
            isknn[k,:] = self.score[k,:] > self.knn_threshold
        isknn = isknn & np.transpose(isknn)
        self.sentence_graph.add_nodes_from(range(self.score.shape[0]))
        for k in range(self.score.shape[0]):
            for j in range(k+1, self.score.shape[1]):
                if isknn[k,j]:
                    self.sentence_graph.add_edge(k, j, similarity = self.score[k,j], distance = 1.0 - self.score[k,j])


    def doSummer(self, method = 'pagerank', ratio = 0.1, weighted = True, save_file = None):
        fo_save = None
        if save_file:
            fo_save = codecs.open(save_file, 'w', encoding='utf-8')
            fo_save.write('[Aspect]\n')
            for asp in self.aspects:
                fo_save.write(''.join(asp) + '\n')

            fo_save.write('\n[Summary]\n')

        self.summary = []
        self.summary_len = min(max(1, int(ratio * len(self.sentences))), settings.MAX_SUMMARY_LEN)
        sum_sen_count = 0
        if method == 'pagerank':
            if weighted:
                pr = nx.pagerank(self.sentence_graph, alpha=0.85, weight = 'similarity')
            else:
                pr = nx.pagerank(self.sentence_graph, alpha=0.85, weight = None)
            ordered_sentence = sorted(pr.items(), key=lambda (v,k):(k,v), reverse=True)
            for k in range(self.summary_len):
                sid = self.senIdMap[ordered_sentence[k][0]]
                self.summary.append(self.sentences[sid])
                if fo_save:
                    fo_save.write('%d: %s\n'% (k, self.sentences[sid]))

                sum_sen_count += 1
            print('Compression ratio including all leaders %f ' % (float(len(self.summary))/len(self.sentences)))
                
        elif method == 'leader':
            leader = leaderDetect.CommunityDetector(self.sentence_graph)

            topic_pool = self.topics.keys()
            sum_cand_id = set()
            sum_sen_count = 0
            for n, nd in enumerate(leader.leader):
                if settings.USE_ASPECT_SELECT:
                    self.summary_sen_id = []
                    if len(leader.community[n]) + 1 < settings.MIN_COMMUNITY_SIZE: # 
                        break
                    if len(topic_pool) == 0:
                        break
                    for cm_id in [nd] + leader.community[n]:
                        cm_id2 = self.senIdMap[cm_id] #sentence position in raw data without sentence filtering
                        if  cm_id2 not in self.sen_label:
                            continue
                        for tp in self.sen_label[cm_id2]:
                            #check aspect label of each sentence in community
                            if tp in topic_pool:
                                p_ct, n_ct = [], []
                                for sid in self.topic_sen_index[tp]:
                                    try:
                                        if self.sent_subject[self.senIdMap2clean[sid]] ==       settings.POSITIVE_SEN:
                                            p_ct.append(sid)
                                        elif self.sent_subject[self.senIdMap2clean[sid]] ==     settings.NEGATIVE_SEN:
                                            n_ct.append(sid)
                                        else:
                                            pass
                                    except:
                                        pass
                                if len(p_ct) > 0:
                                    if p_ct[0] not in sum_cand_id:                                   
                                        ostr = '\n %s [ %s: %d/%d like] ' % (self.sentences[p_ct[0]], tp, len(p_ct), len(p_ct) + len(n_ct))
                                        #print(p_ct[0])
                                        self.summary.append(ostr)
                                        if fo_save:
                                            fo_save.write('%s\n'% ostr)
                                        sum_cand_id.add(p_ct[0])
                                        sum_sen_count += 1
                                if len(n_ct) > 0:
                                    if n_ct[0] not in sum_cand_id:
                                        ostr = '\n %s [ %s: %d/%d dislike] ' % (self.sentences[n_ct[0]], tp, len(n_ct), len(p_ct) + len(n_ct))
                                        #print(n_ct[0])
                                        self.summary.append(ostr)
                                        if fo_save:
                                            fo_save.write('%s\n'% ostr)
                                        sum_cand_id.add(n_ct[0])
                                        sum_sen_count += 1
                                else:
                                    pass
                                if len(n_ct) or len(p_ct):
                                    topic_pool.remove(tp)
    
                else:
                    if len(leader.community[n]) + 1 < settings.MIN_COMMUNITY_SIZE: # 
                        break
                    self.summary.append('\n** %d-th Summary Sentence, Community size %d ***' % (n, len(leader.community[n])+1))
                    self.summary.append(self.sentences[self.senIdMap[nd]])
    
                    if fo_save:
                        fo_save.write('%d: %s\n'% (n, self.sentences[self.senIdMap[nd]]))
                    sum_sen_count += 1
                
                    '''
                    self.summary.append('******* Non-leader sentences **********')
                    for i in leader.community[n]:
                        self.summary.append(self.sentences[self.senIdMap[i]])
                    '''            
                #sum_sen_count += 1

            if fo_save:
                fo_save.write('\n[Compression ratio]\n%f (%d/%d)\n\n' % (float(sum_sen_count)/len(self.sentences), sum_sen_count, len(self.sentences)))

            self.summary.append('\n\n******Outlier sentence %d'%len(leader.outlier))
            if fo_save:
                fo_save.write('\n[Outlier sentences]\n')
            for jj, i in enumerate(leader.outlier):
                self.summary.append('%d: %s'% (jj, self.sentences[self.senIdMap[i]]))
                if fo_save:
                    fo_save.write('%d: %s\n' % (jj, self.sentences[self.senIdMap[i]]))

            print('Compression ratio including all leaders %f (%d/%d)' % (float(sum_sen_count)/len(self.sentences), sum_sen_count, len(self.sentences)))

        if fo_save:
            fo_save.write('\n[Input Document]\n')
            fo_save.write('\n'.join(self.sentences))
            fo_save.close()            
            
        return '\n'.join(self.summary)


        #release memory for Chinese
        if self._doc_info_['lang'] == 'CH':
            settings.releaseToolCh() 
                        
        #print(self.sentences[0])
        #print(self.sent_tokenized[0])
        #print(self.sent_postag[2])
        #for ss in all_sens:
        #    print(ss)
        #print(self.pos_words)
        #print(self.neg_words)

    def alignNer(self, sen_ner_tag):
        align_ner_word = []
        for ele in sen_ner_tag:
            if hasattr(ele, 'label'):
                name_word = ' '.join(c[0] for c in ele.leaves())
                name_pos_tag = ele.leaves()[0][1]
                name_type = ele.label()
                align_ner_word.append((name_word.lower(), name_pos_tag, name_type))
            else:
                if isinstance(ele, tuple):
                    align_ner_word.append((ele[0].lower(), ele[1], None))
                else:
                    align_ner_word.append((ele.lower(), None, None))
        return align_ner_word
        
    def cleaner(self, sentence):
        #remove stop words
        #return [(SnowballStemmer('english').stem(wd[0]), wd[0],wd[1], wd[2]) for wd in sentence if wd[0] not in self.stop_words and wd[0].isalnum()]
        #return [wd[0] for wd in sentence if wd[0] not in self.stop_words and wd[0].isalnum()]
        return sentence

    def stemming(self, sentence):
        #remove stop words
        return [(SnowballStemmer('english').stem(wd[0]), wd[0],wd[1], wd[2]) for wd in sentence]
        #return [SnowballStemmer('english').stem(wd[0]) for wd in sentence]

    #check adjective is in the neighbour around current location
    def isAdjNeighbor(self, last_loc_noun, noun_len, sen_tag):
        isAdj = False
        #forward search adjective
        for ii in range(last_loc_noun + 1, min(last_loc_noun+1+settings.ASPECT_CONFIG[self._doc_info_['lang']]['constrain_window'], len(sen_tag))):
            if sen_tag[ii][2] in settings.ASPECT_CONFIG[self._doc_info_['lang']]['noun_tag']:
                break
            if sen_tag[ii][2] in settings.ASPECT_CONFIG[self._doc_info_['lang']]['adj_tag']:                
                isAdj = True
                break
        if not isAdj:
            for ii in range(max(last_loc_noun-noun_len, 0), max(last_loc_noun-noun_len-settings.ASPECT_CONFIG[self._doc_info_['lang']]['constrain_window'], -1), -1):
                if sen_tag[ii][2] in settings.ASPECT_CONFIG[self._doc_info_['lang']]['noun_tag']:
                    break
                if sen_tag[ii][2] in settings.ASPECT_CONFIG[self._doc_info_['lang']]['adj_tag']:                
                    isAdj = True
                    break
        return isAdj
    
    #check sententwords is in the neighbour around current location
    def isSentimentNeighbor(self, last_loc_noun, noun_len, sen_tag):
        isPolarity = False
        #forward search adjective
        for ii in range(last_loc_noun + 1, min(last_loc_noun+1+settings.ASPECT_CONFIG[self._doc_info_['lang']]['constrain_window'], len(sen_tag))):
            if self.content_filter.isPositive(sen_tag[ii]) or self.content_filter.isNegative(sen_tag[ii]):
                isPolarity = True
                break
        if not isPolarity:
            for ii in range(max(last_loc_noun-noun_len, 0), max(last_loc_noun-noun_len-settings.ASPECT_CONFIG[self._doc_info_['lang']]['constrain_window'], -1), -1):
                if self.content_filter.isPositive(sen_tag[ii]) or self.content_filter.isNegative(sen_tag[ii]):
                    isPolarity = True
                    break
        return isPolarity

    def sentimentDetector(self, sen):
        isSujective = False
        sentiment_count = {'pos':0, 'neg': 0}
        for wd, _, tag, ner in sen:
            if self.content_filter.isPositive(wd):
                sentiment_count['pos'] += 1
            elif self.content_filter.isNegative(wd):
                sentiment_count['neg'] += 1
            else:
                pass
        if sentiment_count['pos'] + sentiment_count['neg'] > 0:
            isSujective = True
        return isSujective

    #summarize single community
    def SummarizeCommunity(self, sen_list):
        select_topic = []
        select_sen_id = []
        for tp in self.aspect_cand:
            if tp in select_topic:
                continue
            topic = ' '.join(tp)
            for i in sen_list:
                ifound = re.findall(r'\b%s\b'%topic, self.sent_processed_seq[self.senIdMap[i]])
                if len(ifound):
                    select_topic.append(tp)
                    if len(tp) > 1:
                        for tw in tp:
                            select_topic.append(tw)
                    select_sen_id.append(i)
                    break
        #remove select topic
        for tp in select_topic:
            try:
                self.aspect_cand.remove(tp)
            except:
                pass
        self.summary_sen_id += select_sen_id

    #tag aspect for sentence
    def AspectLabel(self, pat_func):
        #topics = []
        #for tp in self.aspects:
        #    topics.append(' '.join(tp))

        aspect_sents = []
        sen_idx = dict()
        topic_sen_idx = dict()
        for ss_id, ss in enumerate(self.sentences):
            tag = set()
            for tp in self.topics:
                #regex.compile('\b%s\b')
                ifound = regex.findall(pat_func(tp), ss) #(r'\b%s\b'%tp, ss)
                if len(ifound):
                    tag.add(tp)
                    try:
                        sen_idx[ss_id].append(tp)
                    except:
                        sen_idx[ss_id] = [tp]
                    try: 
                        topic_sen_idx[tp].append(ss_id)
                    except:
                        topic_sen_idx[tp] = [ss_id]

            aspect_sents.append(tag)
        return aspect_sents, sen_idx, topic_sen_idx

    def AspectSimilarity(self, ss, ss2):
        sc = 0.
        if len(ss) and len(ss2):
            sc = float(len(ss.intersection(ss2))) / (float(len(ss.union(ss2))))
        return sc


    def findTransaction(self):
        '''
        from pos-tagged sentence (plus stemming) extract noun/noun-phrase transaction sequence
        to keep word position info, do no remove stopwords
        '''
        noun_sen = []
        for kk, ss2 in enumerate(zip(self.sent_stem, self.sent_postag)):
            if self.sent_subject[kk] == settings.NEUTRAL_SEN: #skip objective sentence
                continue
            noun_seq = []
            pre_noun_loc = 0
            sen = []
            loc = 0
            for wd, tag in zip(ss2[0], ss2[1]):
                if tag in settings.ASPECT_CONFIG[self._doc_info_['lang']]['noun_tag']:
                    if len(noun_seq):
                        if loc - pre_noun_loc == 1:
                            noun_seq.append(wd)
                        else:
                            if self.isSentimentNeighbor(pre_noun_loc, len(noun_seq), ss2[0]):
                                noun_sen.append(noun_seq)
                            noun_seq = [wd]
                        pre_noun_loc = loc
                    else:
                        noun_seq.append(wd)
                        pre_noun_loc = loc
                loc += 1
            if len(noun_seq):
                if self.isSentimentNeighbor(pre_noun_loc, len(noun_seq), ss2[0]):
                    noun_sen.append(noun_seq)
        return noun_sen
                


        
