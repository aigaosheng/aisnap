# -*- coding: utf-8 -*-
import sys, os
from xml.etree import ElementTree as ET
import argparse
from Document.preprocess import Documents
from Utils import ContentFilter
import settings
import logging

if __name__ == '__main__':
    """
    python snapdoc.py --corpus "NOKIA CA-101.xml" --method leader --genre product --format xml --lang en --ratio 0.1
    """
    logging.basicConfig(filename='log/example.log', filemode='w', level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
    logger = logging.getLogger('snap')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename = 'log/example.log', mode = 'w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stdout)
    logger.addHandler(fh)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='', nargs='+',action='store', help='document file list')
    parser.add_argument('--method', default='leader', action='store', help = 'summary method')
    parser.add_argument('--genre', default='product', action='store', help = 'genre of document ')
    parser.add_argument('--format', default='xml', action='store', help = 'document format, xml or plain')
    parser.add_argument('--lang', default='en', action='store', help='document language')
    parser.add_argument('--ratio', default='0.1', action='store', help = 'how many percent sentences of summary length over total sentences in documents. Only used in PageRank. Leader auto setting') #only for PageRank
    parser.add_argument('--save', default='', action='store', help = 'save summary to file') #only for PageRank
    args = parser.parse_args(sys.argv[1:])
    try:
        #print ("".join(args.corpus))
        print('---------------')
        print('Summary document: ' + '; '.join(args.corpus))
        if args.method.lower() in settings.SUMMARY_METHOD:
            #print('Use < %s > method to extract summary' % args.method.lower())
            logger.debug('#######Use < %s > method to extract summary' % args.method.lower())
            logger.info('#######Use < %s > method to extract summary' % args.method.lower())
        else:
            raise ValueError('Summary method only support < %s > ' % ', '.join(settings.SUMMARY_METHOD))
        if args.genre.lower() in settings.SUMMARY_DOC_GENRE:
            print('Document genre < %s > ' % args.genre.lower())
        else:
            raise ValueError('Summary only support genre < %s > ' % ', '.join(settings.SUMMARY_DOC_GENRE))
        if args.format.lower() in settings.SUMMARY_DOC_FORMAT:
            print('Document format < %s > ' % args.format.lower())
        else:
            raise ValueError('Input document support < %s > text ' % ', '.join(settings.SUMMARY_DOC_FORMAT))
        if args.lang.upper() in settings.SUMMARY_LANG:
            print('Document language < %s > ' % args.lang.upper())
        else:
            raise ValueError('Input document support < %s > language ' % ', '.join(settings.SUMMARY_LANG))

        if float(args.ratio) > 0.:
            print('Summary length ratio %s . Only used for PageRank' % args.ratio)
        else:
            raise ValueError('Summary ration must be in (0., 1.0')
        print('---------------')

        save_file = None
        if 'save' in args:
            print('Save summary results into < %s >' % args.save)
            save_file = args.save

        #p = Documents(file_name, 'xml', stopword_file='data/stopwords_Linhong.txt', pos_file='data/positivewords.txt', neg_file='data/negativewords.txt', use_ner=False, use_postag=True)
        filter = ContentFilter.Filter(lang = args.lang.upper())
        p = Documents(args.corpus, doc_format = args.format.lower(), doc_genre = args.genre.lower(), content_filter = filter)
        summary = p.doSummer(method=args.method.lower(), ratio=float(args.ratio), weighted=True, save_file = save_file)
        print(summary)
    except:
        parser.print_help()

