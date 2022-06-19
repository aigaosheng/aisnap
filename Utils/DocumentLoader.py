# -*- coding: utf-8 -*-
from xml.etree import ElementTree as ET

def readDocument(file_name, doc_format = 'xml'):
    '''
    Read content part from XML file or PLAIN text
    Input: document file list and document format
    Return: string of content
    '''
    reviews = []
    if doc_format == 'xml':
        per = ET.parse(file_name)
        p = per.findall('./reviews/review')
        for oneper in p:
            for child in oneper.getchildren():
                if child.tag == 'content':
                    reviews.append(child.text.strip("\n\t "))

    else:
        with open(file_name) as fin:
            for line in fin:
                reviews.append(line.strip("\n\t "))
    reviews = " ".join(reviews)
    return reviews
