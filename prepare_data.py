import os
import json
import gzip
import re
import xml.etree.ElementTree as ET

#macros
pronouns = {'i', 'me', 'myself', 'we', 'us', 'ourselves', 'he', 'him', 'himself', 'she', 'her', 'herself',
            'it', 'they', 'them', 'themselves', 'you'}
nn_tags = {'NN', 'NNP', 'NNS', 'NNPS'}

def 