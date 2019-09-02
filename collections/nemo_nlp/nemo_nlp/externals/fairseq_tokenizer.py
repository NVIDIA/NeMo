import sys
import re
import unicodedata
from collections import defaultdict


def get_unicode_categories():
    cats = defaultdict(list)
    for c in map(chr, range(sys.maxunicode + 1)):
        cats[unicodedata.category(c)].append(c)
    return cats


NUMERICS = ''.join(get_unicode_categories()['No'])


def tokenize_en(line):
    line = line.strip()
    line = ' ' + line + ' '
    # remove ASCII junk
    line = re.sub(r'\s+', ' ', line)
    line = re.sub(r'[\x00-\x1F]', '', line)
    #fix whitespaces
    line = re.sub('\ +', ' ', line)
    line = re.sub('^ ', '', line)
    line = re.sub(' $', '', line)
    #separate other special characters
    line = re.sub(r'([^\s\.\'\`\,\-\w]|[_'+NUMERICS+'])', r' \g<1> ', line)
    line = re.sub(r'(\w)\-(?=\w)', r'\g<1> @-@ ', line)

    #multidots stay together
    line = re.sub(r'\.([\.]+)', r' DOTMULTI\g<1>', line)
    while re.search(r'DOTMULTI\.', line):
        line = re.sub(r'DOTMULTI\.([^\.])', r'DOTDOTMULTI \g<1>', line)
        line = re.sub(r'DOTMULTI\.', r'DOTDOTMULTI', line)

    # separate out "," except if within numbers (5,300)
    line = re.sub(r'([\D])[,]', r'\g<1> , ', line)
    line = re.sub(r'[,]([\D])', r' , \g<1>', line)

    # separate "," after a number if it's the end of sentence
    line = re.sub(r'(\d)[,]$', r'\g<1> ,', line)

    # split contractions right
    line = re.sub(r'([\W\d])[\']([\W\d])', '\g<1> \' \g<2>', line)
    line = re.sub(r'(\W)[\']([\w\D])', '\g<1> \' \g<2>', line)
    line = re.sub(r'([\w\D])[\']([\W\d])', '\g<1> \' \g<2>', line)
    line = re.sub(r'([\w\D])[\']([\w\D])', '\g<1> \'\g<2>', line)
    # special case for "1990's"
    line = re.sub(r'([\W\d])[\']([s])', '\g<1> \'\g<2>', line)

    # apply nonbreaking prefixes
    words = line.split()
    line = ''
    for i in range(len(words)):
        word = words[i]
        match =  re.search(r'^(\S+)\.$', word)
        if match:
            pre = match.group(1)
            if i==len(words)-1:
                # split last words independently as they are unlikely to be non-breaking prefixes
                word = pre+' .'
            # elif ((re.search(r'\.', pre) and re.search(r'[^\.\W\d]', pre))
            #         or (pre in prefixes and prefixes[pre]==1)
            #         or re.search(r'^[a-z]', words[i+1])
            #         or (pre in prefixes and prefixes[pre]==2 and re.search(r'^[0-9]+', words[i+1]))):
            #     pass
            else:
                word = pre+' .'

        word +=' '
        line += word

    # clean up extraneous spaces
    line = re.sub(' +', ' ', line)
    line = re.sub('^ ', '', line)
    line = re.sub(' $', '', line)

    # .' at end of sentence is missed
    line = re.sub(r'\.\' ?$', ' . \' ', line)

    #restore multi-dots
    while re.search('DOTDOTMULTI', line):
        line = re.sub('DOTDOTMULTI', 'DOTMULTI.', line)

    line = re.sub('DOTMULTI', '.', line)

    # escape special characters
    line = re.sub(r'\&', r'&amp;', line)
    line = re.sub(r'\|', r'&#124;', line)
    line = re.sub(r'\<', r'&lt;', line)
    line = re.sub(r'\>', r'&gt;', line)
    line = re.sub(r'\'', r'&apos;', line)
    line = re.sub(r'\"', r'&quot;', line)
    line = re.sub(r'\[', r'&#91;', line)
    line = re.sub(r'\]', r'&#93;', line)

    #ensure final line breaks
    # if line[-1] is not '\n':
    #     line += '\n'

    return line
