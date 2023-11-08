# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Sayma Sultana <sayma@wayne.edu>, Jaydeb Sarker <jaydebsarker@wayne.edu> ,and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import copy
import json
import re
from nltk import  word_tokenize

class BaseTokenizer(object):
    def process_text(self, text):
        raise NotImplemented

    def process(self, texts):
        for text in texts:
            yield self.process_text(text)


def read_lines_from_model(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        return lines


RE_PATTERNS = {

    ' fuck ':
        [
            '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
            ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',
            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',
            'feck ', ' fux ', 'f\*\*',
            'f\-ing', 'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck'

        ],

    ' ass ':
        [
            '[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\$\$'
                                                           '[^a-z]anus', ' a\*s\*s', '[^a-z]ass[^a-z ]',
            'a[@#\$%\^&\*][@#\$%\^&\*]', '[^a-z]anal ', 'a s s'
        ],

    ' ass hole ':
        [
            ' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole'
        ],

     ' bitch ':
        [
            'bitches',  'b!tch', 'bitching', 'bitched',
            'biatch', 'bytch', 'b i t c h'],

    ' kiss ':
        [
            'kissed', 'k([i]+)*ss'
        ],
    ' boob ':
        [
            '^boob ', '^boobs ', '^boobies ', '^b([o]+)b ',
            ' boob ', ' boobs ', ' boobies ', ' b([o]+)b '
        ],

    ' bastard ':
        [
            'ba[s|z]+t[e|a]+rd'
        ],

    ' lesbian ':
    [
        ' lesbo ', ' lez ', ' lezzy '
    ],

    ' gay ':
        [
            '^gay ',' gay ', ' g([a]+)y '
        ],

    ' cock ':
        [
            '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
            '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'
        ],

    ' dick ':
        [
            ' dick[^aeiou]', 'd i c k'
        ],

    ' suck ':
        [
            'sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'
        ],

    ' cunt ':
        [
            'cunt', 'c u n t'
        ],

    ' jerk ':
        [
            'jerk'
        ],

     ' rape ':
        [
            'raped'
        ],

       ' sex ':
        [
            'sexy', 's3x', 'sexuality'
        ],

    ' shut the fuck up':
        [
            ' stfu' '^stfu'
        ],

    ' for your fucking information':
        [
            ' fyfi', '^fyfi'
        ],
    ' get the fuck off':
        [
            'gtfo', '^gtfo'
        ],

    ' oh my fucking god ':
        [
            ' omfg ', '^omfg'
        ],

    ' what the hell ':
        [
            ' wth ', '^wth'
        ],

    ' what the fuck ':
        [
            ' wtf ', '^wtf'
        ],
    ' son of bitch ':
        [
            ' sob ', '^sob '
        ],

    ' pussy ':
        [
            'pussy[^c]', 'pusy', 'pussi[^l]', 'pusses', '(p)(u|[^a-z0-9 ])(s|[^a-z0-9 ])(s|[^a-z0-9 ])(y)',
        ],

    ' faggot ':
        [
            'faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
            '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot',
        ],

    ' mother fucker':
        [
            ' motha fuc', ' mother fuck', 'motherfucker', ' mofo',
        ],

    ' whore ':
        [
            'wh\*\*\*', 'w h o r e'
        ],

    # ' what the fuck ':
    # [
    #     ' wtf',
    # ],
}


class IdentifierTokenizer(BaseTokenizer):
    def __init__(self):

        self.programming_keywords_list = read_lines_from_model('models/programming_keywords.txt')

    def split_identifiers(self, text):
        result = re.sub('[_]+', ' ', text) # replace underscores with space
        result=re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', result))
        return result

    def remove_keywords(self, text):
        words = text.split()
        resultwords = [word for word in words if word.lower() not in self.programming_keywords_list]
        result = ' '.join(resultwords)
        return result



class PatternTokenizer(BaseTokenizer):
    def __init__(self, lower=True, initial_filters=r"[^a-z0-9!@#\$%\^\*\+\?\&\_\-,\.' ]", patterns=RE_PATTERNS,
                 remove_repetitions=True):
        self.lower = lower
        self.patterns = patterns
        self.initial_filters = initial_filters
        self.remove_repetitions = remove_repetitions
        self.word_categories =self.read_word_categories("models/keyword-categories.json")

        self.pejoratives = self.word_categories["pejoratives"]
        self.appearance_reference = self.word_categories["appearance"]
        self.cloth_reference = self.word_categories["women_cloth"]
        self.women_kins =self.word_categories["women_kins"]
        self.women_roles = self.word_categories["women_roles"]
        self.body_parts = self.word_categories["body_parts"]
        self.lgbtq = self.word_categories["lgbtq"]

    def process_text(self, text):
        x = self._preprocess(text)
        for target, patterns in self.patterns.items():
            for pat in patterns:
                x = re.sub(pat, target, x)
        x = re.sub(r"[^a-z' ]", ' ', x)
        return x

    def read_word_categories(self, word_category_file):
        with open(word_category_file) as jsonfile:
            json_list =json.load(jsonfile)
            return json_list

    def replace_emojis(self, text):
        text =re.sub(r':\w+:', 'emoji',text)
        return text

    def process_ds(self, ds):
        ### ds = Data series

        # lower
        ds = copy.deepcopy(ds)
        if self.lower:
            ds = ds.str.lower()

        # replace emojis
        # remove special chars
        if self.initial_filters is not None:
            ds = ds.str.replace(self.initial_filters, ' ')

        # looooooooooser = loser
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
            ds = ds.str.replace(pattern, r"\1")

        for target, patterns in self.patterns.items():
            for pat in patterns:
                ds = ds.str.replace(pat, target)

        ds = ds.str.replace(r"[^a-z' ]", ' ')

        return ds.str.split()

    def count_word_from_list(self, text, wordlist):
        count=0
        words = word_tokenize(text)
        for word in wordlist:
            if word in words:
                # print(profane_word)
                count = count + 1
        return count

    def count_pejoratives(self, text):
        return self.count_word_from_list(text, self.pejoratives)

    def count_appearance_reference(self, text):
        return self.count_word_from_list(text, self.appearance_reference)

    def count_women_roles(self, text):
        return self.count_word_from_list(text, self.women_roles)

    def count_women_kins_reference(self, text):
        return self.count_word_from_list(text, self.women_kins)

    def count_lgbtq_reference(self, text):
        return self.count_word_from_list(text, self.lgbtq)
    def count_women_body_parts(self, text):
        return self.count_word_from_list(text, self.body_parts)

    def count_women_clothes(self, text):
        return self.count_word_from_list(text, self.cloth_reference)

    def _preprocess(self, text):
        # lower
        if self.lower:
            text = text.lower()

        text =self.replace_emojis(text)
        # remove special chars
        if self.initial_filters is not None:
            text = re.sub(self.initial_filters, ' ', text)

        # neeeeeeeeeerd => nerd
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
            text = pattern.sub(r"\1", text)
        return text
