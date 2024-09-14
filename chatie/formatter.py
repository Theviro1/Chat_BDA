from typing import List
import jieba
import math
import re
import pint

from chatie.config import *

class Formatter:
    def __init__(self, params:List[tuple]):
        self.params = params
    
    def cut_phrase(self, phrases:List[str])->List[List[str]]:
        # cut phrases to tokens and eliminate those stop words
        docs = []
        for phrase in phrases:
            tokens = list(jieba.cut(phrase))
            tokens = [token for token in tokens if token not in STOP_WORDS]
            docs.append(tokens)
        return docs

    def filter_phrase(self, docs:List[List[str]]):
        # calculate IDF
        record = {}
        tokens = [token for doc in docs for token in doc]  # flatten
        for token in tokens:
            if token not in record.keys():
                record[token] = 1
            else: record[token] += 1
        valid_tokens = []
        # choose half length token to represent a phrase
        for doc in docs:
            valid_token = [(token, record[token]) for token in doc]
            valid_token.sort(key=lambda x:x[1])
            valid_token = [token for token, _ in valid_token[:int(math.ceil(len(valid_token) * FILTER_RATE))]] 
            valid_tokens.append(valid_token)
        # there might have a tiny possibility that other tokens appear in input_text but these filtered one doesn't, try set FILTER_RATE higher
        return valid_tokens
    
    def match_phrase(self, phrases:List[str], valid_tokens:List[str], input_text:str)->List[str]:
        filtered_phrases = []
        for i in range(len(phrases)):
            valid_token = valid_tokens[i]
            if any(token in input_text for token in valid_token): filtered_phrases.append(phrases[i])
        return filtered_phrases
            
    
    def filter(self, input_text:str)->List[str]:
        phrases = [name for name, _, _ in self.params]
        docs = self.cut_phrase(phrases)
        valid_tokens = self.filter_phrase(docs)
        filtered_phrases = self.match_phrase(phrases, valid_tokens, input_text)
        print(f'reduce from {len(self.params)} to {len(filtered_phrases)} words')
        return filtered_phrases
    





    def handle_i(self, extraction:List[tuple])->List[tuple]:
        i_dict = {}
        fixed_extraction = []
        for name, symbol, value, unit, standard_unit in extraction:
            # record all the params that contains 'i'
            if 'i' in name:
                if name not in i_dict.keys(): i_dict[name] = 1
                else: i_dict[name] += 1
        
        for name, symbol, value, unit, standard_unit in extraction:
            if name in i_dict.keys():
                cnt = i_dict[name]
                for i in range(cnt):
                    # symbol should be endswith '_i'
                    symbol_i = symbol[0:-2] + '_' + str(i+1)
                    fixed_extraction.append((name, symbol_i, value, unit, standard_unit))
            else: fixed_extraction.append((name, symbol, value, unit, standard_unit))
        return fixed_extraction

    def handle_value(self, symbol:str, value:str, unit:str, standard_unit:str)->tuple:
        # check value first
        pattern = r'([<>]=?)?(\d+(\.\d+)?)(-)?(\d+)?'  # remove protential unit in value
        match = re.match(pattern, value)
        ineqs_sign, fixed_value, range_sign, range_value = match.group(1), match.group(2), match.group(4), match.group(5)
        tag, result = '', ''
        # ineqs/range/eqs can only exist one
        if range_sign is not None:
            # range expression
            tag = 'ineqs'
            fixed_value = self.parse_unit(fixed_value, unit, standard_unit)
            range_value = self.parse_unit(range_value, unit, standard_unit)
            result = str(fixed_value) + ' <= ' + symbol + ' <= ' + str(range_value)
        elif ineqs_sign is not None:
            # ineqs expression
            tag = 'ineqs'
            fixed_value = self.parse_unit(fixed_value, unit, standard_unit)
            result = symbol + ' ' + ineqs_sign + ' ' + str(fixed_value) 
        else:
            # eqs expression
            tag = 'eqs'
            fixed_value = self.parse_unit(fixed_value, unit, standard_unit)
            result = symbol + ' = ' + str(fixed_value)
        return tag, result
        


    def parse_unit(self, value:str, unit:str, standard_unit:str)->float:
        ureg = pint.UnitRegistry()
        v = value + unit
        try:
            # if unit can be distinguished, try transform
            fixed_v = ureg(v).to(standard_unit).magnitude
        except:
            # if unit doesn't exist in pint, use default(standard) unit
            fixed_v = float(value)
        if standard_unit == '%': fixed_v /= 100
        return fixed_v

    
    def handle(self, extraction:List[tuple])->tuple:
        fixed_extraction = self.handle_i(extraction)
        ineqs = []
        eqs = []
        for name, symbol, value, unit, standard_unit in fixed_extraction:
            tag, string = self.handle_value(symbol, value, unit, standard_unit)
            if tag == 'ineqs': ineqs.append(string)
            else: eqs.append(string)
        return eqs, ineqs


    