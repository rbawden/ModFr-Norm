#!/usr/bin/python
from transformers import Pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from torch import Tensor
import html.parser
import unicodedata
import sys, os
import re
import pickle
from tqdm.auto import tqdm
import operator
from datasets import load_dataset
from transformers.pipelines import PIPELINE_REGISTRY

def _create_modified_versions(entry=None):
    if entry is None:
        return []
    return _remove_diacritics(entry), _vu_vowel_to_v_vowel(entry), _vowel_u_to_vowel_v(entry), _consonant_v_to_consonant_u(entry), _y_to_i(entry), _i_to_y(entry), _eacute_to_e_s(entry), _final_eacute_to_e_z(entry), _egrave_to_eacute(entry), _vowelcircumflex_to_vowel_s(entry), _ce_to_ee(entry)

def _create_further_modified_versions(entry=None):
    if entry is None:
        return []
    return _s_to_f(entry), _ss_to_ff(entry), _s_to_ff(entry), _first_s_to_f(entry), _first_s_to_ff(entry), _last_s_to_f(entry), _last_s_to_ff(entry), _sit_to_st(entry), _ee_to_ce(entry), _z_to_s(entry)

def _remove_diacritics(s, allow_alter_length=True):
    # 1-1 replacements only (must not change the number of characters
    replace_from = "ǽǣáàâäąãăåćčçďéèêëęěğìíîĩĭıïĺľłńñňòóôõöøŕřśšşťţùúûũüǔỳýŷÿźẑżžÁÀÂÄĄÃĂÅĆČÇĎÉÈÊËĘĚĞÌÍÎĨĬİÏĹĽŁŃÑŇÒÓÔÕÖØŔŘŚŠŞŤŢÙÚÛŨÜǓỲÝŶŸŹẐŻŽſ"
    replace_into = "ææaaaaaaaacccdeeeeeegiiiiiiilllnnnoooooorrsssttuuuuuuyyyyzzzzAAAAAAAACCCDEEEEEEGIIIIIIILLLNNNOOOOOORRSSSTTUUUUUUYYYYZZZZs"
    table = s.maketrans(replace_from, replace_into)
    s = s.translate(table)
    # n-m replacemenets
    if allow_alter_length:
        for before, after in [('œ', 'oe'), ('æ', 'ae'), ('ƣ', 'oi'), ('ĳ', 'ij'),
                              ('ȣ', 'ou'), ('Œ', 'OE'), ('Æ', 'AE'), ('Ƣ', 'OI'), ('Ĳ', 'IJ'), ('Ȣ', 'OU')]:
            s = s.replace(before, after)
        s = s.strip('-')
    return s

def _vu_vowel_to_v_vowel(s):
    s = re.sub('v([aeiou])' , r'vu\1', s)
    return s
    
def _vowel_u_to_vowel_v(s):
    s = re.sub('([aeiou])u' , r'\1v', s)
    return s
    
def _consonant_v_to_consonant_u(s):
    s = re.sub('([^aeiou])v' , r'\1u', s)
    return s
    
def _y_to_i(s):
    s = s.replace('y', 'i')
    return s

def _i_to_y(s):
    s = s.replace('i', 'y')
    return s

def _ss_to_ff(s):
    s = s.replace('ss', 'ff')
    return s

def _s_to_f(s):
    s = s.replace('s', 'f')
    return s

def _s_to_ff(s):
    s = s.replace('s', 'ff')
    return s
    
def _first_s_to_f(s):
    s = re.sub('s' , r'f', s, 1)
    return s

def _last_s_to_f(s):
    s = re.sub('^(.*)s' , r'\1f', s)
    return s
    
def _first_s_to_ff(s):
    s = re.sub('s' , r'ff', s, 1)
    return s
    
def _last_s_to_ff(s):
    s = re.sub('^(.*)s' , r'\1ff', s)
    return s
    
def _ee_to_ce(s):
    s = s.replace('ee', 'ce')
    return s

def _sit_to_st(s):
    s = s.replace('sit', 'st')
    return s

def _z_to_s(s):
    s = s.replace('z', 's')
    return s

def _ce_to_ee(s):
    s = s.replace('ce', 'ee')
    return s

def _eacute_to_e_s(s, allow_alter_length=True):
    if allow_alter_length:
        s = re.sub('é(.)' , r'es\1', s)
        s = re.sub('ê(.)' , r'es\1', s)
    return s
        
def _final_eacute_to_e_z(s, allow_alter_length=True):
    if allow_alter_length:
        s = re.sub('é$' , r'ez', s)
        s = re.sub('ê$' , r'ez', s)
    return s
        
def _egrave_to_eacute(s):
    s = re.sub('è(.)' , r'é\1', s)
    return s

def _vowelcircumflex_to_vowel_s(s, allow_alter_length=True):
    if allow_alter_length:
        for before, after in [('â', 'as'), ('ê', 'es'), ('î', 'is'), ('ô', 'os'), ('û', 'us')]:
            s = s.replace(before, after)
    return s


def basic_tokenise(string):
    # separate punctuation
    for char in r',.;?!:)("…-':
        string = re.sub('(?<! )' + re.escape(char) + '+', ' ' + char, string)
    for char in '\'"’':
        string = re.sub(char + '(?! )' , char + ' ', string)
    return string.strip()

def basic_tokenise_bs(string):
    # separate punctuation
    string = re.sub('(?<! )([,\.;\?!:\)\("…\'‘’”“«»\-])', r' \1', string)
    string = re.sub('([,\.;\?!:\)\("…\'‘’”“«»\-])(?! )' , r'\1 ', string)
    return string.strip()

def homogenise(sent, allow_alter_length=False):
    '''
    Homogenise an input sentence by lowercasing, removing diacritics, etc.
    If allow_alter_length is False, then only applies changes that do not alter
    the length of the original sentence (i.e. one-to-one modifications). If True,
    then also apply n-m replacements.
    '''
    sent = sent.lower()
    # n-m replacemenets
    if allow_alter_length:
        for before, after in [('ã', 'an'), ('xoe', 'œ')]:
            sent = sent.replace(before, after)
        sent = sent.strip('-')
    # 1-1 replacements only (must not change the number of characters
    replace_from = "ǽǣáàâäąãăåćčçďéèêëęěğìíîĩĭıïĺľłńñňòóôõöøŕřśšşťţùúûũüǔỳýŷÿźẑżžÁÀÂÄĄÃĂÅĆČÇĎÉÈÊËĘĚĞÌÍÎĨĬİÏĹĽŁŃÑŇÒÓÔÕÖØŔŘŚŠŞŤŢÙÚÛŨÜǓỲÝŶŸŹẐŻŽſ"
    replace_into = "ææaaaaaaaacccdeeeeeegiiiiiiilllnnnoooooorrsssttuuuuuuyyyyzzzzAAAAAAAACCCDEEEEEEGIIIIIIILLLNNNOOOOOORRSSSTTUUUUUUYYYYZZZZs"
    table = sent.maketrans(replace_from, replace_into)
    return sent.translate(table)

def get_surrounding_punct(word):
    beginning_match = re.match("^(['\-]*)", word)
    beginning, end = '', ''
    if beginning_match:
        beginning = beginning_match.group(1)
    end_match = re.match("(['\-]*)$", word)
    if end_match:
        end = end_match.group(1)
    return beginning, end


def add_orig_punct(old_word, new_word):
    beginning, end = get_surrounding_punct(old_word)
    output = ''
    if beginning != None and not re.match("^"+re.escape(beginning), new_word):
        output += beginning
    if new_word != None:
        output += new_word
    if end != None and not re.match(re.escape(end)+"$", new_word):
        output += end
    return output
    
def get_caps(word):
    # remove any non-alphatic characters at begining or end
    word = word.strip("-' ")
    first, second, allcaps = False, False, False
    if len(word) > 0 and word[0].lower() != word[0]:
        first = True
    if len(word) > 1 and word[1].lower() != word[1]:
        second = True
    if word.upper() == word and word.lower() != word:
        allcaps = True
    return first, second, allcaps

def set_caps(word, first, second, allcaps):
    if word == None:
        return None
    if allcaps:
        return word.upper()
    elif first and second:
        return word[0].upper() + word[1].upper() + word[2:]
    elif first:
        if len(word) > 1:
            return word[0].upper() + word[1:]
        elif len(word) == 1:
            return word[0]
        else:
            return word
    elif second:
        if len(word) > 2:
            return word[0] + word[1].upper() + word[2:]
        elif len(word) > 1:
            return word[0] + word[1].upper() + word[2:]
        elif len(word) == 1:
            return word[0]
        else:
            return word
    else:
        return word


######## Edit distance functions #######
def _wedit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev


def _wedit_dist_step(
    lev, i, j, s1, s2, last_left, last_right, transpositions=False
):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + _wedit_dist_deletion_cost(c1,c2)
    # skipping a character in s2
    b = lev[i][j - 1] + _wedit_dist_insertion_cost(c1,c2)
    # substitution
    c = lev[i - 1][j - 1] + (_wedit_dist_substitution_cost(c1, c2) if c1 != c2 else 0)

    # pick the cheapest
    lev[i][j] = min(a, b, c)#, d)

def _wedit_dist_backtrace(lev):
    i, j = len(lev) - 1, len(lev[0]) - 1
    alignment = [(i, j, lev[i][j])]

    while (i, j) != (0, 0):
        directions = [
            (i - 1, j),  # skip s1
            (i, j - 1),  # skip s2
            (i - 1, j - 1),  # substitution
        ]

        direction_costs = (
            (lev[i][j] if (i >= 0 and j >= 0) else float("inf"), (i, j))
            for i, j in directions
        )
        _, (i, j) = min(direction_costs, key=operator.itemgetter(0))

        alignment.append((i, j, lev[i][j]))
    return list(reversed(alignment))

def _wedit_dist_substitution_cost(c1, c2):
    if c1 == ' ' and c2 != ' ':
        return 1000000
    if c2 == ' ' and c1 != ' ':
        return 30
    for c in ",.;-!?'":
        if c1 == c and c2 != c:
            return 20
        if c2 == c and c1 != c:
            return 20
    return 1

def _wedit_dist_deletion_cost(c1, c2):
    if c1 == ' ':
        return 2
    if c2 == ' ':
        return 1000000
    return 0.8

def _wedit_dist_insertion_cost(c1, c2):
    if c1 == ' ':
        return 1000000
    if c2 == ' ':
        return 2
    return 0.8

def wedit_distance_align(s1, s2):
    """
    Calculate the minimum Levenshtein weighted edit-distance based alignment
    mapping between two strings. The alignment finds the mapping
    from string s1 to s2 that minimizes the edit distance cost, where each
    operation is weighted by a dedicated weighting function.
    For example, mapping "rain" to "shine" would involve 2
    substitutions, 2 matches and an insertion resulting in
    the following mapping:
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)]
    NB: (0, 0) is the start state without any letters associated
    See more: https://web.stanford.edu/class/cs124/lec/med.pdf
    In case of multiple valid minimum-distance alignments, the
    backtrace has the following operation precedence:
    1. Skip s1 character
    2. Skip s2 character
    3. Substitute s1 and s2 characters
    The backtrace is carried out in reverse string order.
    This function does not support transposition.
    :param s1, s2: The strings to be aligned
    :type s1: str
    :type s2: str
    :rtype: List[Tuple(int, int)]
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _wedit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _wedit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                0,
                0,
                transpositions=False,
            )

    # backtrace to find alignment
    alignment = _wedit_dist_backtrace(lev)
    return alignment

def _last_left_t_init(sigma):
    return {c: 0 for c in sigma}

def wedit_distance(s1, s2):
    """
    Calculate the Levenshtein weighted edit-distance between two strings.
    The weighted edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2, weighted 
    by a dedicated weighting function.
    For example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to
    substitutions.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :type transpositions: bool
    :rtype: int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _wedit_dist_init(len1 + 1, len2 + 1)

    # retrieve alphabet
    sigma = set()
    sigma.update(s1)
    sigma.update(s2)

    # set up table to remember positions of last seen occurrence in s1
    last_left_t = _last_left_t_init(sigma)

    # iterate over the array
    # i and j start from 1 and not 0 to stay close to the wikipedia pseudo-code
    # see https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    for i in range(len1):
        last_right_buf = 0
        for j in range(len2):
            last_left = last_left_t[s2[j - 1]]
            last_right = last_right_buf
            if s1[i - 1] == s2[j - 1]:
                last_right_buf = j
            _wedit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                last_left,
                last_right,
                transpositions=False,
            )
        last_left_t[s1[i - 1]] = i
    return lev[len1-1][len2-1]

def space_after(idx, sent):
    if idx < len(sent) -1 and sent[idx + 1] == ' ':
        return True
    return False

def space_before(idx, sent):
    if idx > 0 and sent[idx - 1] == ' ':
        return True
    return False

######## Normaliation pipeline #########
class NormalisationPipeline(Pipeline):

    def __init__(self, beam_size=5, batch_size=32, tokenise_func=None, cache_file=None, no_postproc_lex=False, 
                 no_post_clean=False, **kwargs):
        self.beam_size = beam_size
        # classic tokeniser function (used for alignments)
        if tokenise_func is not None:
            self.classic_tokenise = tokenise_func
        else:
            self.classic_tokenise = basic_tokenise

        self.no_post_clean = no_post_clean
        self.no_postproc_lex = no_postproc_lex
        # load lexicon
        if no_postproc_lex:
            self.orig_lefff_words, self.mapping_to_lefff, self.mapping_to_lefff2 = None, None, None
        else:
            self.orig_lefff_words, self.mapping_to_lefff, self.mapping_to_lefff2 = self.load_lexicon(cache_file=cache_file)
        super().__init__(**kwargs)


    def load_lexicon(self, cache_file=None):
        orig_lefff_words = []
        mapping_to_lefff = {}
        mapping_to_lefff2 = {}
        remove = set([])
        remove2 = set([])

        # load pickled version if there
        if cache_file is not None and os.path.exists(cache_file):
            return pickle.load(open(cache_file, 'rb'))
        dataset = load_dataset("sagot/lefff_morpho")

        for entry in set([x['form'].lower() for x in dataset['test']]):
            orig_lefff_words.append(entry)
            orig_lefff_words.append("-"+entry)
            for mod_entry in set(_create_modified_versions(entry)):
                if mod_entry in mapping_to_lefff and mapping_to_lefff[mod_entry] != entry:
                    remove.add(mod_entry)
                    if mod_entry != mod_entry.upper():
                        remove.add(mod_entry)
                if mod_entry not in mapping_to_lefff and mod_entry != entry:
                    mapping_to_lefff[mod_entry] = entry
                    if mod_entry != mod_entry.upper():
                        mapping_to_lefff2[mod_entry.upper()] = entry.upper()
                for mod_entry2 in set(_create_modified_versions(mod_entry)):
                    if mod_entry2 in mapping_to_lefff2 and mapping_to_lefff2[mod_entry2] != entry:
                        remove2.add(mod_entry2)
                        if mod_entry2 != mod_entry2.upper():
                            remove2.add(mod_entry2)
                    if mod_entry2 not in mapping_to_lefff2 and mod_entry2 != entry:
                        mapping_to_lefff2[mod_entry2] = entry
                        if mod_entry2 != mod_entry2.upper():
                            mapping_to_lefff2[mod_entry2.upper()] = entry.upper()
                for mod_entry2 in set(_create_further_modified_versions(mod_entry)):
                    if mod_entry2 in mapping_to_lefff2 and mapping_to_lefff2[mod_entry2] != entry:
                        remove2.add(mod_entry2)
                        if mod_entry2 != mod_entry2.upper():
                            remove2.add(mod_entry2)
                    if mod_entry2 not in mapping_to_lefff2 and mod_entry2 != entry:
                        mapping_to_lefff2[mod_entry2] = entry
                        if mod_entry2 != mod_entry2.upper():
                            mapping_to_lefff2[mod_entry2.upper()] = entry.upper()
            for mod_entry2 in set(_create_further_modified_versions(entry)):
                if mod_entry2 in mapping_to_lefff2 and mapping_to_lefff2[mod_entry2] != entry:
                    remove2.add(mod_entry2)
                    if mod_entry2 != mod_entry2.upper():
                        remove2.add(mod_entry2)
                if mod_entry2 not in mapping_to_lefff2 and mod_entry2 != entry:
                    mapping_to_lefff2[mod_entry2] = entry
                    if mod_entry2 != mod_entry2.upper():
                        mapping_to_lefff2[mod_entry2.upper()] = entry.upper()
                    
        for mod_entry in list(mapping_to_lefff.keys()):
            if mod_entry != "":
                mapping_to_lefff["-"+mod_entry] = "-"+mapping_to_lefff[mod_entry]
        for mod_entry2 in list(mapping_to_lefff2.keys()):
            if mod_entry2 != "":
                mapping_to_lefff2["-"+mod_entry2] = "-"+mapping_to_lefff2[mod_entry2]

        for entry in remove:
            del mapping_to_lefff[entry]
        for entry in remove2:
            del mapping_to_lefff2[entry]

        if cache_file is not None:
            pickle.dump((orig_lefff_words, mapping_to_lefff, mapping_to_lefff2), open(cache_file, 'wb'))
        return orig_lefff_words, mapping_to_lefff, mapping_to_lefff2

    def _sanitize_parameters(self, clean_up_tokenisation_spaces=None, truncation=None, **generate_kwargs):
        preprocess_params = {}
        if truncation is not None:
            preprocess_params["truncation"] = truncation
        forward_params = generate_kwargs
        postprocess_params = {}
        if clean_up_tokenisation_spaces is not None:
            postprocess_params["clean_up_tokenisation_spaces"] = clean_up_tokenisation_spaces

        return preprocess_params, forward_params, postprocess_params


    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        return True

    def make_printable(self, s):
        '''Replace non-printable characters in a string.'''
        return s.translate(NOPRINT_TRANS_TABLE)


    def normalise(self, line):
        for before, after in [('[«»\“\”]', '"'), ('[‘’]', "'"), (' +', ' '), ('\"+', '"'),
                              ("'+", "'"), ('^ *', ''), (' *$', '')]:
            line = re.sub(before, after, line)
        return line.strip() + ' </s>'
    
    def _parse_and_tokenise(self, *args, truncation):
        prefix = ""
        if isinstance(args[0], list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokeniser has a pad_token_id when using a batch input")
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )
        inputs = [self.normalise(x) for x in args]
        inputs = self.tokenizer(inputs, padding=padding, truncation=truncation, return_tensors=self.framework)
        toks = []
        for tok_ids in inputs.input_ids:
            toks.append(" ".join(self.tokenizer.convert_ids_to_tokens(tok_ids)))
        # This is produced by tokenisers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs
    
    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
        inputs = self._parse_and_tokenise(inputs, truncation=truncation, **kwargs)
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        in_b, input_length = model_inputs["input_ids"].shape
        generate_kwargs["min_length"] = generate_kwargs.get("min_length", self.model.config.min_length)
        generate_kwargs["max_length"] = generate_kwargs.get("max_length", self.model.config.max_length)
        generate_kwargs['num_beams'] = self.beam_size
        self.check_inputs(input_length, generate_kwargs["min_length"], generate_kwargs["max_length"])
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        out_b = output_ids.shape[0]
        output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, clean_up_tok_spaces=False):
        records = []
        for output_ids in model_outputs["output_ids"][0]:
            record = {"text": self.tokenizer.decode(output_ids, skip_special_tokens=True,
                                                    clean_up_tokenisation_spaces=clean_up_tok_spaces).strip()}
            records.append(record)
        return records

    def postprocess_correct_sent(self, alignment):
        output = []
        for i, (orig_word, pred_word, _) in enumerate(alignment):
            if orig_word != '':
                postproc_word = self.postprocess_correct_word(orig_word, pred_word, alignment)
                alignment[i] = (orig_word, postproc_word, -1) # replace prediction in the alignment
        return alignment

    def postprocess_correct_word(self, orig_word, pred_word, alignment):
        # pred_word exists in lexicon, take it
        orig_caps = get_caps(orig_word)
        if re.match("^[0-9]+$", orig_word) or re.match("^[XVUI]+$", orig_word):
            orig_word = orig_word.replace('U', 'V')
            return orig_word
        if pred_word.lower() in self.orig_lefff_words:
            return set_caps(pred_word, *orig_caps)
        # otherwise, if original word exists, take that
        if orig_word.lower() in self.orig_lefff_words:
            return orig_word

        pred_replacement = None
        # otherwise if pred word is in the lexicon with some changes, take that
        if pred_word != '' and pred_word != ' ':
            pred_replacement = self.mapping_to_lefff.get(pred_word, None)
        if pred_replacement is not None:
            return add_orig_punct(pred_word, set_caps(pred_replacement, *orig_caps))
        # otherwise if orig word is in the lexicon with some changes, take that
        orig_replacement = self.mapping_to_lefff.get(orig_word, None)
        if orig_replacement is not None:
            return add_orig_punct(pred_word, set_caps(orig_replacement, *orig_caps))

        # otherwise if pred word is in the lexicon with more changes, take that
        if pred_word != '' and pred_word != ' ':
            pred_replacement = self.mapping_to_lefff2.get(pred_word, None)
        if pred_replacement is not None:
            return add_orig_punct(pred_word, set_caps(pred_replacement, *orig_caps))
        # otherwise if orig word is in the lexicon with more changes, take that
        orig_replacement = self.mapping_to_lefff2.get(orig_word, None)
        if orig_replacement is not None:
            return add_orig_punct(pred_word, set_caps(orig_replacement, *orig_caps))

        if orig_word == pred_word:
            return orig_word
        if orig_word == " " and pred_word == "":
            return orig_word

        wed = wedit_distance(pred_word,orig_word)
        if wed > 2:
            return orig_word
        return add_orig_punct(pred_word, set_caps(pred_word, *orig_caps))


    def __call__(self, input_sents, **kwargs):
        r"""
        Generate the output texts using texts given as inputs.
        Args:
            args (`List[str]`):
                Input text for the encoder.
            apply_postprocessing (`Bool`):
                Apply postprocessing using the lexicon
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).
        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:
            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
        """
        result = super().__call__(input_sents, **kwargs)
            
        output = []
        for i in range(len(result)):
            input_sent, pred_sent = input_sents[i].strip(), result[i][0]['text'].strip()
            input_sent = input_sent.replace('ſ' , 's')
            if not self.no_post_clean:
                pred_sent = self.post_cleaning(pred_sent)
            alignment, pred_sent_tok = self.align(input_sent, pred_sent)

            if not self.no_postproc_lex:
                alignment = self.postprocess_correct_sent(alignment)
            pred_sent = self.get_pred_from_alignment(alignment)
            if not self.no_post_clean:
                pred_sent = self.post_cleaning(pred_sent)
            char_spans = self.get_char_idx_align(input_sent, pred_sent, alignment)
            output.append({'text': pred_sent, 'alignment': char_spans})
        return output

    def post_cleaning(self, s):
        s = s.replace(' ' , '')
        s = s.replace('ſ' , 's')
        s = s.replace('ß' , 'ss')
        s = s.replace('&' , 'et')
        s = re.sub('ẽ([mbp])' , r'em\1', s)
        s = s.replace('ẽ' , 'en')
        s = re.sub('ã([mbp])' , r'am\1', s)
        s = s.replace('ã' , 'an')
        s = re.sub('õ([mbp])' , r'om\1', s)
        s = s.replace('õ' , 'on')
        s = re.sub('ũ([mbp])' , r'um\1', s)
        s = s.replace('ũ' , 'un')
        return s

    def align(self, sent_ref, sent_pred):
        sent_ref_tok = self.classic_tokenise(re.sub('[  ]', '  ', sent_ref))
        sent_pred_tok = self.classic_tokenise(re.sub('[  ]', '  ', sent_pred))
        backpointers = wedit_distance_align(homogenise(sent_ref_tok), homogenise(sent_pred_tok))
        alignment, current_word, seen1, seen2, last_weight = [], ['', ''], [], [], 0
        for i_ref, i_pred, weight in backpointers:
            if i_ref == 0 and i_pred == 0:
                continue
            # next characters are both spaces -> add current word straight away
            if i_ref <= len(sent_ref_tok) and sent_ref_tok[i_ref-1] == ' ' \
                and i_pred <= len(sent_pred_tok) and sent_pred_tok[i_pred-1] == ' ' \
                and i_ref not in seen1 and i_pred not in seen2:

                # if current word is empty -> insert a space on both sides
                if current_word[0] == '' and current_word[1] == '':
                    alignment.append((' ', ' ', weight-last_weight))
                # else add the current word to both sides
                else:
                    alignment.append((current_word[0], current_word[1], weight-last_weight))
                last_weight = weight
                current_word = ['', '']
                seen1.append(i_ref)
                seen2.append(i_pred)
            # if space in ref and dash in pred
            elif i_ref <= len(sent_ref_tok) and sent_ref_tok[i_ref-1] == ' ' \
                and i_pred <= len(sent_pred_tok) and sent_pred_tok[i_pred-1] == '-' \
                and i_ref not in seen1 and i_pred not in seen2 \
                and current_word[0] == '' and current_word[1] == '':
                alignment.append((' ', '', weight-last_weight))
                last_weight = weight
                current_word = ['', '-']
                seen1.append(i_ref)
                seen2.append(i_pred)
            else:
                end_space = '' #'░'
                # add new character to ref
                if i_ref <= len(sent_ref_tok) and i_ref not in seen1:
                    if i_ref > 0:
                        current_word[0] += sent_ref_tok[i_ref-1]
                        seen1.append(i_ref)
                # add new character to pred
                if i_pred <= len(sent_pred_tok) and i_pred not in seen2:
                    if i_pred > 0:
                        current_word[1] += sent_pred_tok[i_pred-1] if sent_pred_tok[i_pred-1] != ' ' else ' ' #'▁'
                        end_space = '' if space_after(i_pred, sent_pred_tok) else ''# '░'
                        seen2.append(i_pred)
                if i_ref <= len(sent_ref_tok) and sent_ref_tok[i_ref-1] == ' ' and current_word[0].strip() != '':
                    alignment.append((current_word[0].strip(), current_word[1].strip() + end_space, weight-last_weight))
                    last_weight = weight
                    current_word = ['', '']
                # space in ref but aligned to nothing in pred (under-translation)
                elif i_ref <= len(sent_ref_tok) and sent_ref_tok[i_ref-1] == ' ' and current_word[1].strip() == '':
                    alignment.append((current_word[0], current_word[1], weight-last_weight))
                    last_weight = weight
                    current_word = ['', '']
                    seen1.append(i_ref)
                    seen2.append(i_pred)
        # final word
        alignment.append((current_word[0].strip(), current_word[1].strip(), weight-last_weight))
        # check that both strings are entirely covered
        recovered1 = re.sub(' +', ' ', ' '.join([x[0] for x in alignment]))
        recovered2 = re.sub(' +', ' ', ' '.join([x[1] for x in alignment]))

        assert re.sub('[  ]+', ' ', recovered1) == re.sub('[  ]+', ' ', sent_ref_tok), \
            '\n1: *' + re.sub('[  ]+', ' ', recovered1) + "*\n1: *" + re.sub('[  ]+', ' ', sent_ref_tok) + '*'
        assert re.sub('[░▁ ]+', '', recovered2) == re.sub('[▁ ]+', '', sent_pred_tok), \
            '\n2: ' + re.sub('[  ]+', ' ', recovered2) + "\n2: " + re.sub('[  ]+', ' ', sent_pred_tok)
        return alignment, sent_pred_tok

    def get_pred_from_alignment(self, alignment):
         return re.sub(' +', ' ', ''.join([x[1] if x[1] != '' else '\n' for x in alignment]).replace('\n', ''))
    
    def get_char_idx_align(self, sent_ref, sent_pred, alignment):
        covered_ref, covered_pred = 0, 0
        ref_chars = [i for i, character in enumerate(sent_ref)] + [len(sent_ref)]  #
        pred_chars = [i for i, character in enumerate(sent_pred)] + [len(sent_pred)]# if character not in [' ']]
        align_idx = []

        for a_ref, a_pred, _ in alignment:
            if a_ref == '' and a_pred == '':
                covered_pred += 1
                continue
            a_pred = re.sub(' +', ' ', a_pred).strip()
            span_ref = [ref_chars[covered_ref], ref_chars[covered_ref + len(a_ref)]]
            covered_ref += len(a_ref)
            span_pred = [pred_chars[covered_pred], pred_chars[covered_pred + len(a_pred)]]
            covered_pred += len(a_pred)
            align_idx.append((span_ref, span_pred))

        return align_idx
   
def normalise_text(list_sents, batch_size=32, beam_size=5, cache_file=None, no_postproc_lex=False, no_post_clean=False):
    tokeniser = AutoTokenizer.from_pretrained("rbawden/modern_french_normalisation")
    model = AutoModelForSeq2SeqLM.from_pretrained("rbawden/modern_french_normalisation")
    normalisation_pipeline = NormalisationPipeline(model=model,
                                                   tokenizer=tokeniser,
                                                   batch_size=batch_size,
                                                   beam_size=beam_size,
                                                   cache_file=cache_file,
                                                   no_postproc_lex=no_postproc_lex,
                                                   no_post_clean=no_post_clean)
    normalised_outputs = normalisation_pipeline(list_sents)
    return normalised_outputs

def normalise_from_stdin(batch_size=32, beam_size=5, cache_file=None, no_postproc_lex=False, no_post_clean=False):
    tokeniser = AutoTokenizer.from_pretrained("rbawden/modern_french_normalisation")
    model = AutoModelForSeq2SeqLM.from_pretrained("rbawden/modern_french_normalisation")
    normalisation_pipeline = NormalisationPipeline(model=model,
                                                   tokenizer=tokeniser,
                                                   batch_size=batch_size,
                                                   beam_size=beam_size,
                                                   cache_file=cache_file,
                                                   no_postproc_lex=no_postproc_lex,
                                                   no_post_clean=no_post_clean
                                                   )
    list_sents = []
    ex = ["7. Qu'vne force plus grande de ſi peu que l'on voudra, que celle auec laquelle l'eau de la hauteur de trente & vn pieds, tend à couler en bas, ſuffit pour faire admettre ce vuide apparent, & meſme ſi grãd que l'on voudra, c'eſt à dire, pour faire des-vnir les corps d'vn ſi grand interualle que l'on voudra, pourueu qu'il n'y ait point d'autre obſtacle à leur ſeparation ny à leur eſloignement, que l'horreur que la Nature a pour ce vuide apparent."]
    for sent in sys.stdin:
        list_sents.append(sent.strip())
    normalised_outputs = normalisation_pipeline(list_sents)
    for s, sent in enumerate(normalised_outputs):
        alignment=sent['alignment']

        print(sent['text'])
        # checking that the alignment makes sense
        #for b, a in alignment:
        #    print('input: ' + ''.join([list_sents[s][x] for x in range(b[0], max(len(b), b[1]))]) + '')
        #    print('pred: ' + ''.join([sent['text'][x] for x in range(a[0], max(len(a), a[1]))]) + '')

    return normalised_outputs


PIPELINE_REGISTRY.register_pipeline(
    "modern-french-normalisation",
    pipeline_class=NormalisationPipeline,
    pt_model=AutoModelForSeq2SeqLM,
    default={"pt": ("rbawden/modern_french_normalisation", "main")},
    type="text",
)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--batch_size', type=int, default=32, help='Set the batch size for decoding')
    parser.add_argument('-b', '--beam_size', type=int, default=5, help='Set the beam size for decoding')
    parser.add_argument('-i', '--input_file', type=str, default=None, help='Input file. If None, read from STDIN')
    parser.add_argument('-c', '--cache_lexicon', type=str, default=None, help='Path to cache the lexicon file to speed up loading')
    parser.add_argument('-n', '--no_postproc_lex', default=False, action='store_true', help='Deactivate postprocessing to speed up normalisation, but this may degrade the output')
    parser.add_argument('-m', '--no_post_clean', default=False, action='store_true', help='Deactivate postprocessing to speed up normalisation, but this may degrade the output')
        
    args = parser.parse_args()

    if args.input_file is None:
         normalise_from_stdin(batch_size=args.batch_size,
                              beam_size=args.beam_size,
                              cache_file=args.cache_lexicon,
                              no_postproc_lex=args.no_postproc_lex,
                              no_post_clean=args.no_post_clean)
    else:
         list_sents = []
         with open(args.input_file) as fp:
              for line in fp:
                   list_sents.append(line.strip())
         output_sents = normalise_text(list_sents,
                                       batch_size=args.batch_size,
                                       beam_size=args.beam_size,
                                       cache_file=args.cache_lexicon,
                                       no_postproc_lex=args.no_postproc_lex,
                                       no_post_clean=args.no_post_clean)
         for output_sent in output_sents:
              print(output_sent['text'])
