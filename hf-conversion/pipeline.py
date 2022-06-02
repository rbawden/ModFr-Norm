#!/usr/bin/python
from transformers import Pipeline, pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from torch import Tensor
import html.parser
import unicodedata
import sys, os
import re
from tqdm.auto import tqdm
import operator

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
    Calculate the minimum Levenshtein edit-distance based alignment
    mapping between two strings. The alignment finds the mapping
    from string s1 to s2 that minimizes the edit distance cost.
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


######## Normaliation pipeline #########
class NormalisationPipeline(Pipeline):

    def __init__(self, beam_size=5, batch_size=32, **kwargs):
        self.beam_size = beam_size
        super().__init__(**kwargs)

    
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
        #line = unicodedata.normalize('NFKC', line)
        #line = self.make_printable(line)
        for before, after in [('[«»\“\”]', '"'),
                              ('[‘’]', "'"),
                              (' +', ' '),
                              ('\"+', '"'),
                              ("'+", "'"),
                              ('^ *', ''),
                              (' *$', '')]:
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

    def postprocess(self, model_outputs, clean_up_tokenisation_spaces=False):
        records = []
        for output_ids in model_outputs["output_ids"][0]:
            record = {
                "text": self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenisation_spaces=clean_up_tokenisation_spaces,
                )
            }
            records.append(record)
        return records

    def correct_hallunications(self, orig, output):
        # align the original and output tokens

        # check that the correspondences are legitimate and correct if not

        # replace <EMOJI> symbols by the original ones
        return output

    def __call__(self, *args, **kwargs):
        r"""
        Generate the output text(s) using text(s) given as inputs.
        Args:
            args (`str` or `List[str]`):
                Input text for the encoder.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenisation_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (`TruncationStrategy`, *optional*, defaults to `TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenisation within the pipeline. `TruncationStrategy.DO_NOT_TRUNCATE`
                (default) will never truncate, but it is sometimes desirable to truncate the input to fit the model's
                max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).
        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:
            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
        """

        result = super().__call__(*args, **kwargs)
        if (
            isinstance(args[0], list)
            and all(isinstance(el, str) for el in args[0])
            and all(len(res) == 1 for res in result)
        ):
            return [res[0] for res in result]
        if type(args) is List:
            alignments = [cls.align(args[i], result[i]) for x in range(len(*args))]}
        else:
            alignments = cls.align(args)
        return {'predictions': result, 'alignments': alignments}

    def align(cls, src, pred):
        backpointers = wedit_distance_align(homogenise(sent_ref), homogenise(sent_pred))
        alignment, current_word, seen1, seen2, last_weight = [], ['', ''], [], [], 0
        for i_ref, i_pred, weight in backpointers:
            if i_ref == 0 and i_pred == 0:
                continue
            # spaces in both, add straight away
            if i_ref <= len(sent_ref) and sent_ref[i_ref-1] == ' ' and \
               i_pred <= len(sent_pred) and sent_pred[i_pred-1] == ' ':
                alignment.append((current_word[0].strip(), current_word[1].strip(), weight-last_weight))
                last_weight = weight
                current_word = ['', '']
                seen1.append(i_ref)
                seen2.append(i_pred)
            else:
                end_space = '░'
                if i_ref <= len(sent_ref) and i_ref not in seen1:
                    if i_ref > 0:
                        current_word[0] += sent_ref[i_ref-1]
                        seen1.append(i_ref)
                if i_pred <= len(sent_pred) and i_pred not in seen2:
                    if i_pred > 0:
                        current_word[1] += sent_pred[i_pred-1] if sent_pred[i_pred-1] != ' ' else '▁'
                        end_space = '' if space_after(i_pred, sent_pred) else '░'
                        seen2.append(i_pred)
                if i_ref <= len(sent_ref) and sent_ref[i_ref-1] == ' ' and current_word[0].strip() != '':
                    alignment.append((current_word[0].strip(), current_word[1].strip() + end_space, weight-last_weight))
                    last_weight = weight
                    current_word = ['', '']
        # final word
        alignment.append((current_word[0].strip(), current_word[1].strip(), weight-last_weight))
        # check that both strings are entirely covered
        recovered1 = re.sub(' +', ' ', ' '.join([x[0] for x in alignment]))
        recovered2 = re.sub(' +', ' ', ' '.join([x[1] for x in alignment]))

        assert recovered1 == re.sub(' +', ' ', sent_ref), \
            '\n' + re.sub(' +', ' ', recovered1) + "\n" + re.sub(' +', ' ', sent_ref)
        assert re.sub('[░▁ ]+', '', recovered2) == re.sub('[▁ ]+', '', sent_pred), recovered2+" / "+sent_pred
    return alignment

   
def normalise_text(list_sents, batch_size=32, beam_size=5):
    tokeniser = AutoTokenizer.from_pretrained("rbawden/modern_french_normalisation", use_auth_token=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("rbawden/modern_french_normalisation", use_auth_token=True)
    normalisation_pipeline = NormalisationPipeline(model=model,
                                              tokenizer=tokeniser,
                                              batch_size=batch_size,
                                              beam_size=beam_size)
    normalised_outputs = normalisation_pipeline(list_sents)
    return normalised_outputs

def normalise_from_stdin(batch_size=32, beam_size=5):
    tokeniser = AutoTokenizer.from_pretrained("rbawden/modern_french_normalisation", use_auth_token=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("rbawden/modern_french_normalisation", use_auth_token=True)
    normalisation_pipeline = NormalisationPipeline(model=model,
                                              tokenizer=tokeniser,
                                              batch_size=batch_size,
                                              beam_size=beam_size)
    list_sents = []
    for sent in sys.stdin:
        list_sents.append(sent)
    normalised_outputs = normalisation_pipeline(list_sents)
    for sent in normalised_outputs:
        print(sent['text'].strip())
    return normalised_outputs

    
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--batch_size', type=int, default=32, help='Set the batch size for decoding')
    parser.add_argument('-b', '--beam_size', type=int, default=5, help='Set the beam size for decoding')
    parser.add_argument('-i', '--input_file', type=str, default=None, help='Input file. If None, read from STDIN')
    args = parser.parse_args()

    if args.input_file is None:
         normalise_from_stdin(batch_size=args.batch_size, beam_size=args.beam_size)
    else:
         list_sents = []
         with open(args.input_file) as fp:
              for line in fp:
                   list_sents.append(line.strip())
         output_sents = normalise_text(list_sents, batch_size=args.batch_size, beam_size=args.beam_size)
         for output_sent in output_sents:
              print(output_sent)
