from typing import Generator, List
import regex as re
import string
#from pie.tagger import regexsplitter, SECTION, FULLSTOP

def regexsplitter(regex):
    def func(line):
        for line in re.split(regex, line):
            line = line.strip()
            if line:
                yield line
    return func


SECTION = r'([A-Z\. ]+\.)'
FULLSTOP = r'([^\.]+\.)'


class SimpleTokenizer(object):
    """ Tokenizer that memoryze what it tokenized.

    Mostly used to normalized input as input time and then reinserting normalized input

    """
    def __init__(self):
        self.section = regexsplitter(SECTION)
        self.fullstop = regexsplitter(FULLSTOP)
        self.word = regexsplitter(WORD)

    def sentence_tokenizer(self, text: str, lower: bool = False) -> Generator[List[str], None, None]:
        for line in self.section(text):
            for sentence in self.fullstop(line):
                yield self.word_tokenizer(sentence, lower=lower)

    def word_tokenizer(self, text: str, lower: bool = False) -> List[str]:
        sentence = [w for raw in text.split() for w in self.word(raw)]
        if lower:
            sentence = [w.lower() for w in sentence]
        return sentence

    def reset(self):
        """Can be used between documents for example """
        pass
