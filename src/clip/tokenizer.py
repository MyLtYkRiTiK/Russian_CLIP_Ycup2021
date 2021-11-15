import html
from functools import lru_cache

import ftfy
import regex as re


@lru_cache( )
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    rus = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    rus_cap = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    bs = list(range(33)) + list(range(127, 160)
                                ) + [160] + [173] + list(range(ord("!"), ord("~") + 1)
                                                         ) + list(range(ord("¡"),
                                                                        ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))

    cs = list(rus) + list(rus_cap) + ['Ǽ', 'ǽ'] + [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set( )
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip( )


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip( )
    return text


@lru_cache( )
def add_merges(bpe_path):
    merges = open(bpe_path).read( ).split('\n')
    merges = merges[1:]
    merges = [tuple(merge.split( )) for merge in merges]
    return merges


class SimpleTokenizer(object):
    def __init__(self):
        self.byte_encoder = bytes_to_unicode( )
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items( )}
        vocab = list(bytes_to_unicode( ).values( ))
        vocab = vocab + [v + '</w>' for v in vocab]
        merges = add_merges('path_to_file/bpe_simple_vocab.txt')
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        merges = add_merges('./src/clip/merges_russian.txt')
        for merge in merges:
            vocab.append(''.join(merge))
        merges = add_merges('path_to_file/bpe_simple_vocab.txt') + add_merges('path_to_file/merges_russian.txt')
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items( )}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower( )
        for token in re.findall(self.pat, text):
            try:
                token = ''.join(self.byte_encoder[self.byte_decoder[b]] for b in token)
            except KeyError:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens]).replace('</w>', ' ')
        return text