from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import Levenshtein as Lev


def calc_sentence_bleu_score(reference,hypothesis,n=4):
    """
    This function receives as input a reference sentence(list) and a
    hypothesis(list) and
    returns bleu score.
    (list of words should be given)
    Bleu score formula: https://leimao.github.io/blog/BLEU-Score/
    """
    reference = [reference]
    weights = [1/n for _ in range(n)]
    smoothie = SmoothingFunction()
    return sentence_bleu(reference, hypothesis, weights, SmoothingFunction().method1)
    #return corpus_bleu([reference], [hypothesis], weights)


def calc_word_error_rate(s1,s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences (list of words should be given).
    s1:reference
    s2:hypothesis
    """

    # build mapping of words to integers
    ba = set(s1+ s2)
    word2idx = dict(zip(ba, range(len(ba))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2idx[w]) for w in s1]
    w2 = [chr(word2idx[w]) for w in s2]
    return Lev.distance(''.join(w1), ''.join(w2)) / len(s1)