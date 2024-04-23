import sys
from pathlib import Path
import gensim
from gensim import models
import tqdm
import datetime
from . import file_util
import global_options


def train_bigram_model(input_path, model_path):
    """Train a phrase model and save it to the disk.
    
    Arguments:
        input_path {str or Path} -- input corpus
        model_path {str or Path} -- where to save the trained phrase model?
    
    Returns:
        gensim.models.phrases.Phrases -- the trained phrase model
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    print(datetime.datetime.now())
    print("Training phraser...")
    corpus = gensim.models.word2vec.PathLineSentences(str(input_path), max_sentence_length=10000000)
    n_lines = file_util.line_counter(input_path)
    bigram_model = models.Phrases(tqdm.tqdm(corpus, total=n_lines),
                                  min_count=global_options.PHRASE_MIN_COUNT,
                                  threshold=global_options.PHRASE_THRESHOLD,
                                  scoring="default")  # Removed common_terms if unsupported
    bigram_model.save(str(model_path))
    return bigram_model

def bigram_transform(line, bigram_phraser):
    """Helper function for file_bigramer
    Note: Needs a phraser object or phrase model.
    
    Arguments:
        line {str} -- a line 
        return: a line with phrases joined using "_"
    """
    return " ".join(bigram_phraser[line.split()])

def file_bigramer(input_path, output_path, model_path, threshold=None, scoring=None):
    """Transform an input text file into a file with 2-word phrases. 
    Apply again to learn 3-word phrases.
    
    Arguments:
        input_path {str} -- Each line is a sentence
        output_file {str} -- Each line is a sentence with 2-word phrases concatenated
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    bigram_model = gensim.models.Phrases.load(str(model_path))
    if scoring is not None:
        bigram_model.scoring = getattr(gensim.models.phrases, scoring)
    if threshold is not None:
        bigram_model.threshold = threshold
    # bigram_phraser = models.Phraser(bigram_model)
    with open(input_path, "r") as f:
        input_data = f.readlines()
    data_bigram = [bigram_transform(l, bigram_model) for l in tqdm.tqdm(input_data)]
    with open(output_path, "w") as f:
        f.write("\n".join(data_bigram) + "\n")
    assert len(input_data) == file_util.line_counter(output_path)

def train_w2v_model(input_path, model_path, *args, **kwargs):
    """ Train a word2vec model using the LineSentence file in input_path, 
    save the model to model_path.
    
    Arguments:
        input_path {str} -- Corpus for training, each line is a sentence
        model_path {str} -- Where to save the model?
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    corpus = gensim.models.word2vec.PathLineSentences(str(input_path), max_sentence_length=10000000)
    
    # Update kwargs for vector_size if size is provided
    if 'size' in kwargs:
        kwargs['vector_size'] = kwargs.pop('size')
    if 'iter' in kwargs:
        kwargs['epochs'] = kwargs.pop('iter')
    
    model = gensim.models.Word2Vec(corpus, *args, **kwargs)
    model.save(str(model_path))