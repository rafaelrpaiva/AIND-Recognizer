import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    X_lengths = test_set.get_all_Xlengths()

    for X, lengths in X_lengths.values():
        max_score = None
        best_choice = None
        scores_dict = dict()

        for word, model in models.items():
            try:
                score = model.score(X, lengths)
            except:
                score = float("-inf")

            scores_dict[word] = score

            # Here, a better score increases the level and adopts the word as the best choice.
            if (max_score is None) or (score > max_score):
                max_score = score
                best_choice = word

        probabilities.append(scores_dict)
        guesses.append(best_choice)

    return probabilities, guesses
