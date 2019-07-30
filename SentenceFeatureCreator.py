from collections import Counter
import logging
from pathlib import Path

from benepar.spacy_plugin import BeneparComponent
from nltk import Tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import spacy
from spacy.util import minibatch as mb

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SentenceFeatureCreator:
    def __init__(self,
                 spacy_mdl='en_core_web_sm',
                 benepar_mdl='benepar_en2',
                 is_segmented=True,
                 is_tokenised=False,
                 batch_size=20,
                 take_sent_average=True,
                 scaler='minmax'):
        self.nlp = spacy.load(spacy_mdl)
        self.nlp.add_pipe(BeneparComponent(benepar_mdl), name='benepar')

        if is_segmented:
            self.nlp.add_pipe(self._prevent_sbd, name='prevent-sbd', before='parser')

        self.is_tokenised = is_tokenised
        if is_tokenised:
            self.nlp.tokenizer = self.nlp.tokenizer.tokens_from_list

        self.batch_size = batch_size
        self.take_sent_average = take_sent_average

        if scaler == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler is None:
            self.scaler = None
        else:
            raise ValueError("'scaler' has an unexpected value. Use 'minmax' or 'standard' or None.")

    def process_file(self, fin_src, fin_label, fout=None):
        pfin_src = Path(fin_src).resolve()
        pfin_label = Path(fin_label).resolve()
        pfout = pfin_label.with_suffix('.out') if fout is None else Path(fout).resolve()

        X = []

        with pfin_src.open('r', encoding='utf-8') as fhin_src, \
                pfin_label.open('r', encoding='utf-8') as fhin_label:
            logging.info('Reading labels...')
            labels = [float(l.rstrip()) for l in fhin_label.readlines()]

            input_data = enumerate(mb(fhin_src, self.batch_size), 1)
            for result in map(self.process_batch, input_data):
                X.extend(result)

        if self.scaler is not None:
            logger.info(f"Scaling data with {str(self.scaler)}...")
            X = self.scaler.fit_transform(X).tolist()

        feats_reprs = self.create_feats_str(X, labels)

        logger.info(f"Writing output to {pfout}...")
        with pfout.open('w', encoding='utf-8') as fhout:
            fhout.write('\n'.join(feats_reprs) + '\n')

        logger.info('Finished processing!')

    def process_batch(self, batch_tuple):
        batch_idx, batch = batch_tuple
        del batch_tuple

        logging.info(f"Parsing batch #{batch_idx:,}...")
        batch = [s.rstrip() for s in batch]

        if self.is_tokenised:
            # Because the text was already tokenised we can split on white space to get tokens
            batch = [s.split() for s in batch]

        docs = self.nlp.pipe(batch)
        del batch
        sents = [sent for doc in docs for sent in doc.sents]

        sents_repr = []
        for sent_idx, sent in enumerate(sents, 1):
            logger.info(f"Processing batch #{batch_idx:,}, sentence #{sent_idx:,}")
            pos_counter = self.count_prop(sent, 'pos_')
            dep_counter = self.count_prop(sent, 'dep_')
            n_ents = len(sent.ents)
            tree_depth = self.tree_depth(sent)
            n_toks = len(sent)

            sent_repr = [tree_depth,  # tree depth
                         pos_counter['CCONJ'],  # no. coordinating conjuncts
                         pos_counter['SCONJ'],  # no. subordinating conjuncts
                         pos_counter['PUNCT'],  # no. punctuation marks
                         pos_counter['ADJ']  # no. content words
                         + pos_counter['NOUN']
                         + pos_counter['PROPN']
                         + pos_counter['VERB'],
                         dep_counter['subj'],  # no. subjects
                         dep_counter['dobj'],  # no. objects
                         n_ents  # no. named entities
                         ]

            if self.take_sent_average:
                # Normalize by number of tokens
                sent_repr = [x / n_toks for x in sent_repr]

            sent_repr.append(n_toks)
            sents_repr.append(sent_repr)

        return sents_repr

    @staticmethod
    def create_feats_str(sents_feats, labels):
        feats_reprs = []
        for sent_feats, label in zip(sents_feats, labels):
            s = str(label)
            for idx, feat in enumerate(sent_feats, 1):
                s += f" {idx}:{feat}"
            feats_reprs.append(s)

        return feats_reprs

    @staticmethod
    def count_prop(sent, prop):
        """ Counts values for a given prop.
            E.g. for prop pos_, count how many PRON, VERB, ... are in the sentence"""
        return Counter([getattr(token, prop) for token in sent])

    @staticmethod
    def tree_depth(span):
        """ Gets the depth of a parse tree by using the spaCy parse (Berkley parser) and NLTK.
            This is probably quite inefficient because we parse the tree with spaCy, turn it into
            a string, and then read that string into an NLTK tree. benepar doesn't seem to provide a straightforward
            way to access the non-terminal nodes, though, so using NLTK seems the best approach.
            We use spaCy for the rest because we like the API and speed.
            """
        tree = Tree.fromstring(span._.parse_string)

        return tree.height()

    @staticmethod
    def _prevent_sbd(doc):
        # If you already have one sentence per line in your file
        # you may wish to disable sentence segmentation with this function,
        # which is added to the nlp pipe in the constructor
        for token in doc:
            token.is_sent_start = False

        return doc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Generate feature representation of sentence in SVM format.')
    parser.add_argument('fin_src', help='path to text file.')
    parser.add_argument('fin_labels', help='path to file with labels.')
    parser.add_argument('--fout', help='path to output file.')

    parser.add_argument('--spacy-mdl', default='en_core_web_sm', help='spaCy model to use.')
    parser.add_argument('--benepar-mdl', default='benepar_en2', help='benepar model to use.')

    parser.add_argument('--not-segmented', action='store_true', default=False,
                        help='has the text not been sentence segmented yet?')
    parser.add_argument('--is-tokenised', action='store_true', default=False,
                        help='has the text nbeen tokenized?')

    parser.add_argument('-b', '--batch-size', default=1000, type=int, help='number of sentences to process per batch.')
    parser.add_argument('--no-average', action='store_true', default=False,
                        help='do not normalize the values by sentence length.')

    parser.add_argument('--scaler', default=None,
                        choices={'minmax', 'standard', None},
                        help='scale the data with this scaler.')

    args = parser.parse_args()

    creator = SentenceFeatureCreator(spacy_mdl=args.spacy_mdl,
                                     benepar_mdl=args.benepar_mdl,
                                     is_segmented=not args.not_segmented,
                                     is_tokenised=args.is_tokenised,
                                     batch_size=args.batch_size,
                                     take_sent_average=not args.no_average,
                                     scaler=args.scaler)

    creator.process_file(args.fin_src, args.fin_labels)
