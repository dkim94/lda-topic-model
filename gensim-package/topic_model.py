'''
class LDAmodel is used for actually building and running the LDA model
'''

# gensim for LDA model
import gensim
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel

import numpy as np

from wrapper import timed


class LDAModel:

    def __init__(self, documents):
        self.documents = documents

    def _create_dictionary_and_corpus(self) -> None:
        self.id2word = gensim.corpora.Dictionary(self.documents)
        self.corpus = [self.id2word.doc2bow(document) for document in self.documents]

    @timed
    def run(self, num_topics, passes, alpha, eta) -> LdaModel:
        self._create_dictionary_and_corpus()
        self.lda_model = LdaModel(corpus=self.corpus,
                                    id2word=self.id2word,
                                    num_topics=num_topics,
                                    random_state=100,
                                    update_every=1,
                                    chunksize=100,
                                    passes=passes,
                                    alpha=alpha,
                                    eta=eta,
                                    per_word_topics=True,
                                    minimum_probability=0.0)    # set minimum_probability to 0.0 to show all topics
        self._save_model(num_topics, passes, alpha, eta)
        self.document_topics = self.lda_model[self.corpus]
        return self.lda_model


    # save and load model 

    def _save_model(self, num_topics, passes, alpha, eta):
        self.lda_model.save('./saved_topic_models/topic_model_%d_%d_%.2f_%.2f' % (num_topics, passes, alpha, eta))

    def load_model(self, num_topics, passes, alpha, eta):
        try:
            model_name = './saved_topic_models/topic_model_%d_%d_%.2f_%.2f' % (num_topics, passes, alpha, eta)
            self.lda_model = LdaModel.load(model_name) 
            self.id2word =  gensim.corpora.Dictionary.load(f'{model_name}.id2word')
            self.corpus = [self.id2word.doc2bow(document) for document in self.documents]
            self.document_topics = self.lda_model[self.corpus]
            print("load model_%d_%d_%.2f_%.2f" % (num_topics, passes, alpha, eta)) 
        except:
            print("No model found.")    


    # print model info

    def get_topic_information_of_document(self, document_num):
        return self.document_topics[document_num][0]       

    def get_word_information_of_topic(self, topic_id: int):
        return self.lda_model.print_topic(topic_id)


    # TODO
    # need ignored word set
    def inference(self, new_document) -> np.ndarray:
        self.unseen_doc = [self.id2word.doc2bow(new_document)]
        result = []
        for id, prob in self.lda_model[self.unseen_doc[0]][0]:
            result.append(round(prob, 3))
        return np.asarray(result)
        
    
    # perplexity and coherence score

    def compute_perplexity(self) -> float:
        return self.lda_model.log_perplexity(self.corpus)
        
    def compute_coherence_score(self) -> float:
        coherence_model_lda = CoherenceModel(model=self.lda_model, 
                                            texts=self.documents, 
                                            dictionary=self.id2word,
                                            coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda