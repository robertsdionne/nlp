require 'torch'
require 'nn'

dofile 'TaggedSentence.lua'
dofile 'DataLoader.lua'

tagged_sentences = loadData(TRAIN_FILENAME)
assert(39815 == #tagged_sentences)
assert("In_IN an_DT Oct._NNP 19_CD review_NN of_IN ``_`` The_DT Misanthrope_NN ''_'' at_IN " ..
    "Chicago_NNP 's_POS Goodman_NNP Theatre_NNP (_( ``_`` Revitalized_VBN Classics_NNS Take_VBP " ..
    "the_DT Stage_NN in_IN Windy_NNP City_NNP ,_, ''_'' Leisure_NN &_CC Arts_NNS )_) ,_, the_DT " ..
    "role_NN of_IN Celimene_NNP ,_, played_VBN by_IN Kim_NNP Cattrall_NNP ,_, was_VBD " ..
    "mistakenly_RB attributed_VBN to_IN Christina_NNP Haag_NNP ._." ==
        tagged_sentences[1]:toString())
assert("That_DT could_MD cost_VB him_PRP the_DT chance_NN to_TO influence_VB the_DT outcome_NN " ..
  "and_CC perhaps_RB join_VB the_DT winning_VBG bidder_NN ._." ==
      tagged_sentences[39815]:toString())
