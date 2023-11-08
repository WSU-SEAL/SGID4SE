from SGID4SE import SGID4SE

SGIDModel =SGID4SE(ALGO="BERT", count_words=False)

SGIDModel.init_predictor(strategy="random", ratio=1)



sentences=["A faggot wrote this code",
           "Yo momma is so fat",
           "Crap, this is an artifact of a previous revision. It's simply the last time a change was made to Tuskar's cloud.",
            "girl, you will look too sexy in a bikini, so horny",
           "you are an absolute b!tch",
           "I love this implementation"]


results = SGIDModel.get_SGID_probablity(sentences)
for i in range(len(sentences)):
    print("\"" + sentences[i] + "\" ->" + str(results[i]))  # probablity of being an SGID.
