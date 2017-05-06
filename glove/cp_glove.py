import tf_glove
import matplotlib
embSize = 50
minOccur = 5

model = tf_glove.GloVeModel(embedding_size=embSize, context_size=10, min_occurrences=minOccur,
                            learning_rate=0.5, batch_size=512)
# min_occurrences=25,
import re
import nltk


def extract_reddit_comments(path):

    # A regex for extracting the comment body from one line of JSON (faster than parsing)
    body_snatcher = re.compile(r"\{.*?(?<!\\)\"body(?<!\\)\":(?<!\\)\"(.*?)(?<!\\)\".*}")
    with open(path) as file_:
        for line in file_:
            yield line
            # match = body_snatcher.match(line)
            # if match:
            #     body = match.group(1)
            #     # Ignore deleted comments
            #     if not body == '[deleted]':
            #         # Return the comment as a string (not yet tokenized)
            #         yield body


def tokenize_comment(comment_str):
    # Use the excellent NLTK to tokenize the comment body
    #
    # Note that we're lower-casing the comments here. tf_glove is case-sensitive,
    # so if you want 'You' and 'you' to be considered the same word, be sure to lower-case everything.
    return nltk.wordpunct_tokenize(comment_str.lower())


def reddit_comment_corpus(path):
    # A generator that returns lists of tokens representing individual words in the comment
    return (tokenize_comment(comment) for comment in extract_reddit_comments(path))


# Replace the path with the path to your corpus file
# corpus = reddit_comment_corpus("/media/grady/PrimeMover/Datasets/RC_2015-01-1m_sample")
corpus = reddit_comment_corpus("./../data/output2glove_152m.txt")
# corpus = reddit_comment_corpus("./miniInput.csv")

model.fit_to_corpus(corpus)

if embSize < 300:
    num_epo = 50
else:
    num_epo = 100
model.train(num_epochs=num_epo, log_dir="log/example", summary_batch_interval=1000)


# print embeddingsOut
# print embFor96315
words = model.words
# print words
fileOut = file("./../data/model_152m_" + str(embSize) + ".vector", "a+")
# fileOut.write(str(len(words)) + "\t" + str(embSize) + "\n")
for word in words:
    wordEmb = model.embedding_for(word)
    emd_str = word
    for a in wordEmb:
        emd_str += "\t" + str(a)
    emd_str += "\n"
    print emd_str
    fileOut.write(emd_str)
fileOut.close()


model.generate_tsne()
# model.generate_tsne()
print "end"
# fileOut = file("./model_50.txt", "a+")
# with open("./map.csv") as file:
#     for line in file:
#         wordNum = line.split()[1]
#         wordEmb = model.embedding_for(wordNum)
#         emd_str = wordNum
#         for a in wordEmb:
#             emd_str += "\t" + str(a)
#         emd_str += "\n"
#         print emd_str
#         fileOut.write(wordEmb)
# fileOut.close()
# print "end"