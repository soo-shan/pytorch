PAD_token = 0 # used for padding short sentences
SOS_token = 1 # start of sentence token <START>
EOS_token = 2 # end of sentence token <END>

class Vocabulary:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token:'PAD',SOS_token:'SOS', EOS_token:'EOS'}
        self.num_words = 3 # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        keep_words = []
        for k,v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        
                print('keep words {}/{}={:.4f}'.format(len(keep_words),len(self.word2index),len(keep_words)/len(self.word2index)))
                # Reinitialize dictionaries
                self.word2index = {}
                self.word2count = {}
                self.index2word = {PAD_token:'PAD',SOS_token:'SOS', EOS_token:'EOS'}
                self.num_words = 3 #count default tokens

        for word in keep_words:
            self.addWord(word)

# ms_dict =  Vocabulary('msdict')
# ms_dict.addSentence('this is first test sentence')
# ms_dict.addSentence('this is the next sentence and this will be fun to the add. please add more words and sentence')
# ms_dict.trim(2)