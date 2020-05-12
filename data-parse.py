import numpy as np
import torch as torch
import string

class Data():

    def __init__(self, file_location, size, from_file, num_test):

        self.size = size

        # ingesting from previosuly saved numpy arrays
        if from_file:
            self.train_data = np.fromfile("train_data.npy")
            self.test_data = np.fromfile("test_data.npy")
            
        # not from file, we are ingesting from the corpus
        else:
            # ingest and clean tokens from corpus
            inuktitut_tokens, english_tokens = self.__ingest(file_location)

            # combine into an 2d array
            # 1s representing a english and 0s inuktitut

            ones = np.ones((len(english_tokens),1),dtype=np.bool)
            zeros = np.zeros((len(inuktitut_tokens),1),dtype=np.bool)

            e_data = np.concatenate((english_tokens,ones),axis=1)
            i_data = np.concatenate((inuktitut_tokens,zeros),axis=1)

            # join english and inuktitut data together
            data = np.concatenate((e_data,i_data))
            
            # shuffle data to randomize
            np.random.shuffle(data)

            # Break into training and testing data
            num_train = len(data) - num_test

            self.train_data = data[:num_train]
            self.test_data = data[num_train:]

            # save data
            np.save("train_data",self.train_data)
            np.save("test_data",self.test_data)


    def __ingest(self,file_location):

        # Define list of sentences in each language
        english = []
        inuktitut = []

        # Read in the file
        corpus_file =  open(file_location, "r",)
        corpus = corpus_file.readlines()
        corpus_file.close()

        # Get size of corpus
        length = len(corpus)

        # loop index
        i = 0

        while i < length-3:

            # Determine we have the start of a couplet
            if "***************" in corpus[i]:
        
                # Clean the inuktitut and english text
                i_text = corpus[i+1]
                e_text = corpus[i+3]

                # Append to corpus lists
                inuktitut.append(i_text)
                english.append(e_text)

                i+= 4

            else: # we don't have the start of a couplet, next line
                i+=1

        # tokienize our lines
        print("English line tokenization")
        english_tokens = np.array([self.__tokienize(english)]).transpose()
        print("Inuktitut line tokenization")
        inuktitut_tokens = np.array([self.__tokienize(inuktitut)]).transpose()

        return inuktitut_tokens, english_tokens

    def __tokienize(self,lines):

        # dictonary so we only make unique tokiens
        t_dict = {}

        tokens = []
        ctr = 0

        # Tokienize
        for line in lines:

            line_tokens = []
            current_token = ""

            # loop through and find token delimiters
            for c in line:

                # whitespaces incidactes delimiter between words
                if c == " " or c == "\n":
                    if len(current_token) > 0:
                        current_token = self.__clean(current_token)
                        if len(current_token) > 0: 
                            if not current_token in t_dict: # make sure not already tokenized
                                line_tokens.append(current_token)
                                t_dict[current_token] = True 
                        current_token = ""
                else:
                    current_token += c


            tokens = tokens + line_tokens
    
            ctr += 1
            if ctr % 10000 == 0:
                print("{0}/{1} lines tokenized".format(ctr,len(lines)))

        return tokens

    # Clean a token of punctuation
    def __clean(self,token):

        # make lowercase
        token = token.lower()

        # If we have an empty string we return it
        if len(token) == 0:
            return ""

        # If string contains numberic information, it is discarded as a token
        if any(char.isdigit() for char in token):
            return ""

        # Deal with the inuktitut character & by changing it to a uppercase letter
        token = token.replace("&","A")

        # remove any punctuation from our string
        table = str.maketrans("", "", string.punctuation)
        token = token.translate(table)

        # If not alphabetic, remove so we are just left with letters (including &)
        if not token.isalpha():
            return ""

        # replace our & consonent from A to %26 (as required by the analyzer)
        token = token.replace("A","%26")

        # Add padding
        while len(token) < self.size:
            token += "@"

        return token

data = Data("/Users/aleccooper/Documents/Translate/Corpus/test.txt",10,False,3)