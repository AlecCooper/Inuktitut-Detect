import numpy as np
import torch as torch
import string
import os

class Data():

    def __init__(self, file_location, size, from_file, num_test):

        self.size = size

        # ingesting from previosuly saved numpy arrays
        if from_file:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            dir_path += "/Data"
            x_train = np.load(dir_path + "/x_train.npy")
            x_test = np.load(dir_path + "/x_test.npy")
            y_train = np.load(dir_path + "/y_train.npy")
            y_test = np.load(dir_path + "/y_test.npy")
            
        # not from file, we are ingesting from the corpus
        else:
            # ingest and clean tokens from corpus
            inuktitut_tokens, english_tokens = self.__ingest(file_location)

            # convert to ascii representation
            inuktitut_tokens = self.__to_ascii(inuktitut_tokens)
            english_tokens = self.__to_ascii(english_tokens)

            # combine into an 2d array
            # 1s representing a english and 0s inuktitut

            ones = np.ones((len(english_tokens),1),dtype=np.uint8)
            zeros = np.zeros((len(inuktitut_tokens),1),dtype=np.uint8)

            e_data = np.concatenate((english_tokens,ones),axis=1)
            i_data = np.concatenate((inuktitut_tokens,zeros),axis=1)

            # join english and inuktitut data together
            data = np.concatenate((e_data,i_data))
            
            # shuffle data to randomize
            np.random.shuffle(data)

            # Break into training and testing data
            num_train = len(data) - num_test

            train_data = data[:num_train]
            test_data = data[num_train:]
            print("Num of training samples")
            print(len(train_data))

            # break into our x and y data
            x_train = np.copy(train_data[:,:size])
            x_test = np.copy(test_data[:,:size])

            y_train = np.copy(train_data[:,size])
            y_test = np.copy(test_data[:,size])

            # save data
            dir_path = os.path.dirname(os.path.realpath(__file__))
            if not os.path.exists(dir_path + "/Data"):
                os.mkdir(dir_path + "/Data")
            dir_path += "/Data"
            np.save(dir_path + "/x_train",x_train)
            np.save(dir_path + "/x_test",x_test)
            np.save(dir_path + "/y_train",y_train)
            np.save(dir_path + "/y_test",y_test)
        
        # Convert to one hot vectors
        x_train = self.__one_hot(x_train)
        x_test = self.__one_hot(x_test)

        # Convert to tensor
        x_train = torch.from_numpy(x_train)
        x_test = torch.from_numpy(x_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        # Convert to double type to satisfy pytorch's lstm requirements
        self.x_train = x_train.float().clone()
        self.x_test = x_test.float().clone()
        self.y_train = y_train.float().clone()
        self.y_test = y_test.float().clone()


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
        english_tokens = np.array(self.__tokienize(english))
        print("Inuktitut line tokenization")
        inuktitut_tokens = np.array(self.__tokienize(inuktitut))

        # Remove and tokens that appear in both lists
        inuktitut_tokens, english_tokens = self.__remove_dupes(inuktitut_tokens,english_tokens)

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

        # If token is larger than our size we discard it
        if len(token) > self.size:
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

        # Confirm token is ascii
        if any(ord(char) > 128 for char in token):
            return ""

        # replace our & consonent from A to { (one about z in ascii)
        token = token.replace("A","{")

        # Add padding
        while len(token) < self.size:
            token += "`"  #when mapping to out one hot vector, will map to 0 vector

        return token

    # removes any tokens that appear in both token lists
    def __remove_dupes(self,i_tokens,e_tokens):

        # Inefficent method, must update

        new_i_token = []
        new_e_token = []

        for i_token in i_tokens:

            if not i_token in e_tokens:
                new_i_token.append(i_token)

        for e_token in e_tokens:

            if not e_token in i_tokens:
                new_e_token.append(e_token)

        return new_i_token, new_e_token

    # convert each token into a numpy array of its ascii representation
    def __to_ascii(self,tokens):

        token_array = np.empty((len(tokens),self.size),dtype=np.uint8)

        i = 0

        for token in tokens:

            # convert token to ascii representation
            char_array = np.array(str(token),dtype="c")

            #-96 so when we map to vectors the values in this array are between 0-97
            ascii_array = np.array(char_array.view(dtype=np.uint8)) - 96

            # append to array
            token_array[i] = ascii_array
            
            i += 1

        return token_array

    def __one_hot(self,token_array):

        # Create array for 1 hot data
        hot_array = np.zeros([token_array.shape[0],token_array.shape[1],29])

        # loop index
        i = 0

        # loop through each character and encode
        for token in token_array:
            j = 0
            for char in token:

                # Create one hot vector
                ohv = np.zeros([29],dtype=np.bool)
                ohv[char] = 1

                # Add to our array
                hot_array[i,j,:] = ohv

                j+=1

            i+=1

        return hot_array

#data = Data("/Users/aleccooper/Documents/Translate/Corpus/test.txt",10,False,2)
#print(data.x_test.size())