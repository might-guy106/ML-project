import numpy as np


# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT PERFORM ANY FILE IO IN YOUR CODE

# DO NOT CHANGE THE NAME OF THE METHOD my_fit or my_predict BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc
class Tree:
    def __init__(self, min_leaf_size=1, max_depth=None):
        self.root = None
        self.words = None
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth if max_depth is not None else float('inf')

    def fit(self, words, verbose=False):
        self.words = words
        self.root = Node(depth=0, parent=None)
        if verbose:
            print("root")
            print("└───", end='')
        self.root.fit(all_words=self.words, my_words_idx=np.arange(len(self.words)), min_leaf_size=self.min_leaf_size,
                      max_depth=self.max_depth, verbose=verbose)

        # Clear unnecessary data after training
        self.root.clear_history()

    def predict(self, bigrams):
        node = self.root
        while not node.is_leaf:
            node = node.get_child(node.get_query() in bigrams)
        return [self.words[i] for i in node.my_words_idx][:5]


class Node:
    __slots__ = ['depth', 'parent', 'my_words_idx', 'children', 'is_leaf', 'query', 'history']

    def __init__(self, depth, parent):
        self.depth = depth
        self.parent = parent
        self.my_words_idx = None
        self.children = {}
        self.is_leaf = True
        self.query = None
        self.history = set()

    def get_query(self):
        return self.query

    def get_child(self, response):
        if self.is_leaf:
            return self
        if response not in self.children:
            response = list(self.children.keys())[0]
        return self.children[response]

    def process_leaf(self, my_words_idx):
        self.my_words_idx = my_words_idx

    def get_bigrams(self, word):
        bigrams = sorted([word[i:i + 2] for i in range(len(word) - 1)])
        return bigrams[:5]

    def get_optimal_bigram(self, all_words, my_words_idx, history):
        bigram_counts = {}
        for idx in my_words_idx:
            bigrams = set(self.get_bigrams(all_words[idx]))
            for bigram in bigrams:
                if bigram in history:
                    continue
                if bigram not in bigram_counts:
                    bigram_counts[bigram] = 0
                bigram_counts[bigram] += 1

        if not bigram_counts:
            return None

        optimal_bigram = max(bigram_counts, key=bigram_counts.get)
        return optimal_bigram

    def process_node(self, all_words, my_words_idx, history):
        query = self.get_optimal_bigram(all_words, my_words_idx, history)
        if query is None:
            return query, {True: my_words_idx, False: []}

        split_dict = {True: [], False: []}
        for idx in my_words_idx:
            bigrams = set(self.get_bigrams(all_words[idx]))
            split_dict[query in bigrams].append(idx)

        if len(split_dict[True]) == 0 or len(split_dict[False]) == 0:
            split_dict[True] = my_words_idx
            split_dict[False] = []

        return query, split_dict

    def fit(self, all_words, my_words_idx, min_leaf_size, max_depth, fmt_str="    ", verbose=False):
        self.my_words_idx = my_words_idx

        if len(my_words_idx) <= min_leaf_size or self.depth >= max_depth:
            self.is_leaf = True
            self.process_leaf(self.my_words_idx)
            if verbose:
                print('█')
        else:
            self.is_leaf = False
            self.query, split_dict = self.process_node(all_words, my_words_idx, self.history)
            if self.query is None:
                self.is_leaf = True
                self.process_leaf(self.my_words_idx)
                if verbose:
                    print('█')
            else:
                if verbose:
                    print(self.query)

                for i, (response, split) in enumerate(split_dict.items()):
                    if verbose:
                        if i == len(split_dict) - 1:
                            print(fmt_str + "└───", end='')
                            fmt_str += "    "
                        else:
                            print(fmt_str + "├───", end='')
                            fmt_str += "│   "

                    self.children[response] = Node(depth=self.depth + 1, parent=self)
                    history = self.history.copy()
                    history.add(self.query)
                    self.children[response].history = history
                    self.children[response].fit(all_words, split, min_leaf_size, max_depth, fmt_str, verbose)

        # Clear history to save space after training
        self.clear_history()

    def clear_history(self):
        self.history = None
        if not self.is_leaf:
            for child in self.children.values():
                child.clear_history()


################################
# Non Editable Region Starting #
################################
def my_fit(words):
################################
#  Non Editable Region Ending  #
################################

    # Do not perform any file IO in your code
    # Use this method to train your model using the word list provided

    model = Tree(min_leaf_size=1, max_depth=None)
    model.fit(words, verbose=False)
    return model  # Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict(model, bigram_list):
################################
#  Non Editable Region Ending  #
################################

    # Do not perform any file IO in your code
    # Use this method to predict on a test bigram_list
    # Ensure that you return a list even if making a single guess
    guess_list = model.predict(bigram_list)
    return guess_list  # Return guess(es) as a list