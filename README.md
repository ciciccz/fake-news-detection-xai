# fake-news-detection-xai

This is a learning journal for MRU's BCIS Senior Project. 
Supervisor: Eric Chalmer

Bag of words**
Doesn't care about order words. Only counts the appearances of unique words in document(s)
1. **Creating a Dictionary of Words**:
    
    - First, you go through all the text in your training set (a collection of documents or sentences you're using to teach your machine learning model).
    - You make a list (or dictionary) of all unique words found in these texts. Each word gets a unique number called an "id" (identifier). For example, "apple" might be 1, "banana" might be 2, and so on.
2. **Counting Words in Each Document**:
    
    - Now, for each document (or piece of text) in your training set, you count how many times each word appears.
    - You create a sort of spreadsheet (let's call it X). Each row in X represents a different document, and each column represents a different word from your dictionary.
3. **Filling in the Spreadsheet (X)**:
    
    - In this spreadsheet, you fill in the counts. For example, if document #1 has the word "apple" three times, and "apple" is word #1 in your dictionary, then you put the number 3 in column 1 of row 1 in X.
    - You do this for every word in every document. If a word doesn't appear in a document, you put a 0 in that cell.
4. **Understanding n_features**:
    
    - In this context, "n_features" refers to the number of different words in your dictionary. Since it includes every unique word from all the documents, this number can be very large, often more than 100,000.

**TfidfVectorizer**
```
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char',

                             use_idf=False)
```
ngram_range will examine solo, pair, or triple char. not using the idf method of assigning higher weights based on lower frequency.


**Experimenting with XGBoost**

1. Data is downloaded from HuggingFace, it's already been split up into test, train and valid, in tsv format. 
	1. to read tsv, I used `data = pd.read_csv('./datasets/LIAR/train.tsv', sep='\t', header=None)
	2. The original file contained labels that are from pants-fire (blatant lie) to barely-true, to true statements. These were mapped to integer values 0 - 5
	3. The labels are replaced with the integer values
2. Attempted to use a pipeline that consisted tfidvecrtorizer() and XGBClassifier(softprob). With default params, the model yielded around 20% accuracy, which is bad. 
3. Learned to use GridSearchCV to optimize the params for the pipeline. Decided to try optimizing for max, min_df and ngram_range. The fitting took 15 minutes and still yielded around 20% accuracy. 
4. Decided to try using word2vec and see its performance taking into account semantics, instead of using a bag of words model.
5. Downloaded word2vec and fitted XGBClassifier, still achieving sub 0.3 score.
6. Ran GridSearchCV for optimizing XGB params. Let ran for 2 days, did not achieve a score higher than 0.3.
7. Currently researching models such as BERT and investigating its compatibility with XAI framworks.
