# Authorship Attribution Helper

Tools to make authorship attribution faster.

## Dependencies

Libraries required:
```
numpy
sklearn
joblib
keras
matplotlib
```

## How to Use
Currently available classifiers are SVM and CNN.
To use SVM:
```
from classifiers import SVM
svm = SVM(C=1, analyzer="word", ngram_range=(1,2))
```
SVM takes 3 arguments.
- C: The original C value to use.
- analyzer: The type of the analyzer. Either "word" or "char".
- ngram_range: The range of ngrams to use.

To use CNN:
```
from classifeirs import CNN
cnn = CNN(max_length=500, vector_size=150, minibatch_size=50, ngram=4, epochs=10) 
```
CNN takes 5 arguments.
- max_length: How many input nodes.
- vector_size: the embedding dimensions.
- minibatch_size: The size of the minibatch to use.
- ngram: Which ngram size to use. Currently can only take one number, not a range.
- epochs: How many epochs to train.
- 

After the classifer has been imported and created, the next step is to create a Handler to handle the rest.
```
from wrapper import Handler
attributer = Handler(train_data=train_data, test_data=test_data, positive_class=["Author 1"], split_size=500, split_type="char", data_type="book_split", classifier=svm, threads=10, iteration=-1)
```
Here the Handler takes in the data and multiple different arguments.
- train_data: The train data. Needs to be a JSON dictionary, where the key is the name of the author, and the value is a dictionary of books / manuscripts from the author. e.g. train_data = {"auth1": {"book1": "book text"}}
- test_data: The test data. Same format as train_data.
- positive_class: List of authors to consider the positive class.
- split_size: The size of splits to do, if any. -1 if no splits.
- split_type: Whether to split by charcters or words.
- data_type: How to do the splitting. "book": No splitting, one book is one training sample. "single": Books are split into parts and they are classified independent of each other. "book_split": Books are split into parts, but they are classified together.
- classifier: The classifier to use. SVM or CNN.
- threads: How many threads to use with SVM. CNN uses all by default.
- iteration: Current iteration. If -1, do everything in one go, but otherwise only perform one classification. Useful if classifying in parallel.

Handler has multiple functions that can be called to perform different tasks.
```
attributer.optimize_C([2,4,6,8])
```
Optimizes the SVM C value using Leave-one-out Cross Validation. C values to test are give as argument.
```
attributer.cross_validate()
```
Performs Cross Validation to get values for the train data samples as well.

```
attributer.attribute.test_data()
```
Attributes the test data samples.

```
attributer.print_results(normalize=True)
```
Print results. Normalizing scales the values to be between 1 and -1, where 0 is the threshold.

```
attributer.plot_values(scale=True, title="Title")
```
Plots values. Scale scales the values. Title for the plot.

```
attributer.get_best_features()
```
Extract best features (Requires SVM).






