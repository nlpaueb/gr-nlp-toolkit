# gr-nlp-toolkit

A Transformer-based natural language processing toolkit for (modern) Greek. The toolkit has state-of-the art performance in Greek and supports named entity recognition, part-of-speech tagging, morphological tagging, as well as dependency parsing. For more information, please consult the following theses:

C. Dikonimaki, "A Transformer-based natural language processing toolkit for Greek -- Part of speech tagging and dependency parsing", BSc thesis, Department of Informatics, Athens University of Economics and Business, 2021. http://nlp.cs.aueb.gr/theses/bsc_thesis_dikonimaki.pdf

N. Smyrnioudis, "A Transformer-based natural language processing toolkit for Greek -- Named entity recognition and multi-task learning", BSc thesis, Department of Informatics, Athens University of Economics and Business, 2021. http://nlp.cs.aueb.gr/theses/smyrnioudis_bsc_thesis.pdf

### Performance Comparison

We compared our toolkit's accuracy to Stanza on the Greek test corpus of Universal Dependencies. (20 sentences were 
excluded because there were tokenization mismatches)

Dependency Parsing Results:

| Metric      | Stanza | gr-nlp-toolkit |
| ----------- | ----------- | -----------| 
| UAS      | 0.91       | 0.94|
| LAS   | 0.88       | 0.92|

Part-of-Speech tagging results

| Metric      | Stanza | gr-nlp-toolkit |
| ----------- | ----------- | -----------| 
| micro-f1      | 0.98       | 0.98|
| macro-f1   | 0.96       | 0.97|


## Installation

You can install the toolkit by executing the following in the command line:
```sh
pip install gr-nlp-toolkit
```

## Usage

To use the toolkit first initialize a Pipeline specifying which processors you need. Each processor 
annotates the text with a specific task's annotations.

- To obtain Part-of-Speech and Morphological Tagging annotations add the `pos` processor
- To obtain Named Entity Recognition annotations add the `ner` processor
- To obtain Dependency Parsing annotations add the `dp` processor

```python
from gr_nlp_toolkit import Pipeline
nlp = Pipeline("pos,ner,dp") # Use ner,pos,dp processors
# nlp = Pipeline("ner,dp") # Use only ner and dp processors
```

The first time you use a processor, that processors data files are cached in the .cache folder of 
your home directory so you will not have to download them again.

## Generating the annotations

After creating the pipeline you can annotate a text by calling the pipeline's `__call__` method.

```python
doc = nlp('Η Ιταλία κέρδισε την Αγγλία στον τελικό του Euro 2020')
```
A `Document` object is then created and is annotated. The original text is tokenized 
and split to tokens

## Accessing the annotations

The following code explains how you can access the annotations generated by the toolkit.

```python
for token in doc.tokens:
  token.text # the text of the token
  
  token.ner # the named entity label in IOBES encoding : str
  
  token.upos # the UPOS tag of the token
  token.feats # the morphological features for the token
  
  token.head # the head of the token
  token.deprel # the dependency relation between the current token and its head
```

`token.ner` is set by the `ner` processor, `token.upos` and `token.feats` are set by the `pos` processor
and `token.head` and `token.deprel` are set by the `dp` processor.

A small detail is that to get the `Token` object that is the head of another token you need to access
`doc.tokens[head-1]`. The reason for this is that the enumeration of the tokens starts from 1 and when the
field `token.head` is set to 0, that means the token is the root of the word.
