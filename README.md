# vuln_classification
Multi-class classification deep learning models using word embedding vectors to predict vulnerability categories on code snippets.

### Replication Package of our research work entitled "Vulnerability Classification on Source Code using Text Mining and Deep Learning Techniques"

To replicate the analysis and reproduce the results:
~~~
git clone https://github.com/certh-ai-and-softeng-group/vuln_classification.git
~~~
and navigate to the cloned repository.

The "data" directory contains the data required for training and evaluating the models.

The csv files in the repository are the pre-processed formats of the dataset (bag of words, sequences of tokens).

The jupyter notebook files (.ipynb) are python files, which perform the whole analysis. Specifically:


• data_preparation constructs the dataset

• train_embeddings trains custom word embedding vectors using either word2vec or fastText

• category_prediction contains the source code for employing word embedding algorithms (bow, word2vec, fastText, bert, codebert) and training Machine Learning models

• category_prediction_RF_averagedEmbeddings creates sentence-level vectors from the word embeddings (word2vec, fastText) and feeds them to ML models (Random Forest)

• category_prediction_sentenceBertRF extracts sentence-level contextual embeddings from transformer models and feeds them to ML models (Random Forest)

• finetuning_category_prediction_trainTestSplit performs fine-tuning of the CodeBERT model to the downstream task of vulnerability classification

• finetuning_category_prediction_trainTestSplit_Bert performs fine-tuning of the BERT model to the downstream task of vulnerability classification


### Acknowledgements

Special thanks to HuggingFace for providing the transformers libary

Special thanks to Gensim for providing word embedding models

Special thanks to VUDENC - Vulnerability Detection with Deep Learning on a Natural Codebase - for providing their dataset. For the dataset cite:

~~~
@article{wartschinski2022vudenc,
  title={VUDENC: vulnerability detection with deep learning on a natural codebase for Python},
  author={Wartschinski, Laura and Noller, Yannic and Vogel, Thomas and Kehrer, Timo and Grunske, Lars},
  journal={Information and Software Technology},
  volume={144},
  pages={106809},
  year={2022},
  publisher={Elsevier}
}
~~~


### Appendix

Evaluation results of the Random Forest classifier per text vectorizing method
| Vectorizing Method | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|--------------------|--------------|---------------|------------|--------------|
| Bag-of-Words       | 81.9         | 82.3          | 77.2       | 79.1         |
| Word2vec           | 71.6         | 76.2          | 64.3       | 68.0         |
| fastText           | 80.2         | 84.0          | 73.9       | 77.7         |
| BERT               | 76.9         | 86.6          | 69.4       | 75.1         |
| CodeBERT           | 80.7         | 87.6          | 72.9       | 78.0         |


Classification Performance of NLP models with prior knowledge of natural language versus programming language
| Vectorizing Method        | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|---------------------------|--------------|---------------|------------|--------------|
| pre-trained Word2vec      | 68.1         | 73.2          | 59.9       | 63.8         |
| re-trained Word2vec       | 71.6         | 76.2          | 64.3       | 68.0         |
| pre-trained fastText      | 74.9         | 78.0          | 68.0       | 71.5         |
| re-trained fastText       | 80.2         | 84.0          | 73.9       | 77.7         |
| pre-trained BERT          | 76.9         | 86.6          | 69.4       | 75.1         |
| pre-trained CodeBERT      | 80.7         | 87.6          | 72.9       | 78.0         |


Comparison of embeddings extraction and fine-tuning of Transformer models approaches
| Vectorizing Method | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|--------------------|--------------|---------------|------------|--------------|
| BERT + RF          | 76.9         | 86.6          | 69.4       | 75.1         |
| BERT fine-tuning  | 84.5         | 82.4          | 82.7       | 82.5         |
| CodeBERT + RF      | 80.7         | 87.6          | 72.9       | 78.0         |
| CodeBERT fine-tuning | 87.4       | 86.3          | 85.2       | 85.5         |



F1-score per category for the best examined models
| Category                | CodeBERT fine-tuning | BERT fine-tuning | BoW + RF | CodeBERT + RF | fastText + RF |
|-------------------------|----------------------|------------------|----------|---------------|---------------|
| SQL Injection           | 90                   | 86               | 89       | 82            | 86            |
| XSRF                    | 90                   | 91               | 86       | 86            | 80            |
| Open Redirect           | 75                   | 72               | 82       | 77            | 77            |
| XSS                     | 86                   | 87               | 77       | 67            | 73            |
| Remote Code Execution   | 81                   | 71               | 86       | 80            | 81            |
| Command Injection       | 91                   | 86               | 77       | 85            | 81            |
| Path Disclosure         | 87                   | 85               | 68       | 72            | 79            |


### License

[MIT License](https://github.com/certh-ai-and-softeng-group/vuln_classification/blob/main/LICENSE)

### Citation

I. Kalouptsoglou, M. Siavvas, A. Ampatzoglou, D. Kehagias, A. Chatzigeorgiou, Vulnerability classification on source code using text mining and deep learning techniques, in: 24th IEEE International Conference on Software Quality, Reliability, and Security (QRS’ 24), 2024
