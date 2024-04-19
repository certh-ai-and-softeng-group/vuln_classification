# vuln_classification
Multi-class classification deep learning models using word embedding vectors to predict vulnerability categories on code snippets.

### Replication Package of our research work entitled "Vulnerability Classification on Source Code using Text Mining and Deep Learning Techniques"

To replicate the analysis and reproduce the results:
~~~
git clone https://github.com/iliaskaloup/vuln_classification.git
~~~
and navigate to the cloned repository.

The "data" directory contains the data required for training and evaluating the models.

The csv files in the repository are the pre-processed formats of the dataset (bag of words, sequences of tokens).

The jupyter notebook files (.ipynb) are python files, which perform the whole analysis. 


Specifically:
• data_preparation constructs the dataset

• train_embeddings trains custom word embedding vectors using either word2vec or fastText

• category_prediction contains the source code for employing and training word embedding algorithms (bag of words, word2vec, fastText, bert, codebert), Machine Learning and Deep Learning models

• category_prediction_RF_averagedEmbeddings creates sentence-level vectors from the word embeddings (word2vec, fastText) and feeds them to ML models (Random Forest)

• category_prediction_sentenceBertRF extracts sentence-level contextual embeddings from transformer models and feeds them to ML models (Random Forest)

• finetuning_category_prediction_trainTestSplit performs fine-tuning of the CodeBERT model to the downstream task of vulnerability classification

• finetuning_category_prediction_trainTestSplit_Bert performs fine-tuning of the BERT model to the downstream task of vulnerability classification


### Acknowledgements

Special thanks to HuggingFace for providing the transformers libary
Special thanks to Gensim for providing word embeddings models
Special thanks to VUDENC - Vulnerability Detection with Deep Learning on a Natural Codebase for providing their dataset.

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

F1-score per category for different examined models
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

[MIT License](https://github.com/iliaskaloup/vuln_classification/blob/main/LICENSE)

### Citation

To be filled...
