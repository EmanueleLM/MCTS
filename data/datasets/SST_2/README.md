# Stanford Sentiment Treebank Binary Classification

Stanford Sentiment Treebank was introduced by [1], it is the first corpus with fully labeled parse trees, which could be normally used to capture linguistic features and predict the presented compositional semantic effect. It contains 5 sentiment classes: very negative, negative, neutral, positive and very positive; however we have filtered this to just two classes: positive, and negative.

The training data is split into phrases rather than sentences, following the approach of [2]. The training data has 117220 sentences, the validation set has 872 sentences and the test set has 1821 sentences.

#### References
[1] Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C.D., Ng, A. and Potts, C., 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing (pp. 1631-1642).

[2] https://gist.github.com/wpm/52758adbf506fd84cff3cdc7fc109aad