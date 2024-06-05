### Getting Started

1. Create a new anaconda environment with python version ``3.10``

    ```bash
    $ conda create -n next-word-prediction python=3.10
    ```

2. Install all dependencies

    ```bash
    $ pip install -r requirements.txt
    ```
3. Training script is located in `train.py`.

### Jupyter Notebook Files:
1. `Next Word Prediction (Approach 1).ipynb` is the same as `train.py` but in a Jupyter Notebook format.
2. `next-word-prediction (Approach 2 - Google USE).ipynb` is the second approach using Google Universal Sentence Encoder
to retrieve the embedding of the input text.
3. `Next Word Prediction (word2vec).ipynb` is the third approach using word2vec to retrieve the embedding.

### Difficulties

1. The most difficult part of this project was the **overfitting issue**. I tried using some popular
techniques to reduce overfitting, such as dropout, L2 regularization. However, none of them produce a significant
improvement.


### Reflections and future steps to consider

1. One major reason for the overfitting issue is that my input data and output data are too sparse.
Especially I set the labels to be one-hot-vectors. This makes the model hard to learn and generalize.

2. What I can do instead is to replace the sparse input vectors with **word embeddings**. Embeddings are dense,
low-dimensional representations of words that capture semantic similarities.

3. Similarly, I should also consider using embeddings as the target output as well. This can be achieved by
using an embedding layer for the output and training with a cosine similarity loss.

4. I could try to use a different model architecture, such as a transformer, to improve the performance.

5. If time permits, I am also intended to dockerize everything into a container for easy deployment.