# RatingYelp - RECOMMENDATION SYSTEMS

Yelp data set
https://www.yelp.com/dataset/documentation/main
  
Setup, prerequisites and packages
We use the following libraries:
!pip install testfixtures
!pip install nltk
!pip install turicreate
!pip install twython
!pip install scikit-surprise


Step 1 : Prepare and Process Data
1.1 Train Data
This step cleans the train data and maps the fields: user_id, business id using dictionary data types to unique numeric IDs, by setting key-value mapping. The generated outputs are two files for the training process:
1)	X.to_csv(ID4Train.csv) : contains User_id, business_id, Stars 
2)	df.to_csv(ID4TrainFull.csv') : 
contains the original table with a change where user_id , business_id are replaced by their numeric id’s per a mapping record.

User dictionary and Business dictionary are exported to the following files:
output  
userdict.pickle
busdict.pickle

1.2 Test Data
This step cleans and maps the test data : Removes NA records and sets the user_id and business_id keys to its proper numeric IDs values - as defined for training data (step 1). In case the user_id or the business_id does not exist, we set its numeric ID to -1.
The generated outputs are two files for the test process, as has been done for training:
output
X.to_csv(ID2test.csv')
df.to_csv(ID2testFull.csv')

Step 2 : Sentiment Analysis Setup and Preparations

2.1 overview

In order to accomplish sentiment analysis, we performed pre-processing on both train and test data. We executed the function SentimentModelPreProcess on the following files:

ID4TrainFull.csv	
ID2testFull.csv
The function cleans the text fields and adds a sentiment, by using SentimentIntensityAnalyzer.

2.2 SentimentModelPreProcess

The following features were added per each row:
Review_clean : the outcome of the pre-process that was performed on the original text.
nb_words: number of words 
nb_chars: number of characters
sentiments metrics: sentiments.compound, sentiments.neg, sentiments.neu, sentiments.pos.
Our Sentiment Analyzing is based on VADER (Valence Aware Dictionary and sEntiment Reasoner. VADER is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.

 
VADER produces four sentiment metrics per review, which are listed below. The first 3: positive, neutral and negative, represent the proportion of the text that falls into those categories. The final metric, compound score, is the sum of all of the lexicon ratings.
In the following example, we show metrics of a review that was rated as 45% positive, 55% neutral and 0% negative. calculated compound score is  1.9 and 1.8, and has been standardized to range between -1 and 1. For our example, the review has a rating of 0.69, which is pretty strongly positive.


Sentiment metric	Value
Positive	0.45
Neutral	0.55
Negative	0.00
Compound	0.69

when SentimentModelPreProcess finishes, it generates two files :
trainsen.export_csv('gdrive/My Drive/reviewTrfull4.csv')
trainsen.export_csv('gdrive/My Drive/reviewTsfull2.csv')

Lets look on the sentiment result. what are the top negative reviews and top positive review?  
 

Step 3: Models Setup and adjustments

3.1 setup

In order to perform text embedding, we used the GENSIM (https://radimrehurek.com/gensim/) library to build two models: DBOW (DISTRIBUTED BAG OF WORDS) and DM (DISTRIBUTED MEMORY). Those models have been trained over all training dataset, using Doc2vec (as described on https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html).
Doc2vec is a common NLP tool for representing documents as a vector, and it is a generalization of the word2vec method.
3.2 Model 1: Distributed Bag of Words (DBOW)
DBOW is a doc2vec model, that is analogous to Skip-gram model in word2vec. The paragraph vectors are obtained by training a neural network, on the task of predicting a probability distribution of words in a paragraph, given a randomly sampled word from the paragraph.
function : trainBOW 
output: model_dbow.save(dbow_model_1.model')


3.3 Model 2: Distributed Memory (DM)
Distributed Memory (DM) behaves as a memory, which remembers what is missing from the current context, or as the topic of a given paragraph. While the word vectors represent the concept of a word - the document vector intends to represent the concept of a document.
function  : trainDM
output: model_dmm.save(dm_model_1.model')

3.4 Parameters Initialization
We initialize the models’ parameters with the following values:
●	100-dimensional vectors: we tried different sizes, including increasing value up to 200 -  but no significant improvement was observed.

●	cbow=0: setting skip-gram, which is equivalent to the paper’s ‘PV-DBOW’ mode, matched in gensim with dm=0.

●	DM  model -  one which averages context vectors (dm_mean) .We also tried the dm_contact, but no improvement was observed. 

●	A min_count=2 saves quite a bit of the models’ memory, discarding only words that appear in a single document.

●	Training data: The training has been done over all training data. 

●	Number of epochs: Each model iterates over the training corpus for 20 epochs.







 
Step 4 : Train, fit and predict


4.1 Process Overview

We defined and 6 models by combinations of 2 regression types and 3 embedding methods, of the form: {Liner, Logistic} and {DM, BOW, DM+BOW}.

We trained each variation, fitted the models, ran predictions and evaluated its performances by calculating RMSE metric. The splitflag variable indicates whether the prediction and evaluation will use validation set in addition to training and test set, or not.

4.2 Process Description

In particular, The SentimentModelTrain function reads the training data and uses it to build a tagged document.
Then, using the checkmodel function, it performs text embedding to each cleaned review.
Each review transformed into a vector with size of 100, according to the embedding modelflag (modelflag can be one of the following options: 0-DM,1-BOW ,2 DM+BOW).
Finally, the embedding vector is transformed into a data frame.
Each type of embedding model is being fitted and then saved in one of the following lists:

●	lr_models (for linear regression regression models) or
●	logreg_models (for logistic regression models).
●	
The prediction process is performed after models’ training, by the SentimentModelPredict function.

 
4.3 Results

Prediction results are listed below. The best model which has the lowest RMSE is marked in yellow.

4.3.1 Evaluation on Validation Set

 

4.3.2 Evaluation on Validation set and Test Set

 


 
Step 5 : SVD

5.1 Process Description

In this step we perform SVD training over the training dataset.
After reading training data, which is restored on ID4Train file, the function splits it into a training set and a validation set.

TrainBaseModel function is used to train SVD model.(the function TrainBaseModel is used for training and should be run only after running the block 1 (Look for example to train SVD and Example to run SVD on test) 

It returns four-learning parameters: bu,bi,qi,pu, based on the SVD algorithm.
We set TrainBaseModel to execute with 100 latent features, epochs limit is 20, reg value is 0.02 and learning rate is 0.005.
For every epoch, we evaluate performance on the validation set.
When training is finished, the final learnt parameters are dump to an external file that is located here: SVD.npz
We can run SVD on test data by calling to SVDontest function.

5.2 Output

We provide few examples of the SVD process output:

 
 
Step 6 : Ensemble of the Predictors 

6.1 Setting Up Ensemble
We built an ensemble model that performs training over a limited number of 500,000 examples (the limit was picked due to memory restrictions).
The ensemble model provides predictions that are based on SVD, BOW, DM and DM+BOW models predictions. In order to implement ensemble decision, we built a data frame that contains each models’ prediction per every record.
Then, a Linear Regression model is trained based on the models’ predictions, and calculates RMSE to evaluate its performance.
The data frame structure is as follow:
 
In order to perform the training process, we used the function TrainEnsamble. This function can be executed only after the basic models were trained, using the SentimentModelTrain function.
For testing predictions and evaluation we run the function TestEnsamble.
6.2 Results
Below we present calculated RMSE for validation and test sets, with and without ensemble model. As expected, ensemble model provides better results compared to simple models.
  
Step 7: Surprise library

We performed a benchmark on Surprise library and evaluated 3 models. The evaluation results are listed below: 
  


In order to evaluate our SVD implementation, we compared our model’s RMSE result to Surprise’s SVD RMSE result, with the same parameters and on the same test dataset. We can see that our RMSE result (1.271) is relatively close to Surprise’s SVD (1.2704) 😊

  


 
Step 8: SVDpp 
Function trainSVDPlus is used for training and should be run only after running the block 1 part which read the files and update the business per user and split to train , val 
We train due to time and memory only on 100000 rows the learned parameter saved to the following files
np.savez('SVDpp.npz', bu=bu, bi=bi,pu = pu,qi=qi,y=y)

The result on the validation: 1.359
The result on tests 1.4

