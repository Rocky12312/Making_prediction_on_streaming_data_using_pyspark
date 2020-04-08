#!/usr/bin/env python
# coding: utf-8

# Basically in this notebook our focus will be on building a classification model on twitter sentiment analysis data and using it to return result on streaming data of sentiments

# Baically before creating a new spark context do "sc.stop()" as the system have automatically initialized the SparkContext(maybe a Object?)when we type "pyspark" in the terminal,so to stop we did sc.stop() before creating a new one otherwise it will give an error showing "cannot run multiple spark context at a time".

# In[1]:


#sc.stop()


# In[2]:

import pyspark
from pyspark import SparkContext
from pyspark.sql.session import SparkSession


# In[3]:


#Initializing the spark session
sc=SparkContext(appName = "Sent_prec")
spark=SparkSession(sc)


# In[4]:


#Defining the schema type
#Defining the datatypes of the columns(setting id and label of integer type)
from pyspark.sql import types as tp
my_schema=tp.StructType([tp.StructField(name="id",dataType= tp.IntegerType(),nullable= True),
                         tp.StructField(name="label",dataType= tp.IntegerType(),nullable= True),
                         tp.StructField(name="tweet",dataType= tp.StringType(),nullable= True)])


# In[5]:


#Reading the data
data = spark.read.csv("train.csv",my_schema,header = True)


# In[6]:


#Printing the schema(our datatypes are being set by us to be int ,int and string)
data.printSchema()


# In[7]:


data.show()


# Now we need to preprocess and clean our tweet column as it contains punctuations,digits,stopwords etc

# In[8]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.sql.functions import udf, col, lower, regexp_replace

#Cleaning the text replacing every thing except [^a-zA-Z\\s] with a empty space
data_clean = data.select('id', (lower(regexp_replace('tweet', "[^a-zA-Z\\s]", "")).alias('tweet')))


# In[9]:


#Now building a model pipeline
stage1 = RegexTokenizer(inputCol= "tweet", outputCol= "tokens", pattern= "\\W")
stage2 = StopWordsRemover(inputCol= "tokens", outputCol= "filtered_words")
stage3 = Word2Vec(inputCol= "filtered_words", outputCol= "vector", vectorSize= 1000)


# Can also use the below code for text preprocessing and cleaning

# In[ ]:


#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer(language='english')

#Tokenizing the text
#tokenizer = Tokenizer(inputCol="tweet", outputCol="words_token")
#df_words_token = tokenizer.transform(data_clean).select('id', 'words_token')

#Removing stopwords
#remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')
#df_words_no_stopw = remover.transform(df_words_token).select('id', 'words_clean')

#Stemming the text(can also use WordNetLemmatizer or Porter Stemmer)
#from nltk.stem.snowball import SnowballStemmer
#Taking english words only
#stemmer = SnowballStemmer(language='english')
#stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
#df_stemmed = df_words_no_stopw.withColumn("words_stemmed", stemmer_udf("words_clean")).select('id', 'words_stemmed')
#
#Filter length word > 3(Not much important but again depends)
#filter_length_udf = udf(lambda row: [x for x in row if len(x) >= 3], ArrayType(StringType()))
#df_final_words = df_stemmed.withColumn('words', filter_length_udf(col('words_stemmed')))


# In[10]:


#Creating classification model(using gradient boosted tree classifier model)
from pyspark.ml.classification import LogisticRegression
model = LogisticRegression(featuresCol = "vector",labelCol = "label")


# In[11]:


#Setting up the pipeline
from pyspark.ml.pipeline import Pipeline
pipeline = Pipeline(stages = [stage1,stage2,stage3,model])
pip_fit = pipeline.fit(data)


# Now we will be getting the data in streams and we have to return results

# In[15]:


import sys
def rx_predictions(tweets):
    try:
        #Filtering the tweets and taking ones len(x)>0
        tweets = tweets.filter(lambda x:len(x)>0)
        #Creating a data frame containing tweet column
        data_tweet = tweets.map(lambda w:Row(tweet = w))
        #Spark data frame
        data_tweet_frame = spark.createDataFrame(data_tweet)
        #Transforming the data using pipeline
        pip_fit.transform(data_tweet_frame).select("tweet","prediction").show()
    except:
        print("No data")
from pyspark.streaming import StreamingContext
ssc = StreamingContext(sc,batchDuration = 3)
#Creating a Dstream that will connect the hostname,port
lines = ssc.socketTextStream(sys.argv[1],int(sys.argv[2]))
#Now splitting the tweets so that we can identify which set of words are from which tweet
words = lines.flatMap(lambda line:line.split("PICO"))
#Get the predicted sentiments from tweet received
words.foreachRDD(rx_predictions)
#Starting the computation
ssc.start()
#Waiting for the computation to terminate
ssc.awaitTermination()


# In[ ]:




