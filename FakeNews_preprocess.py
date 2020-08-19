#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#data=pd.read_csv("D:/Fake News/TICNN/all_data.csv")


# In[2]:


import numpy as np


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data['title'].isnull().sum()


# In[6]:


data['type'].isnull().sum()


# In[7]:


data['text'].isnull().sum()


# In[8]:


data1=data[['text', 'title_len','anger','anticipation','disgust','fear','joy','sadness','surprise','trust','negative','positive','type']]


# In[9]:


data1.head()


# In[10]:


export_csv = data1.to_csv (r'C:\Users\sivay\OneDrive\Desktop\export_dataframe.csv', index = None, header=True)


# In[11]:


data1.shape


# In[12]:


data1['title_len'].isnull().sum()


# In[13]:


data1['anger'].isnull().sum()


# In[14]:


data1['anticipation'].isnull().sum()


# In[15]:


data1['disgust'].isnull().sum()


# In[16]:


data1['fear'].isnull().sum()


# In[17]:


data1['joy'].isnull().sum()


# In[18]:


data1['sadness'].isnull().sum()


# In[19]:


data1['surprise'].isnull().sum()


# In[20]:


data1['trust'].isnull().sum()


# In[21]:


data1['negative'].isnull().sum()


# In[22]:


data1['positive'].isnull().sum()


# In[23]:


data1.head()


# In[24]:


from nltk.corpus import stopwords
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[25]:


conda install -c conda-forge python-levenshtein


# In[133]:


from Levenshtein import distance


# In[26]:


from collections import Counter
from itertools import chain
from nltk import word_tokenize, pos_tag
nltk.download('all')


# In[27]:


def remove_count_urls(Sentence):
    Sentence_url_removed = re.subn('https?://[A-Za-z0-9./]+','',Sentence)
    Sentence = Sentence_url_removed[0]
    return Sentence


# In[28]:


def remove_stopwords(Sentence_tokens):
    stop_words = stopwords.words('english')
    Sentence_tokens = [word for word in Sentence_tokens if word not in (stop_words)]
    return Sentence_tokens


# In[29]:


def lemmatization(Sentence_tokens):
    lem = WordNetLemmatizer()
    Sentence_tokens = [lem.lemmatize(token,get_wordnet_pos(token)) for token in Sentence_tokens]
    return Sentence_tokens


# In[30]:


def remove_count_user_mentions(Sentence):
    Sentence_mentions_removed = re.subn(r'@[A-Za-z0-9]+','',Sentence)
    Sentence = Sentence_mentions_removed[0]
    no_user_mentions = Sentence_mentions_removed[1]
    return Sentence,no_user_mentions


# In[31]:


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# In[32]:


def remove_count_hashtags(Sentence):
    no_hashtags = len({tag.strip("#") for tag in Sentence.split() if tag.startswith("#")})
    Sentence = re.sub("[^a-zA-Z]", " ",Sentence)
    return Sentence,no_hashtags


# In[33]:


def remove_count_urls(Sentence):
    Sentence_url_removed = re.subn('https?://[A-Za-z0-9./]+','',Sentence)
    Sentence = Sentence_url_removed[0]
    return Sentence


# In[34]:


def preprocessing(Sentence):
    preprocessed = []
    Sentence = Sentence.lower()
       
    #removing URLs
    Sentence = remove_count_urls(Sentence)
       
    #Remove unicode kind of characters \x99s
    Sentence = Sentence.encode('ascii', 'ignore').decode("utf-8")
    
    #Compute number of characters in a Sentence
    #no_chars = len(Sentence) - Sentence.count(' ')
    #preprocessed.append(no_chars)
    
    #remove short words
    Sentence = " ".join(word for word in Sentence.split() if len(word)>2)
    
    #remove special characters i.e, retain only characters
    Sentence = re.sub('[^A-Za-z ]+', '',Sentence)
    
    #Generate tokens for Stop word removal and Stemming
    tok = WordPunctTokenizer()
    Sentence_tokens = tok.tokenize(Sentence)
    no_words = len(Sentence_tokens)
    #preprocessed.append(no_words)
    
    #Stop Word Removal
    Sentence_tokens = remove_stopwords(Sentence_tokens)
    
    #Removing hash tags
    Sentence=remove_count_hashtags(Sentence)
    
    #Removing User Mentions
    #Sentence=remove_count_user_mentions(Sentence)
    
    #POS Tagging
    #Lemmatization on POS Tag is better than Lemmatization only or Stemming
    Sentence_tokens = lemmatization(Sentence_tokens)
        
    #Convert tokens to string and Remove unnecessary white spaces 
    Sentence = " ".join(Sentence_tokens).strip()
    preprocessed.append(Sentence)
    
    #create a list of required attributes obtained after preprocessing
    return (preprocessed)


# In[35]:


def clean_and_get_features(data):
    print('Cleaning the Sentences, Stop Word Removal, Lemmatization, Feature Extraction ')
    Sentences = data['content']
    Sentences = Sentences.astype(str)
    preprocessed = Sentences.apply(preprocessing)
    preprocessed_df = pd.DataFrame(columns=['no_chars','no_words','Sentence'])
    preprocessed_df['no_chars']=preprocessed_df['no_chars'].astype(int)
    preprocessed_df['no_words']=preprocessed_df['no_words'].astype(int)
    
    for i in range(len(preprocessed)):
        preprocessed_df.loc[i] = preprocessed[i]

    cleaned_Sentence = preprocessed_df['Sentence'] 
    return preprocessed_df, cleaned_Sentence


# In[36]:


s=data1["text"]


# In[146]:


preprocessingtry=preprocessing(s[0])


# In[147]:


for x in range(0,len(s)):
    s[x]=preprocessing(s[x])


# In[44]:


data1["text"]=s


# In[41]:


export1_csv = data1.to_csv (r'C:\Users\sivay\OneDrive\Desktop\export1_dataframe.csv', index = None, header=True)


# In[25]:


pro_data=pd.read_csv("D:/Fake News/TICNN/export1_dataframe.csv")
pro_data.head()


# In[35]:


def get_pos_tag_count_matrix(data):
    tokens= pro_data.text.apply(nltk.word_tokenize)
    pos_tag  = tokens.apply(nltk.pos_tag)
    noun_count = pos_tag.apply(NounCounter).str.len()
    pronoun_count = pos_tag.apply(PRPCounter).str.len()
    verb_count = pos_tag.apply(VerbCounter).str.len()
    adj_count = pos_tag.apply(AdjCounter).str.len()
    adverb_count = pos_tag.apply(AdVCounter).str.len()
    pattern = pos_tag.apply(Pattern)
    pos_count_df = pd.concat([noun_count,pronoun_count,verb_count,adj_count,adverb_count],axis=1)
    pos_count_df.columns=['noun_count','pronoun_count','verb_count','adj_count','adverb_count']
    return(pos_count_df)

#%%
def NounCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("NN"):
            nouns.append(word)
    return nouns      
#%%
def PRPCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("PRP"):
            nouns.append(word)
    return nouns      
#%%
def VerbCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("VB"):
            nouns.append(word)
    return nouns      
#%%
def AdjCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("JJ"):
            nouns.append(word)
    return nouns      
#%%
def AdVCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("RB"):
            nouns.append(word)
    return nouns      
#%%
def Pattern(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith('NN'):
            nouns.append("sw")
        elif pos in ["VB", "VBD", "VBG", "VBN" , "VBP"]: 
            nouns.append('sw')
        elif pos.startswith("JJ"): 
            nouns.append('sw')
        elif pos.startswith("RB"): 
            nouns.append('sw')
        else:
            nouns.append('cw')   
  
    return nouns      
#%%


# In[36]:


pos=get_pos_tag_count_matrix(pro_data["text"])


# In[37]:


pos


# In[38]:


type(pos)


# In[39]:


pos.to_csv("output.csv")  


# In[40]:


type(TFIDF_df)


# In[41]:


TFIDF_df.to_csv("output1.csv")  


# In[42]:


for x in range(0, len(pro_data['text'])):
    pro_data["text"].iloc[x]= pro_data['text'].iloc[x].strip('[]')


# In[43]:


export2_csv = pro_data.to_csv (r'C:\Users\sivay\OneDrive\Desktop\export2_dataframe.csv', index = None, header=True)


# In[44]:


type(pro_data['text'])


# In[45]:


p = pro_data.head()


# In[46]:


pro_data.shape


# In[47]:


pro_data['words'] = pro_data.text.apply(lambda l : len(l.split()) )
pro_data['charaters'] = pro_data.text.apply(lambda l : len(l))


# In[48]:


words=pro_data.text.apply(lambda l : len(l.split()) )
characters=pro_data.text.apply(lambda l : len(l))


# In[49]:


pro_data.head()


# In[50]:


output = pd.read_excel(r'D:\\Fake News\\TICNN\\output.xlsx')


# In[51]:


output.head()


# In[52]:


emotions=pd.read_csv("emotions.csv")


# In[53]:


TYPE=data[["type"]]
TYPE=TYPE.rename(columns={"type": "TYPE"})


# In[54]:


TYPE=TYPE.replace('real',0)
TYPE=TYPE.replace('fake',1)
TYPE.head()


# In[55]:


res1 = pd.concat([emotions, pos], axis=1)


# In[57]:


res1 = pd.concat([res1,words], axis=1)


# In[58]:


res1 = pd.concat([res1,characters], axis=1)


# In[60]:


res1=pd.concat([res1,TYPE],axis=1)


# In[61]:


res1.shape


# In[62]:


res1 = res1.loc[:, ~res1.columns.str.contains('^Unnamed')]


# In[64]:


res1.to_csv("final.csv")

