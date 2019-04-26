
# coding: utf-8

# In[15]:


from appJar import gui 
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
list_len=25;
books = pd.read_csv('books_db.csv', encoding = "ISO-8859-1")
ratings = pd.read_csv('ratings.csv', encoding = "ISO-8859-1")
#ratings.head()
book_tags = pd.read_csv('book_tags.csv', encoding = "ISO-8859-1")
#book_tags.head()
tags = pd.read_csv('tags.csv')
#tags.tail()
tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
#tags_join_DF.head()
to_read = pd.read_csv('to_read.csv')
#to_read.head()
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(books['authors'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Build a 1-dimensional array with book titles
titles = books['title']
indices = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of book authors
def title_recommendations(authors):
    idx = indices[authors]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(list_len+1)]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')
tf1 = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix1 = tf1.fit_transform(books_with_tags['tag_name'].head(10000))
cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)

# Build a 1-dimensional array with book titles
titles1 = books['title']
indices1 = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def tags_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim1[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]
temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
temp_df.head()
books = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')
books['corpus'] = (pd.Series(books[['authors', 'tag_name']]
                .fillna('')
                .values.tolist()
                ).str.join(' '))
tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

# Build a 1-dimensional array with book titles
titles = books['title']
indices = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def corpus_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

corpus_recommendations("The Hobbit")


# In[16]:


from appJar import gui
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
user = pd.read_csv('user_db.csv',encoding = "ISO-8859-1")
user_index=user['User_Index'].max()
print(user)
def press2(btn):
    if btn == "Author Names":
        app.infoBox("Author Names list","Suzanne Collins, J.K. Rowling, Stephenie Meyer, Harper Lee, F. Scott Fitzgerald, John Green, J.R.R. Tolkien, J.D. Salinger, Dan Brown, Jane Austen, Khaled Hosseini, Veronica Roth, George Orwell")
def press1(btn):
    if btn == "Night Mode":
        app.infoBox("NONO","unless you want to see all black, do not click")
def press(button):
    if button == "Cancel":
            app.stop()
    else:
        user_ID=app.getEntry("user_ID")
        user_index=((user['User_Index'].max())+1)
        name= app.getEntry("Name")
        gender=app.getEntry("Gender (M or F)")
        age=app.getEntry("Age (Years)")
        RH=app.getEntry("Reading Habit (Hours per Week)")
        GP=app.getEntry("Genre Preference")
        FA = app.getEntry("Favorite Author")
        BR=app.getEntry("Previous Book rating of Favorite Author (1-5)")
        user.loc[user_index] = [user_index,user_ID,name,gender,age,RH,GP,FA,BR]
        app.infoBox("close the whole window","press the 'x' button to completely close this messsage and where you entered your user info, as it lets you resume running the rest of the code")
        curr_user=user_ID
        print(user)
        

app=gui()
app.addButtons(["Night Mode"], press1)
app.addLabel("l2", "User Information")
app.addLabelEntry("user_ID")
app.addLabelEntry("Name")
app.addLabelEntry("Gender (M or F)")
app.addLabelEntry("Age (Years)")
app.addLabelEntry("Reading Habit (Hours per Week)")
app.addLabelEntry("Genre Preference")
app.addLabelEntry("Favorite Author")
app.addButtons(["Author Names"],press2)
app.addLabelEntry("Previous Book rating of Favorite Author (1-5)")
app.addButtons(["Enter", "Abort"], press)
app.go()


# In[17]:


curr_data=(user.loc[user['User_Index'] == (user_index+1)])
print(curr_data)
author = curr_data['Author_Preference'].values[0]
print(author)
author_fav=(books.loc[books['authors'] == author])
author_fav[author_fav['ratings_5']==author_fav['ratings_5'].max()]
user_title = author_fav['title'].values[0]
print(user_title)
recommendations=title_recommendations(user_title)


# In[22]:


def press2(btn):
    if btn == "Night Mode":
        rec.infoBox("NONO","unless you want to see all black, do not click")
rec=gui()
rec.addButtons(["Night Mode"], press2)
rec.addLabel("l3", "Recommendations")
rec.addMessage("mess",recommendations)
recommendations.to_csv('user_recommendation.csv')
rec.go()

