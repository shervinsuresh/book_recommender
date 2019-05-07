
# coding: utf-8

# In[1]:


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Book_Tender",
    version="0.0.1",
    author="Group 6",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shervinsuresh/book_recommender",
    packages=setuptools.find_packages(),
    classifiers=[
        "book_tags",
        "books_db",
        "ratings",
        "tags",
        "to_read",
        "user_db",
        "user_recommendation",
    ],
)

