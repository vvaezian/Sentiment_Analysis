### To-Do
- [ ] All-caps tweets are usually negative. But converting to lower-case we lose this info.
- [x] Multi_class Logistic regression
- [ ] Explore misclassfications
- [x] Correct the `Compress` function
- [x] Create simple baseline rule_based model
- [ ] Reconsider applying Lemmatization, Stemming and POS tagging to the new cleaned dataset
- [x] Test performance with cleaned stopwords

### Rule_based
- Pos: http://ptrckprry.com/course/ssd/data/positive-words.txt
- Neg: http://ptrckprry.com/course/ssd/data/negative-words.txt

### Remarks
- "A lemmatizer needs a part of speech tag to work correctly. This is usually inferred using the `pos_tag` nltk function before tokenization."
- Consider using stemming before or after lemmatization
- [The paper](https://s3.amazonaws.com/academia.edu.documents/34632156/Twitter_Sentiment_Classification_using_Distant_Supervision.pdf?response-content-disposition=inline%3B%20filename%3DTwitter_Sentiment_Classification_using_D.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWOWYYGZ2Y53UL3A%2F20190620%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20190620T213431Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=99677c7040f123dec6fff770d493bda4218015f4c24fd3d8d676a8eef18c55b5) that produced the dataset. It labelled tweets based on :-) and :-( symbols. Its best accuracy is 83%. 

### Services
- Doing sentiment analysis on brands
- Exploring the correlation between a brand name and demographics, weather, ...

### Research Topics
- Look into LDA ([Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation))

### Tools and Libraries to Explore
- [Spacy](https://en.wikipedia.org/wiki/SpaCy)
