# no-package


- Jupyter Notebook code for LDA model from scratch
- Huge thanks to [this blog](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/07/09/lda/)
<br/><br/>

## About the Code
- TopicModel class
    - Creates the topic model 
    - Parameters: document(type list of list(str)), num_topics, alpha, beta
    - run function takes number of iterations as parameter
      - Currently set to 1000
    - #TODO: import a better data set
- Results
    - Results of topic modeling is saved here
    - Document topic counts: Shows topics of documents
      - Document 0: (2,4) indicates Topic 2 appeared 4 times in document 0
      - How does it know a certain topic appeared? 
        - Words in the document have topics designated to them (look at the next line)
    - Topic word counts: Shows words of topics
      - Topic 0: ("Python", 4) indicates the word "Python" appeared 4 times for topic 0
<br/><br/>

## Limitations
- How do you know if a topic is well-made?
  - Topics are just indexes
  - You must look at the words which make up the topics and decide what to call the topic
  - Some methods are suggested(perplexity, coherence score, correlation, etc.)