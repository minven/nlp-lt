The main intention of this research is to study and learn natural language processing (NLP) principals  for Lithuanian language. It is interesting to analyze classical NLP methods and see how they work on it, so in this work I implemented text classification, topics extraction, search query and clustering ideas. Implementation details and futher information is stored at **paper/paper.pdf**

# Classification

Classification performance is measured using confusion matrix where rows are
true category and columns predicted category. Furthermore such approach reach above 90% recall and 90% precision.
![GitHub Logo](/visualizations/confussion_matrix.png)

# Topics extraction
Figure shows $6$ components with $10$ tokens for each component. From these results we can detect most important words and intuitively guess topic for each principal component. For example 3 principal component store information about  sports and music whereas 5 principal component store information about criminals.

Main results are presented below:
![GitHub Logo](/visualizations/main_term_components.png)