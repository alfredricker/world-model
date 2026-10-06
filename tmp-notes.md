1. Can you generate me a graphic describing the architecture? It could be several images. Some information to include is the encoder mapping tiles to tokens. 

### Questions
1. How might the encoder scale to real environments? Segmentation? Large interaction data to form unsupervised distinctions between objects?
2. What is the statistical method we use to determine the substate required to produce a given effect? Meaning how does our model know that it needs to face a red key tile and do the "pick up action" to possess a red key through its recall? 
3. What is a class? A:

### Comments
The essence of the architecture is the backwards planning. The model learns from experience training state and effect transitions. 

Capabilities:
* Doors with a switch, key, either, or both: 100% on new layouts
* New room sizes: 100%, 1-11% longer than shortest path
* Harder rooms (two locked doors, clutter): >99%
* Goals from example frames:  95-97%