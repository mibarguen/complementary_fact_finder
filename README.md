##### Nathaniel Andre and Mateo Ibarguen
### Project 1 Proposal: Fact-Retrieval

### Instructions
> Before running our project, make sure you have the following files under the data directory. If you don't have these files, find the instructions below to install them. 
- articles.pkl
- articles_keywords.pkl
- student_keywords.json

```bash
python3 -m src.main 
```

You will get the following results: 
```
Initializing model...
Write 'exit' to escape program. Type --facts after you have written your essay to get results.
Copy-paste your essay here:    
```
Write your essay on the prompt. 
After you have finished, press enter and type --facts.
```buildoutcfg
Essay about dinosaurs...
--facts
```

```buildoutcfg
Finding Facts... 
Here are some facts we think would make your essay more detailed. 
Topic:  Dinosaur
William Buckland Gideon Mantell and Richard Owen were scientists who saw these bones were a special group of animals. 

Topic:  Pterosaur
The first fossils occur in the Upper Triassic and the group continues until the KT extinction event at the end of the Cretaceous 220 to 655 million years ago. 

Topic:  Tetrapod
The earliest tetrapods evolved from the Sarcopterygii or lobefinned fish into airbreathing amphibians perhaps in the Upper Devonian period. 

Topic:  Dromaeosauridae
General plan of saurischian hip bonesModel of the foot bones of a typical dromaeosaurDromaeosaurs are a family of birdlike theropod dinosaurs which included the famous Velociraptor. 

Topic:  Tyrannosaurus
Houston Museum of Natural History. Skeletons mounted as if in copulation Jurassic Museum of Asturias SpainThe Field Museum in Chicago is the most complete Tyrannosaurus skeletonCharles at the Portobello Fossil MuseumTyrannosaurus and human size difference showing many specimensTyrannosaurus was a large predatory dinosaur from the Upper Cretaceous 68 to 66 million years ago. 

Topic:  Origin of birds
Thomas Henry Huxley Darwins bulldog who was a comparative anatomist made a study of this nearly 150 years ago. 

Topic:  Allosaurus
This might be evidence of pack behavior or just the result of lone individuals feeding on the same carcass. 

Topic:  Triceratops
horridus skeleton mounted with modern limbposture Natural History Museum of Los Angeles CountyTyrannosaurus and Triceratops models in an abandoned hotel1949Triceratops compared in size with a humanTriceratops was a huge herbivorous ceratopsid dinosaur from the late Cretaceous. 
```

After you have received your facts, you may write another essay or escape the program by typing `exit`.


#### Intention
We aim to create a tool which provides interesting facts that are highly personalized to the subjects that students write about on their papers. These facts will motivate students to do more in-depth research about topics that would complement their essays.


#### Impact
In order to improve their writing skills, students should include interesting facts and details about their subject matter. However, it is important for these facts to fit well in their essays and they shouldn't be added without considering the context of their writing. In order to motivate students to include facts in their writing and improve their researching skills, our tool will provide interesting facts that are specifically suited for the essays they write. We hope our tool will motivate students to do further research about their topics and give them ideas about further topics to research that would fit well with their essays.

#### How do we seek to accomplish this?
We will implement an unsupervised system that determines the similarity between blocks of text written by students and information that’s provided on Wikipedia. Based on previous work, we plan on exploring paragraph-vectors and simple word-vector averaging, along with other techniques to determine the most effective method to compare text similarity for this particular problem. 

We have a few implementation ideas for determining which text in Wikipedia articles best relate to the student’s main topics. We are exploring exhaustive search methods where we look through all possible information to find the most relevant Wikipedia content. We are also considering simplifying this process by using article titles to rule out Wikipedia information that is nowhere close to being similar to the student’s main topic. 

Another idea is using clustering to define groups of text that revolve around broader encompassing topics, and using this to further decrease the search complexity. In order to obtain the topics that form a student’s essay, we plan on using Latent Dirichlet Allocation. 

In terms of determining the similarity between blocks of text, we plan on using standard methods to compare the distance of the vector representations in multidimensional space, such as cosine-similarity and euclidean distance. Once the text that is similar to a student’s topic or topics is found, we will output that text.

##### Model Evaluation
We can evaluate how well our model is performing at providing facts relevant to the student essays in the following steps:
  1) Use the model we are evaluating to obtain two sentences from Wikipedia that it thought would be relevant to the student's essay.
  2) Randomly obtain a sentence from the Wikipedia corpus without using our model. 
  3) If the first two sentences are more similar to each other than to the random sentence, then we would consider the outcome of our model a "success." On the other hand, if the first two sentences are more different to each other than to the randomly obtained sentence, then we would consider the outcome to be an error. 
  4) We record the amount of errors we obtain over the test set of student essays. 

#### What data is necessary and how will you obtain it?
In order to get information about a wide range of topics, we got access to a dump of simple english Wikipedia articles. This dump includes the articles and main bodies of text for a wide range of articles, written to be easily understood by a younger audience. These articles are necessary so that we can provide students with easy to digest information that is related to their topic of choice. It was obtained on the Wikipedia website.
