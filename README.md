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

![](example_facts.gif)

#### Intention
We aim to create a tool which provides interesting facts that are highly personalized to the subjects that students write about on their papers. These facts will motivate students to do more in-depth research about topics that would complement their essays.


#### Impact
In order to improve their writing skills, students should include interesting facts and details about their subject matter. However, it is important for these facts to fit well in their essays and they shouldn't be added without considering the context of their writing. In order to motivate students to include facts in their writing and improve their researching skills, our tool will provide interesting facts that are specifically suited for the essays they write. We hope our tool will motivate students to do further research about their topics and give them ideas about further topics to research that would fit well with their essays.

### Tools and packages used
    NLTK: Word and sentence tokenizer, along with POS tagger.
    Spacy: Wrapper for Word2Vec vectors.
    Gensim: NLP tools.
    Numpy: Numerical computation.
    Json/Pickle: Efficient data storage.


### Data
You may download the Simple English Wikipedia corpus in the following website: https://dumps.wikimedia.org/simplewiki/latest/. 
The file we used is called: `simplewik-latest-pages-articles.xml.bz2`.