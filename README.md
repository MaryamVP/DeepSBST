# DeepSBST

## Deep Search Based Software Testing 

This work is an extension of testRNN [1] that we applied on DeepCodeSearch [2]. In the following, I would explain the testRNN method, as well as DeepCodeSearch. Furthermore, I go through methodology details and the current results. In the last section, I would explain what we expect from our collaboration.

### 1.	The testRNN Method
As far as we know, testRNN introduces the first coverage-guided testing tool, called testRNN, for the verification and validation of long short-term memory networks (LSTMs). The tool testRNN outputs a test suite that includes some extra tests. The user can ask for a minimum test suite size to optimize toward a higher test coverage. The tool implements a generic mutation-based test case generation method, and it empirically evaluates the robustness which defines as network coverage. 
A brief description to their tool can be explained as below:

Given an input of a LSTM Model (text or image), testRNN will generate a test suite, together with a test report that logs the update of coverage rate and mutated test inputs called adversarial examples.
TestRNN currently supports four structure-based test metrics: neutron coverage, cell coverage, gate coverage and sequence coverage. 

The mutation module mutates the input dataset for a higher coverage rate and production of more adversarial examples. There are two kinds of mutations: continuous input mutation, and discrete input mutation. For continuous inputs, such as images, the tool provides a Stochastic Gradient Ascent (SGA) engine. For discrete input problems, a series of customized mutation operations is very often needed. testRNN also provides several default mutation methods for commonly-seen discrete problems. For example, for Natural Language Processing (NLP), available mutation operations include (1) synonym replacement, (2) random insertion, (3) random swap, (4) random deletion. 

They also have used a set of Euclidean norm balls with the seed inputs as centres as an oracle to automatically consider a mutated test case as a valid one.

### 2.	DeepCodeSearch Method
DeepCodeSearch (DeepCS) introduces a novel deep neural network to find the code snippet based on the query. Code snippets are in Java language, and the tool jointly embeds code snippets and natural language descriptions into a high-dimensional vector space, in such a way that code snippet and its corresponding description have similar vectors. 

Using the unified vector representation, code snippets related to a natural language query can be retrieved according to their vectors. Semantically related words can also be recognized and irrelevant/noisy keywords in queries can be handled. 

The method details as shown in figure on top can be explained as follow:
The tool has two machine learning models. The first model embeds the code and the query, and the second one tries to find the top k most relevant code snippets to the query. 
For the training corpus, they first extract the code elements from a Java method, called feature extraction. Following is an example of feature extraction:

Given features, the model also adds its description (can be comments or description above the code snippet) to the vector. Then, they vectorize this vector using embedding layer of DNN. Here is a sample of the vector after adding description and feature extraction including Method Name, Api Sequence and Tokens:

The query also has been embedded through the embedding layer of DNN. 
After having vectors of query and vectors of codes from code repository, the two vectors can be compared together, to find the most relevant code snippets to the query.

There are 100,000 code snippets trained in the model. It means, 100K of sets of Method Namess, API Sequences, and Tokens, in addition to the description.






### 3.	Methodology
In this work, we propose a method to test an RNN model using GA. We test DeepCS model by generating test cases using GA. We apply mutation methods in testRNN for 100 generations on the inputs considering Neuron Coverage as our fitness function. 
Since calculating the coverage and generating new test cases for 100,000 inputs takes almost 20 days on a cluster for 100 generation, we decided to choose a random 10K out of 100K samples and repeat our experiment concurrently, 10 times with 10 different random seeds.  We then calculate the measures in the original DeepCS using the same 10K inputs sets, to compare the results for any improvement.
Following, describes the configuration of a model in DNNTestSuite.py file, as well as the results.

  #### 3.1.Test Suit Generation:
DNNTestSuite.py file is for generating test suits for the DeepCS. It contains mutation, crossover and generateTestSuit methods. For specified number of iterations, it does mutation and crossover on the inputs, which are lists of extracted vectors from code snippets, with rates 0.05 and 0.7, respectively, to generate new test suits. These rates are default rates in many GA application literature.
Finally, having mutated code snippets, we re-train our model, and evaluate the evaluation metrics reported on DeepCS.

  #### 3.2. Results:
Currently we are running all 100K inputs to generate test cases and calculate their coverage. We then go through a GA run and take a random 10K out of those 100K and start evolving them. The code has an optimization part where for each newly generate child in GA we first check to see if they are equal to one of the recorded 100K. If so we just take the coverage value and do not run the DeepCS which is an expensive calculation. 
As a preliminary study, we have tested our model on 2,000 input, with 100 generations of mutation through GA (no crossover) and compared it with testRNN and the default DeepCS results. It worth mentioning that the DeepCS paper results are on 100,000 data. Here, I ran only 2000 to compare with the current results we have. The following table shows the results:

Metric	DeepCS	testRNN(1-time mutation)	DeepSBST(100-times mutation)
MRR	0.004	0.012	0.155
Accuracy	0.02	0.0325	0.3
MAP	0.004	0.012	0.155
nDGC	0.007	0.017	0.189

	
### 4.	Collaboration Expectations
What we would like you to contribute here are as follows:

  1. Find a meaningful embedding for any code snippet. The current model applies the mutation and crossover on the already vectorized data (using the original embedding of DeepCS). This makes the definition of mutation and crossover easy (since they are vectors) but potentially creates many unrealistic or invalid tests. However, we think that as long as re-training the model using these new test data improves the results on the original test data, having invalid artificial test data is fine. So, we will end up proposing two methods: one applies GA operators on the embedded data and the other applies on the raw data. We then compare their results with two baselines testRNN and original DeepCS.
    
  2. We have enclosed the code for our current method. What you need to do after implementing the mutation and crossover operators on source code (#1), is to replace our current mutation and cross over with yours and rerun the code with the same setup we had. 
    
  3. Lastly, we want to try this whole thing on one more software tool other than DeepCS. Our choice so far is DeepSim [3]. Basically, while you are working on #1 we finish our part on #2 then while you are working on #2 we do ours in #3 and finally you repeat yours for DeepSim. 





### 5. References
[1] Huang, Wei, et al. "testRNN: Coverage-guided Testing on Recurrent Neural Networks." arXiv preprint arXiv:1906.08557 (2019).
[2] Gu, Xiaodong, Hongyu Zhang, and Sunghun Kim. "Deep code search." 2018 IEEE/ACM 40th International Conference on Software Engineering (ICSE). IEEE, 2018.
[3] Zhao, Gang, and Jeff Huang. "Deepsim: deep learning code functional similarity." Proceedings of the 2018 26th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering. 2018.




