# literature_for_controllable_generation
Model steerability could be viewed from following directions: 
* Apply guided decoding strategies and select desired outputs at test time.
* Optimize for the most desired outcomes via good prompt design.
* Fine-tune the base model or steerable layers to do conditioned content generation.
 
## Common Decoding Methods
Greedy search, Beam search, Top-k sampling, Nucleus sampling, Penalized sampling

## Guided Decoding
*Part of this section refers to this [blog](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/)*

Guided decoding essentially runs a more expensive beam search where the sampling probability distribution is altered by side information about human preferences.
Our preferences on topic or sentiment can be baked into the candidate ranking function to guide the sample generation by altering the candidate ranking score. The ranking score for token selection at each decoding step can be set as a combination of LM log-likelihood and a set of desired feature discriminators. The features are designed to quantify human preferences by heuristics.  

* [Hafez: an Interactive Poetry Generation System](https://aclanthology.org/P17-4008.pdf) [acl 2017] incorporate many different features for steering the style of the output. A set of feature functions define the preferences and the associated weights work like “control knobs” that can be easily customized at decoding time. Features can measure a variety of attributes and can be easily combined; for example,
  * whether exists in a bag of desired or banned topical words.
  * whether indicates certain sentiments.
  * whether is a repeated token。
  * if longer or shorter words are in particular preferred.

* [Generating More Interesting Responses in Neural Conversation Models with Distributional Constraints](https://arxiv.org/pdf/1809.01215.pdf) [emnlp 2018] manually designed features for ranking and altered the sampling distribution by appending similarity scores between topic distribution or embeddings of the context and the completion.

* [Learning to Write with Cooperative Discriminators](https://arxiv.org/pdf/1805.06087.pdf) [2018] adopted a set of learned discriminators, each specializing in a different principle of communication guided by Grice’s maxims: quality, quantity, relation and manner. The discriminators learn to encode these desired principles by measuring repetition, entailment, relevance, and lexical diversity, respectively. Given some ground truth completion, all the discriminator models are trained to minimize the ranking log-likelihood, because the gold continuation is expected to obtain a higher score than the generated one.  

* [Discriminative Adversarial Search for Abstractive Summarization](https://arxiv.org/pdf/2002.10375.pdf) [pmlr 2020] Discriminative Adversarial Search is inspired by GAN and trains the discriminator to tell apart human created text from machine generated text. The discriminator predicts a label for each token instead of for the entire sequence. The discriminator logprob is added to the score to guide sampling towards the human-written style.

* [If Beam Search is the Answer, What was the Question?](https://arxiv.org/pdf/2010.02650.pdf) [2021] studied beam search in a regularized decoding framework. The MAP part demands for sequences with maximum probability given context, while the regularizer introduces other constraints. It is possible a global optimal strategy may need to have a high-surprisal step occasionally so that it can shorten the output length or produce more low-surprisal steps afterwards.   
Beam search has gone through the test of time in the field of NLP. The question is: If we want to model beam search as exact search in a regularized decoding framework. The paper proposed a connection between beam search and the uniform information density (UID) hypothesis. “The uniform information density hypothesis (UID; Levy and Jaeger, 2007) states that—subject to the constraints of the grammar—humans prefer sentences that distribute information (in the sense of information theory) equally across the linguistic signal, e.g., a sentence.”   
It hypothesizes that humans prefer text with evenly distributed surprisal. Popular decoding methods like top-k sampling or nuclear sampling actually filter out high-surprisal options, thus implicitly encouraging the UID property in output sequences. Beam search has gone through the test of time in the field of NLP. The question is: If we want to model beam search as exact search in a regularized decoding framework. The paper proposed a connection between beam search and the uniform information density (UID) hypothesis. 


