---
title: How does forced alignment work?
author: [Elena Rastorgueva]
author_gh_user: [erastorgueva-nv]
read_time: 15 minutes
publish_date: 08/14/2023

# DO NOT CHANGE BELOW
template: blog.html
---

# How does forced alignment work?

In this blog post we will explain how you can use an Automatic Speech Recognition (ASR) model[^1] to match up the text spoken in an audio file with the time when it is spoken. Once you have this information, you can do downstream tasks such as:

[^1]: Specifically we will be explaining how to use CTC-like ([Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)) models which output a probability distribution over vocabulary tokens per audio timestep. We will explain how forced alignment works using a simplified CTC-like model, and explain how to extend it to a CTC model at the end of the tutorial. There are many CTC models [available](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#speech-recognition-languages) out of the box in NeMo. Alternative types of ASR models inlcude Transducer models (also [available](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#speech-recognition-languages) in NeMo), and attention-based encoder-decoder models (many of which build on this [work](https://arxiv.org/pdf/1508.01211.pdf), and a recent example of which is [Whisper](https://cdn.openai.com/papers/whisper.pdf)). These types of models would require a different approach to obtaining forced alignments.

* creating subtitles such as in the video below[^butter_betty_bought] or in the Hugging Face [space](https://huggingface.co/spaces/erastorgueva-nv/NeMo-Forced-Aligner)

* obtaining durations of tokens or words to use in [Text To Speech](https://arxiv.org/pdf/2104.08189.pdf) or speaker diarization models

* splitting long audio files (and their transcripts) into shorter ones. This is especially useful when making datasets for training new ASR models, since audio files that are too long will not be able to fit onto a single GPU during training. [^2]

[^butter_betty_bought]: This video is of an excerpt from 'The Jingle Book' by Carolyn Wells. The audio is a reading of a poem called "The Butter Betty Bought". The audio is taken from a [LibriVox recording](https://www.archive.org/download/jingle_book_blb_librivox/jinglebook_03_wells.mp3) of the [book](https://librivox.org/the-jingle-book-by-carolyn-wells/). We used NeMo Forced Aligner to generate the subtitle files for the video. The text was adapted from [Project Gutenberg](https://www.gutenberg.org/cache/epub/24560/pg24560.txt). Both the original audio and the text are in the public domain.

[^2]: There are toolkits specialized for this purpose such as [CTC Segmentation](https://github.com/lumaku/ctc-segmentation), which has an [integration](https://github.com/NVIDIA/NeMo/tree/main/tools/ctc_segmentation) in NeMo. It uses an extended version of the algorithm that we describe in this tutorial. The key difference is that the algorithm in this tutorial is *forced alignment* which assumes that the ground truth text provided is exactly what is spoken in the text. However, in practice the ground truth text may be different from what is spoken, and the algorithm in CTC Segmentation accounts for this.

<figure markdown>
  ![type:video](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-butter_betty_bought_words_aligned.mp4)
  <figcaption>Video with words highlighted according to timestamps obtained with NFA</figcaption>
</figure>


## What is forced alignment?

This task of matching up text to when it is spoken is called 'forced alignment'. We use 'best alignment', 'most likely alignment' or sometimes just 'the alignment' to refer to the most likely link between the text and where in the audio it is spoken. Normally these links are between chunks of the audio and the text tokens[^3]. If we are interested in word alignments, we can simply group together the token alignments for each word.
[^3]: These can be graphemes (i.e. letters or characters), phonemes, or subword-tokens

<figure markdown>
  ![What is alignment](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-what_is_alignment.png)
  <figcaption>Diagram of a possible alignment between audio with 5 timesteps and text with 3 tokens ('C', 'A', 'T')</figcaption>
</figure>


The 'forced' in 'forced alignment' refers to the fact that we provide the reference text ourselves and use the ASR model to get an alignment based on the assumption that this reference text is the real ground truth, i.e. exactly what is spoken - sometimes it makes sense to drop this requirement in case your reference text is incorrect. There are various other aligners that work on this assumption[^5].

Sometimes in discussing this topic, we may drop the 'forced' and just call it 'alignment' when we mean 'forced alignment'. We will sometimes do this in this tutorial, for brevity.

[^5]: Such as [CTC Segmentation](https://github.com/lumaku/ctc-segmentation) (also [integrated](https://github.com/NVIDIA/NeMo/tree/main/tools/ctc_segmentation) in NeMo), [Gentle](https://github.com/lowerquality/gentle) aligner.

<figure class="inline end" markdown> <!--doing class="inline end" as a way to get around the fact that without it, the image will be off-center if we specify its width-->
  ![Alignment as token duplication](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-alignment_slots.png){ width="400"}
  <figcaption>We can think of an alignment as a way to arrange the S number of tokens into T number of boxes</figcaption>
</figure>


In forced alignment our two inputs are the audio and the text. You can think of the audio as being split into $T$ equally-sized chunks, or 'timesteps', and the text as being a sequence of $S$ tokens. So we can think of an alignment as either a mapping from the $S$ tokens to the $T$ timesteps, or as a way of duplicating some of the tokens so that we have a sequence of $T$ of them, each being mapped to the timestep when it is spoken. Thus this alignment algorithm will only work if $T \ge S$.


The task of forced alignment is basically figuring out what the exact $T$-length sequence of these tokens should be in order to give you the best alignment.

## Formulating the problem

To do forced alignment, we will need an already-trained ASR model [^7]. This model's input is the spectrogram of an audio file, which is a representation of the frequencies present in the audio signal. The spectrogram will have $T_{in}$ timesteps. The ASR model will output a probability matrix of size $V \times T$ where $V$ is the number of tokens in our vocabulary (e.g. the number of letters in the alphabet of the language we are transcribing) and $T$ is the number of output timesteps. $T$ may be equal to $T_{in}$, or it may be smaller by some ratio if our ASR model has some downsampling in its neural net architecture. For example, NeMo's pretrained models have the following downsampling ratios:

* NeMo [QuartzNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#quartznet) models have $T = \frac{T_{in}}{2}$
* NeMo [Conformer](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#conformer-ctc) models have $T = \frac{T_{in}}{4}$
* NeMo [Citrinet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#citrinet) and [FastConformer](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#fast-conformer) models have $T = \frac{T_{in}}{8}$

[^7]: If you want to learn more about how ASR models are trained, [this](https://distill.pub/2017/ctc/) is an excellent tutorial.

In the diagram below, we have drawn $T_{in} = 40$ and $T = 5$, as one would obtain from one of the pretrained NeMo Citrinet or FastConformer models, which have a downsampling ratio of 8. In terms of seconds, as spectrogram frames are 0.01 seconds apart, each column in the spectrogram corresponds to 0.01 seconds, and each column in the ASR Model output matrix corresponds to $0.01 \times 8 = 0.08$ seconds.


<figure markdown>
  ![ASR model diagram](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-asr_model.png){ width="450" }
  <figcaption>The input audio contains T_{in} timesteps. The matrix outputted by the ASR Model has shape V x T</figcaption>
</figure>

As mentioned above, the ASR Model's output matrix is of size $V \times T$. The number found in row $v$ of column $t$ of this matrix is the probability that the $v$-th token is being spoken at timestep $t$. Thus, all the numbers in a given column must add up to 1.

If we didn't know anything about what is spoken in the audio, we would need to "decode" this output matrix to produce the best possible transcription. That is a whole research area of its own - a topic for another day.

Our task is **forced alignment**, where by definition we have some reference text matching the ground truth text that is actually spoken (e.g. "cat") and we want to specify exactly when each token is spoken.

As mentioned in the previous section, we essentially have $T$ slots, and we want to fill each slot with the tokens `'C', 'A', 'T'` (in that order) in the locations where the sound of each token is spoken.

To make sure we go through the letters in order, we can think of this $T$-length sequence as being one which passes through the graph below from start to finish, making a total of $T$ stops on the red tokens.

<figure markdown>
  ![Allowed token sequence](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-alowed_seq.png)
  <figcaption>Every possible alignment passes from "START" to "END" with T stops at each red token</figcaption>
</figure>


For now we ignore the possibility of some of the audio not containing any speech and the possibility of a 'blank' token, which is a key feature of CTC ("Connectionist Temporal Classification") models — we will get to that [later](#extending-to-ctc).

Let's look at the (made-up) output of our ASR model. We've removed all the tokens from our vocabulary except `CAT` and normalized the columns to make the maths easier for our example. 
We denote the values in this matrix as $p(s,t)$, where $s$ is the index of the token in the ground truth, and $t$ is the timestep in the audio.

   <table style=\"margin-left: 0 !important;"\>
      <tr>
       <th>Timestep:</th> <th>1</th> <th>2</th> <th>3</th> <th>4</th> <th>5</th>
      </tr>
      <tr>
        <td>C</td> <td>0.7</td> <td>0.4</td> <td>0.1</td> <td>0.1</td> <td>0.1</td>
      </tr>
      <tr>
        <td>A</td> <td>0.1</td> <td>0.3</td> <td>0.2</td> <td>0.4</td> <td>0.2</td>
      </tr>
      <tr>
        <td>T</td> <td>0.2</td> <td>0.3</td> <td>0.7</td> <td>0.5</td> <td>0.7</td>
      </tr>
    </table>

Our first instinct may be to try to take the argmax of each column in $p(s,t)$, however that may lead to an alignment which does not match the order of tokens in the reference text, or may leave out some tokens entirely. In the current example, such a strategy will yield `C (0.7) -> C (0.4) -> T (0.7) -> T (0.5) -> T (0.5)`, which spells `CT` instead of `CAT`.

## Forced alignment the naive way

The issue with the above attempt is we did not restrict our search to only alignments that would spell out `CAT`.

Since our number of timesteps ($T=5$) and tokens ($S=3$) is small, we can list out every possible alignment, i.e. every possible arrangement of `'CAT'` that will fit in our 5 slots:

   <table style=\"margin-left: 0 !important;"\>
      <tr>
       <th>Timestep:</th> <th>1</th> <th>2</th> <th>3</th> <th>4</th> <th>5</th>
      </tr>
      <tr>
        <td>alignment 1</td> <td>C</td> <td>C</td> <td>C</td> <td>A</td> <td>T</td> 
      </tr>
      <tr>
        <td>alignment 2</td> <td>C</td> <td>C</td> <td>A</td> <td>A</td> <td>T</td> 
      </tr>    
      <tr>
        <td>alignment 3</td> <td>C</td> <td>C</td> <td>A</td> <td>T</td> <td>T</td>
      </tr> 
      <tr>
        <td>alignment 4</td> <td>C</td> <td>A</td> <td>A</td> <td>A</td> <td>T</td>
      </tr> 
      <tr>
        <td>alignment 5</td> <td>C</td> <td>A</td> <td>A</td> <td>T</td> <td>T</td>
      </tr>
      <tr>
        <td>alignment 6</td> <td>C</td> <td>A</td> <td>T</td> <td>T</td> <td>T</td>
      </tr>
    </table>

Each token has a certain probability of being spoken at each time step, determined by our ASR model.
The probability of a particular sequence of tokens is calculated by multiplying together the individual probabilities of each token at each timestep.
Assuming our ASR model is a good one, the best alignment is the one with the highest cumulative probability.

Let's show the $p(s,t)$ probabilities again.

   <table style=\"margin-left: 0 !important;"\>
      <tr>
       <th>Timestep:</th> <th>1</th> <th>2</th> <th>3</th> <th>4</th> <th>5</th>
      </tr>
      <tr>
        <td>C</td> <td>0.7</td> <td>0.4</td> <td>0.1</td> <td>0.1</td> <td>0.1</td>
      </tr>
      <tr>
        <td>A</td> <td>0.1</td> <td>0.3</td> <td>0.2</td> <td>0.4</td> <td>0.2</td>
      </tr>
      <tr>
        <td>T</td> <td>0.2</td> <td>0.3</td> <td>0.7</td> <td>0.5</td> <td>0.7</td>
      </tr>
    </table>

We can calculate the probability of each possible alignment by multiplying together each $p(s,t)$ that it passes through:

   <table style=\"margin-left: 0 !important;"\>
      <tr>
       <th>Timestep:</th> <th>1</th> <th>2</th> <th>3</th> <th>4</th> <th>5</th> <th>Total probability of alignment</th>
      </tr>
      <tr>
        <td>alignment 1</td> <td>C</td> <td>C</td> <td>C</td> <td>A</td> <td>T</td><td>0.7 * 0.4 * 0.1 * 0.4 * 0.7 = 0.008</td> 
      </tr>
      <tr>
        <td>alignment 2</td> <td>C</td> <td>C</td> <td>A</td> <td>A</td> <td>T</td> <td>0.7 * 0.4 * 0.2 * 0.4 * 0.7 = 0.016</td>
      </tr>    
      <tr>
        <td>alignment 3</td> <td>C</td> <td>C</td> <td>A</td> <td>T</td> <td>T</td><td>0.7 * 0.4 * 0.2 * 0.5 * 0.7 = 0.020</td>
      </tr> 
      <tr>
        <td>alignment 4</td> <td>C</td> <td>A</td> <td>A</td> <td>A</td> <td>T</td><td>0.7 * 0.3 * 0.2 * 0.4 * 0.7 = 0.012 </td>
      </tr> 
      <tr>
        <td>alignment 5</td> <td>C</td> <td>A</td> <td>A</td> <td>T</td> <td>T</td><td>0.7 * 0.3 * 0.2 * 0.5 * 0.7 = 0.015</td>
      </tr>
      <tr>
        <td>alignment 6</td> <td>C</td> <td>A</td> <td>T</td> <td>T</td> <td>T</td><td>0.7 * 0.3 * 0.7 * 0.5 * 0.7 = 0.051 <- the max</td>
      </tr>
    </table>

We can see that the most likely path is `'CATTT'` because it has the highest total probability. In other words, based on our ASR model, the most likely alignment is that `'C'` was spoken at the first timestep, `'A'` was spoken at the second timestep, and `'T'` was spoken for the last 3 timesteps. 


## The naive way but listing all the possible paths using a graph

To make further progress in understanding forced alignment, let's list all the possible paths in a systematic way by arranging them in a tree-like graph like the one below. 

We initialize the graph with a 'start' node (for clarity), then connect it to nodes representing the tokens that our alignment can have at the first timestep (`t=1`). In our case, this is only a single token `C`. From that `C` node, we draw arrows to 2 other nodes. The higher node represents staying at the same token (`C`) for the next timestep (`t=2`). The lower node represents going to the next token (`A`) for the next timestep (`t=2`). We continue this process until we have drawn all the possible paths through our reference text tokens for the fixed duration $T$. We do not include paths that reach the final token too early or too late.

We end up with the tree below, which represents all the possible paths through the `CAT` tokens over 5 timesteps. 

You can check for yourself that every alignment we listed in the table in the previous section is represented as a path from left to right in this tree.

We can label each node in the graph with its $p(s,t)$ probabilities (dark green).

Let's also calculate the cumulative product along each path and include that as well (light green).

Once we do that, we can look at all the nodes at the final timestep, and see that the cumulative product at each node is the cumulative probability of the path from start to T that lands at that node. 

As before, we can see that the probability of the most likely path, i.e. the most likely alignment, is 0.051. If we trace back our steps from that T node to the start, then we see that the path is `'CATTT'`.[^greedy_path]

[^greedy_path]: In the graph, we can also try to follow a 'greedy' path where we only take the outgoing path with the highest probability. In this example, this 'greedy' approach would give us the alignment path `CCATT`, which has a probability of 0.020 - lower than the actual most likely path.


<figure markdown>
  ![type:video](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-naive_graph.mp4)
  <figcaption>This graph lists every possible alignment. The most likely alignment becomes highlighted in purple.</figcaption>
</figure>

## The trouble with longer sequences

The naive method in the previous sections was intuitive, easy to calculate and gave us the correct answer, but it was only feasible because we had such a small number of possible alignments. In an utterance of 10 words over 5 seconds, conservatively you can expect 20 tokens and 63 timesteps[^tokens_timesteps] - that would lead to about $4.8 \times 10^{15}$ possible alignments[^stars_bars]! 

Fortunately there is a method to obtain the most likely alignment, for which you:

* don't need to calculate all the cumulative products for every path, and
* don't even need to draw the full tree graph.

We will work towards this method in the next sections.

[^tokens_timesteps]:
    To estimate the number of tokens, we assume there are 2 tokens per word $\implies 10$ words $\times 2$ tokens/word = 20 tokens. 
    
    To estimate the number of timesteps, we assume a spectrogram frame hop size of 0.01 $\implies \frac{5}{0.01} = 500 = T_{in} \implies {T} = \frac{T_{in}}{8} = \frac{500}{8} = 62.5 \approx 63$.

[^stars_bars]: 
    The exact number of possible paths in our formulation is equal to ${T-1 \choose S-1 }$, and if $T=63$ and $S=20$, then ${T-1 \choose S-1 }={63-1 \choose 20-1 }={62 \choose 19 }=4.8 \times 10^{15}$.

    We will explain how we we deduced the formula ${T-1 \choose S-1 }$ below.
    
    The number of possible paths is the same as the number of ways we could fit $S$ tokens into $T$ boxes, with the $S$ tokens being in some
    fixed order.

    This formulation is the same as having $T$ boxes, and putting $S-1$ markers in between the boxes, where the markers indicate a switch from one token to the next. There are $T-1$ locations where the markers can go (i.e. $T-1$ spaces between the boxes), therefore the number of possible ways we could arrange the markers is ${T-1 \choose S-1 }$. Therefore, this is also the number of possible alignment paths in our setup with $S$ tokens and $T$ timesteps.

    This is analogous to a known result in combinatorics called [stars and bars](https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)#Theorem_one_proof) - in our case our 'boxes' are the 'stars' and our 'markers' are the 'bars'.


## Spotting graph redundancies
Fortunately, because we are only interested in the highest probability path from the start node to one of the final `T` nodes on the right, this means that the tree graph has a lot of redundant nodes that we don't need to worry about.

For example, consider the two nodes in the tree corresponding to token `A` (`s=2`) at time `t=3`. Looking at the cumulative products of these two nodes (in light green), we can see that the top `A` node (corresponding to the partial path `CCA`) has a cumulative product of 0.056, while the bottom `A` node (corresponding to the partial path `CAA`) has a smaller cumulative product of 0.042.

The paths downstream of the top node are identical in graph structure to those downstream of the bottom node, as are the $p(s,t)$ values (in dark green) of any nodes in these downstream paths. Therefore, since the cumulative product of any path downstream of an `A` node at `t=3` will just be the cumulative product of that `A` node at `t=3` multiplied by these downstream $p(s,t)$ values, it is impossible for any path downstream of the bottom `A` node to have a higher cumulative product than the corresponding path downstream of the top `A` node. Thus, it is impossible that any of the paths downstream of the bottom `A` node will end up being the optimal path overall, and we can safely discard them. This is shown in the animation below.

<figure markdown>
  ![type:video](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-redundancy_explain.mp4)
  <figcaption>If we examine the two 'A' nodes at 't=3', we see that we can discard the node with the lower cumulative product, and the nodes downstream of it.</figcaption>
</figure>


These redundancies exist between any nodes with the same $s$ and $t$ values. All of their downstream nodes will have the same structure, but we only need to keep the node that corresponds to the **most probable path** from the start node **to (s,t)**.

So, for each $(s,t)$ we only need to record the **most probable path to (s,t)** and can discard the non-winning node and its downstream nodes. Discarding the downstream nodes means that we will have fewer nodes to look at in the next timesteps, helping to keep the number of computations required relatively low.

The animation below shows this. We start with all nodes in their original colors, and color nodes red if they represent the **most probable path to (s,t)**. When there is only one node in the graph that has a particular $(s,t)$ value, it by default is the **most probable path to (s,t)**, so we color it red immediately. When there is more than one node with the same $(s,t)$ value, we look at the cumulative probability (in light green) of these nodes. We mark the node with the lower cumulative probability dark blue, meaning we calculated its cumulative probability but realized that we can discard it. We mark its downstream nodes dark gray, to indicate that we can discard them, and don't need to consider them in future timesteps. Finally, for the node with the higher cumulative probability, we mark it red, to indicate that it is the **most probable path to (s,t)**.


<figure markdown>
  ![type:video](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-redundancy_start_to_end.mp4)
  <figcaption>We cycle through all possible (s,t) and discard the nodes that we do not need.</figcaption>
</figure>

The cumulative product of the remaining node at the final timestep is the probability of the most likely alignment path. Again, we can trace back the steps from that node to the start to recover the exact path that gives that probability (`'CATTT'`).

Although we show the cumulative probabilities for all nodes in the animation (for illustrative purposes), we didn't need to calculate the cumulative probabilities for any of the nodes that are dark gray, but we still managed to find the most likely path. We obtained the same result as with naive graph method but a lot fewer operations.

## Formalizing the efficient forced alignment algorithm

Let’s formalize the steps we followed. If we look at the nodes that we didn’t discard, they form a different shape of graph. The animation below transforms the tree graph into its new shape by hiding the discarded nodes behind the non-discarded nodes.

<figure markdown>
  ![type:video](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-fold_viterbi.mp4)
  <figcaption>We create a different shape of graph using only the nodes that we did not discard.</figcaption>
</figure>

The resulting graph has a single node for each $(s,t)$. This graph is often referred to as a **trellis**. Each node has an associated number in red which is the ***probability* of the most likely path from start to $(s,t)$**. (We also keep the $p(s,t)$ values in dark green for illustrative purposes).

We can recover the most likely alignment by looking at the node at $(S,T)$. Its cumulative probability, in red, is the probability of the most likely alignment. We can also recover the exact path that has that probability by tracing following the non-discarded edges (in light gray) backwards to the start token. This produces the `'CATTT'` sequences yet again.

At each node $(s,t)$, the procedure we used to calculate the **most probable path to $(s,t)$**, was to look at the tokens in the previous timestep that could have  transitioned into this $(s,t)$, calculate the *candidates* for the **most probable path to $(s,t)$**, and pick the maximum value.

In our scenario, where at each timestep the token either stays the same or moves onto the next one, candidates come from $(s-1, t-1)$ & $(s,t-1)$. 

We can denote this formula as:

`prob-of-most-likely-path-to(s, t) = max (prob-of-most-likely-path-to(s-1, t-1) * p(s,t), prob-of-most-likely-path-to(s, t-1) * p(s,t))`.

Let's make the formula shorter by denoting `prob-of-most-likely-path-to(s-1, t-1)` using the letter 'v' (to be explained later). So the formula becomes: 

$$v(s, t) = \max (v(s-1, t-1) \times p(s,t), v(s, t-1) \times p(s,t))$$

We can also show the rule on a subsection of our trellis graph, as below.

<figure markdown>
  ![Viterbi rule](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-viterbi_rule.png)
  <figcaption>The formula we use to calculate v(s,t) probabilities in our current setup.</figcaption>
</figure>


## The Viterbi algorithm
What we described above is actually known as the Viterbi algorithm. What we denoted `prob-of-most-likely-path-to(s, t)` is often denoted `v(s,t)`, A.K.A. the Viterbi probability (of token `s` at time `t`).

The Viterbi algorithm is an efficient method for finding the most likely alignment, and exploits the redundancies in the tree graph discussed above. It involves creating a matrix of size $S \times T$ (called a 'Viterbi matrix') and populating it column-by-column. 

The shape of the initialized Viterbi matrix for our scenario is shown below. Every element in the matrix corresponds to a node in the trellis (except for elements in the bottom left and top right of the matrix, which we did not draw in our trellis as they would not form valid alignments).

   <table style=\"margin-left: 0 !important;"\>
      <tr>
       <th></th> <th>t=1</th> <th>t=2</th> <th>t=3</th> <th>t=4</th> <th>t=5</th> 
      </tr>
      <tr>
        <td>s=0 AKA token is C</td> <td>??</td> <td>??</td> <td>??</td> <td>??</td> <td>??</td> 
      </tr>
      <tr>
        <td>s=1 AKA token is A</td> <td>??</td> <td>??</td> <td>??</td> <td>??</td> <td>??</td> 
      </tr>    
      <tr>
        <td>s=2 AKA token is T</td> <td>??</td> <td>??</td> <td>??</td> <td>??</td> <td>??</td>
      </tr> 
    </table>


We need to fill in every element in the Viterbi matrix column-by-column and row-by-row[^row_by_row]. Once we have finished, $v(s=S,t=T)$ will contain the *probability of the most likely path to (S,T), i.e. from start to finish*, and if we recorded the argmax for computing each $(s,t)$, then we can use these values to recover what the exact sequence is which has this highest probability. The recorded argmaxes (which in our trellis look like light gray arrows) are often called "backpointers". The process of using the backpointers to recover the token sequence of the most likely alignment is known as "backtracking".

[^row_by_row]: Each entry within the same column is actually independent of the other entries in that column. We can compute those entries in any order or, even better, simultaneously - which would speed up the time to complete the algorithm.

Some special cases:

* For the first $(t=1)$ column of the Viterbi matrix, we set $v(s=1,t=1)$ to $p(s=1,t=1)$ and all other $v(s>1,t=1)$ to 0. We do this because all possible alignments must start with the first token in the ground truth $(s=1)$. (This only holds for our current setup, and would be different in a CTC setup, see below).

* For some values of $(s,t)$, either the $(s-1, t-1)$ or $(s, t-1)$ nodes are not reachable, in which cases we ignore their terms in the $v(s,t)$ formula, and do a trivial max over one element.

## Extending to CTC
The example above was simplified from a CTC approach to make it easier to understand and visualize. If you want to work with CTC alignments, you must allow blank tokens in between your ground truth tokens (the blank tokens are always optional except for in between repeated tokens.)

Thus, if the reference text tokens are still `'C', 'A', 'T'`, we add optional 'blank' tokens in between them, which we denote as `<b>`. The diagram of the allowed sequence of tokens would look like this:
<figure markdown>
  ![Allowed sequence for CTC](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-allowed_seq_ctc.png)
  <figcaption>Every possible alignment passes from "START" to "END" with T stops at each red token</figcaption>
</figure>

As you can see, `S` — the total number of tokens — is 7 (3 non-blank tokens and 4 blank tokens). If we keep $T=5$, the trellis for CTC with reference text tokens `'C', 'A', 'T'` would look like this:
<figure markdown>
  ![Trellis for CTC](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-ctc_trellis.png)
  <figcaption>The shape of the trellis if we use a CTC model, our reference text tokens are 'C', 'A', 'T', and the number of timesteps in the ASR model output (T) is 5.</figcaption>
</figure>

The Viterbi algorithm rules would also change, as follows:
<figure markdown>
  ![Viterbi rule for CTC](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/asset-post-2023-08-forced-alignment-ctc_viterbi_rule.png)
  <figcaption>The formula we use to calculate v(s,t) probabilities for a CTC model.</figcaption>
</figure>

However, the principle remains the same: we initialize a Viterbi matrix of size $S \times T$ and fill it in according to the recursive formula. Once we fill it in, because there are 2 valid final tokens, we need to compare $v(s=S-1,t=T)$ and $v(s=S,t=T)$ - the token with the higher value is the end token of the most likely probability. To recover the overall most likely alignment, we need to backtrack from the higher of $v(s=S-1,t=T)$ or $v(s=S,t=T)$.

## Conclusion
In this tutorial, we have shown how to do forced alignment using the Viterbi algorithm, which is an efficient way to find the most likely path through the reference text tokens.

You can obtain forced alignments using the [NeMo Forced Aligner (NFA)](https://github.com/NVIDIA/NeMo/tree/main/tools/nemo_forced_aligner) tool within NeMo, which has an efficient PyTorch tensor-based [implementation](https://github.com/NVIDIA/NeMo/blob/main/tools/nemo_forced_aligner/utils/viterbi_decoding.py) of Viterbi decoding on CTC models. An efficient CUDA-based implementation of Viterbi decoding was also recently [added](https://pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html#torchaudio.functional.forced_align) to TorchAudio, though it is currently only available in the nightly version of TorchAudio, and is not always faster than the current NFA PyTorch tensor-based implementation.

Although our examples used characters are tokens, most NeMo models use sub-word tokens, such as in the diagram below. Furthermore, although we've given examples using probabilities ranging from 0 to 1, most Viterbi algorithms operate on log probabilities, which will make the operations in the algorithm more numerically stable. 

<figure markdown>
  ![NFA pipeline](https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/nfa_forced_alignment_pipeline.png)
  <figcaption>The NFA forced alignment pipeline, which has been described in this blog post.</figcaption>
</figure>

To learn more about NFA, you can now refer to the resources [here](./2023-08-nfa.md).
