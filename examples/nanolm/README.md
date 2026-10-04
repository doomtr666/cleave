# nanoLM — a small language model of 19th-century French, written in cleave

nanoLM is a GPT-style language model, trained from scratch on a CPU. It reads 1,536 books of
19th-century French literature (Sand, Dumas, Verne, Maupassant, Zola, Hugo, Balzac, Flaubert...)
and learns to continue a text. Everything the model computes — the forward pass, the gradient,
the optimizer, the parallel training step — is written in cleave (`src/kernel.cleave`); a small
Rust host prepares the text and prints the results. A PyTorch twin
(`bench/nanolm-pytorch`) implements the same model line for line, starts from the same weights,
and must find the same losses.

This README is also a guided tour of how a language model works, written for readers who know
programming but not machine learning. Every idea is tied to the line of code that implements it.
No prior knowledge of neural networks is assumed; some comfort with matrices and logarithms
helps.

```
Le soir tombait sur Paris
--Mon ami, je ne devai plus la vérité. Vous y êtes encore, je voudrais me trouverais.
--Je ne sais pas, répondit le haut de la porte.
```
*(a sample after 1,000 training steps out of 20,000: French words, French dialogue, no sense
yet)*

## Contents

1. [Quick start](#1-quick-start)
2. [The problem: predicting the next piece of text](#2-the-problem-predicting-the-next-piece-of-text)
3. [Measuring a prediction: surprise, nats, bits, perplexity](#3-measuring-a-prediction-surprise-nats-bits-perplexity)
4. [The corpus](#4-the-corpus)
5. [Tokens: cutting text into pieces (BPE)](#5-tokens-cutting-text-into-pieces-bpe)
6. [Batches: the exercise the model is given](#6-batches-the-exercise-the-model-is-given)
7. [From counting to learning: the three baselines](#7-from-counting-to-learning-the-three-baselines)
8. [The transformer](#8-the-transformer)
9. [Training: gradients, Adam, and the schedule](#9-training-gradients-adam-and-the-schedule)
10. [Parallelism: one step on eight cores](#10-parallelism-one-step-on-eight-cores)
11. [Generation: writing one token at a time](#11-generation-writing-one-token-at-a-time)
12. [Scale: how big, how long, how good](#12-scale-how-big-how-long-how-good)
13. [The PyTorch twin](#13-the-pytorch-twin)
14. [Files](#14-files)
15. [Glossary and further reading](#15-glossary-and-further-reading)

---

## 1. Quick start

All commands run from the repository root. The first run downloads the corpus from Project
Gutenberg (about 1,500 books, paced at one per second: 30 to 60 minutes, once) and learns the
tokenizer (seconds); both are cached in `examples/nanolm/.cache/`.

```
cargo run --release -p nanolm -- corpus                 # build the corpus and the tokens, show them
cargo run --release -p nanolm -- gpt 0 200 100          # train: from step 0, 200 rounds of 100 steps
cargo run --release -p nanolm -- gpt 1000 190 100       # resume from step 1000 (the last checkpoint)
cargo run --release -p nanolm -- write "Le soir tombait sur Paris" 200 0.8 0.9
                                                        # continue a prompt: 200 tokens, temperature 0.8, top-p 0.9
cargo run --release -p nanolm -- bigram                 # the baselines (section 7)
cargo run --release -p nanolm                           # the embedding + MLP baseline
```

`gpt` saves a checkpoint (`gpt.ckpt`: the weights, the optimizer state, the random generator)
after every round, in `.cache/gpt-d384-l8-v4096/` (one directory per model shape). Training can
be stopped at any time and resumed exactly where it was: the batches and the learning rate are
functions of the step number alone. `write` works at any time, on the latest checkpoint.

Each round prints a line like:

```
step 600: train 4.8053, validation 4.9851 nats/token (2.0502 bits/char), learning rate 0.000982, 1387 ms/step, 13.6 min elapsed
```

Section 3 explains every number in it.

## 2. The problem: predicting the next piece of text

A language model does one thing: given the beginning of a text, it gives a **probability for
every possible continuation** of one more piece. After *"Le soir tombait sur"*, a good model puts
a high probability on *" Paris"*, *" la"*, *" les"*, a low one on *" chaise"*, and nearly zero on
*"zzz"*.

Everything else follows from that:

- **Writing** is repeated prediction: draw one piece from the distribution, append it, predict
  again (section 11).
- **Learning** is adjusting the model so that, on real text, the piece that actually came next
  gets a higher probability (section 9).
- **Understanding**, to the extent a model has any, is whatever internal structure turns out to
  help prediction. To predict the end of *"Gervaise posa son fer et elle"* well, a model must
  have noticed that Gervaise is a woman, that she irons, that "elle" refers to her. Nobody tells
  the model about grammar, characters or plots; it builds whatever helps it predict.

This framing — learn by predicting the next piece of real text — is called **self-supervised
learning**: the text is its own answer key, no human labelling is needed, so any amount of text
can be used.

## 3. Measuring a prediction: surprise, nats, bits, perplexity

### Where the probabilities come from: logits and softmax

A neural network computes with unconstrained numbers: they can be any size, positive or
negative. So the model doesn't output probabilities directly. For every token of the vocabulary
it outputs a **score**, called a **logit** (the word comes from statistics, where the logit is the
logarithm of odds): the higher the score, the more likely the model thinks that token comes
next. The 4,096 logits are then turned into 4,096 probabilities by the **softmax** function:

$$p_t = \frac{e^{z_t}}{\sum_{t'} e^{z_{t'}}}$$

Each score is exponentiated (which makes it positive and amplifies the differences between
scores), then divided by the sum of all of them (which makes the results add up to 1). With four
tokens instead of 4,096:

| token | logit $z$ | $e^z$ | probability |
|---|---|---|---|
| `' Paris'` | 4.0 | 54.6 | 0.704 |
| `' la'` | 3.0 | 20.1 | 0.259 |
| `' les'` | 1.0 | 2.7 | 0.035 |
| `' chaise'` | −2.0 | 0.14 | 0.002 |

Only the differences between logits matter: adding the same number to all of them changes
nothing. A difference of 1 between two logits means a ratio of $e \approx 2.7$ between their
probabilities. The name "softmax" says what it does: a smooth version of "pick the maximum",
which still gives every token some probability, and which can be differentiated — training needs
that (section 9).

### Surprise

If the model gave probability $p$ to the token that actually came next, its **surprise** is

$$\text{surprise} = -\ln p$$

- $p = 1$ (certain, and right): surprise $0$.
- $p = 1/2$: surprise $\ln 2 \approx 0.69$.
- $p = 1/4096$ (a uniform guess among 4,096 tokens): surprise $\ln 4096 \approx 8.32$.
- $p \to 0$ (certain it couldn't happen, and it did): surprise $\to \infty$.

The logarithm is what makes surprises **add up**: the probability of a whole text is the
product of the probabilities of its pieces, so its surprise is the sum of theirs. A model is
good when its total surprise on real text is low.

### Nats and bits

Surprise measured with the natural logarithm ($\ln$) is in **nats**; with $\log_2$, in **bits**
($1 \text{ nat} = 1/\ln 2 \approx 1.443$ bits). Bits have a concrete meaning, from Shannon's
information theory: a model with a surprise of $b$ bits per character could be used to
**compress** the text to $b$ bits per character (by arithmetic coding). A good language model
*is* a good compressor; the two problems are the same problem.

### Cross-entropy: the loss

The **loss** printed during training is the average surprise over many predictions — the
**cross-entropy** between the real text and the model's predictions. It's in nats per token:

```
validation 4.9851 nats/token
```

means: on text the model has never seen, the token that came next got, on average (geometric
average), probability $e^{-4.9851} \approx 1/146$.

### Perplexity

$e^{\text{loss}}$ is the **perplexity**: the number of equally likely choices the model hesitates
between, on average. A loss of 4.99 nats/token is a perplexity of about 146: as uncertain as if
it had to choose among 146 tokens at random — far better than 4,096, far from fluent.

| loss (nats/token) | perplexity | what it means |
|---|---|---|
| 8.32 = ln 4096 | 4,096 | uniform guessing: step 0 |
| ~5.9 | ~350 | the bigram baseline (one token of context, section 7) |
| ~5.0 | ~150 | nanoLM after 600 steps |
| ~3.3 | ~27 | the range this run aims at (section 12) |
| 1.0 | 2.7 | hesitating between ~3 choices: beyond what natural text allows |

### Why the loss never reaches zero

Natural language has an entropy of its own: after *"Le soir tombait sur"*, several continuations
are genuinely right. Shannon estimated around one bit per character for English; the best
models today approach that, and no model can go far below it on text it has never seen. A model
whose loss went to zero on its *training* text would simply have memorized it — which is why the
loss that matters is the **validation** loss, on books held out of training (section 4). When
the training loss keeps falling while the validation loss stalls or rises, the model is
**overfitting**: memorizing instead of generalizing.

### Bits per character: comparing across tokenizers

A loss per token depends on what a token is: predicting a whole word is harder than predicting
one letter. To compare models with different tokenizers, the host also prints the loss **per
character**, in bits: nats per token, divided by $\ln 2$ and by the average number of characters
per token (3.51 here, measured on the validation text).

```
4.9851 nats/token  →  4.9851 / ln 2 / 3.51  =  2.05 bits/char
```

The first, character-level version of nanoLM (one token = one character, a single author) ended
at about 1.5 bits/char; this one passes that early in its run.

## 4. The corpus

`books.txt` lists the 1,537 Project Gutenberg texts that are in French, classed as French
literature (Library of Congress class *PQ*, which excludes translations), and whose authors were
all born between 1790 and 1880 — Zola's generation and those around it. The list was selected
once from Gutenberg's catalog and frozen in the repository, so the corpus doesn't change when the
catalog does.

`src/corpus.rs` downloads each book once and cleans it:

- the Gutenberg header and license are cut (older files use other markers; a book whose frame
  isn't recognized is skipped);
- paragraphs that aren't the book's — producers' credits, transcribers' notes, illustration
  captions, digitizers' notices, ASCII tables — are dropped;
- each paragraph is rejoined onto one line, spaces normalized;
- every character is mapped onto a fixed **alphabet of 104 characters**: letters (with French
  accents), digits, punctuation, `«»`, space, line break. Typographic variants are folded (`’` →
  `'`, `—` → `-`, `…` → `...`), anything else is dropped.

About one book in fifty, chosen by a hash of its number (`is_validation`), is held out **whole**
for validation: the model never sees any part of it during training, not even the other half of a
paragraph. Result: **563 million characters** of training text, 12.9 million of validation.

## 5. Tokens: cutting text into pieces (BPE)

### Why not characters, why not words

The model sees text as a sequence of integers, one per **token**. What should a token be?

- **Characters** (the first nanoLM): a tiny vocabulary, any text representable. But a sequence of
  256 characters is barely two sentences, and the model spends most of its capacity learning to
  spell.
- **Words**: meaningful units, but hundreds of thousands of them (every conjugation, every
  proper name), most seen too rarely to learn, and an unseen word can't be written at all.
- **Subwords**, the standard answer: frequent words are one token, rare words are spelled from
  frequent fragments. *" soir"* is one token; *"omnibus"* is *"omn"* + *"ib"* + *"us"*.

### Byte-pair encoding

nanoLM uses **byte-pair encoding** (BPE; Sennrich et al., 2016), the method behind GPT-2's
tokenizer, over its 104 characters (`src/bpe.rs`):

1. Start with one token per character: tokens 0 to 103.
2. Count every pair of adjacent tokens in the training text.
3. The most frequent pair becomes a new token (ties: the smallest pair, so training is
   deterministic). Replace it everywhere.
4. Repeat until the vocabulary has 4,096 tokens: 3,992 merges.

The first merges learned on this corpus are `' d'`, `' l'`, `'es'`, `'en'`, `'ai'`, `' p'`,
`' s'`, `'ou'`, `'on'`... — the most common fragments of French. Later ones are whole words and
endings: `' pouvait'`, `'ément'`, `' honneur'`, `'issements'`.

One rule shapes the vocabulary: text is first cut into **pieces** — a word with the space before
it, a run of digits, a run of punctuation, a line break — and merges never cross a piece. So a
token is never half of one word and half of the next, and the space is part of the word that
follows it:

```
"Le soir tombait sur Paris, et Gervaise attendait l'omnibus."     59 characters, 17 tokens

'Le' | ' soir' | ' tomb' | 'ait' | ' sur' | ' Paris' | ',' | ' et' | ' G' | 'erv' | 'aise'
     | ' attendait' | " l'" | 'omn' | 'ib' | 'us' | '.'
```

Encoding applies the learned merges in the order they were learned; decoding just concatenates
the tokens' texts, so encoding then decoding gives back exactly the original text (tested). Over
the whole corpus: **161 million training tokens, 3.51 characters per token**. A context of 256
tokens covers about 900 characters, a long paragraph.

The tokenized corpus is stored as 16-bit integers (`.cache/french/bpe4096/train.bin`), which the
PyTorch twin reads directly.

## 6. Batches: the exercise the model is given

A training **batch** is 32 windows of 257 consecutive tokens taken at random positions in the
training text (`src/data.rs`). For each window, the first 256 tokens are the **input** and the
last 256 (shifted by one) are the **targets**:

```
input:   Le  │ soir │ tomb │ ait  │ sur  │ ...
target:  soir│ tomb │ ait  │ sur  │ Paris│ ...
```

At every position, the model must predict the next token from all the tokens before it. A batch
is therefore $32 \times 256 = 8{,}192$ predictions, all made at once (section 8 explains how the
model is prevented from peeking at the answer).

Batch $i$ is a pure function of $i$: its offsets come from a hash (`splitmix64`) of the batch
number. A resumed run sees exactly the batches it would have seen, and the PyTorch twin draws
the same ones.

## 7. From counting to learning: the three baselines

Before the transformer, the kernel builds three simpler models. They are the history of the
field in miniature, and each sets a bar the next must clear.

**The counted bigram** (`bigram_baseline`). Count, in the training text, how often each token
follows each other token: a $4096 \times 4096$ table. The probability of $b$ after $a$ is its
count divided by the row's total (plus one everywhere — *add-one smoothing* — so that no pair has
probability zero). This model only ever looks at **one token of context**. On the validation
text: about 5.9 nats/token. No learning, just statistics: this is the bar.

**The learned bigram** (`train_bigram`). The same table, but as **parameters** adjusted by
gradient descent (section 9) instead of counted. Its loss converges towards the counted one —
a check that the whole learning machinery (loss, gradient, optimizer) works.

**Embedding + MLP** (`train_lm`). Still one token of context, but now the token goes through a
small neural network: an **embedding** (each token mapped to a vector of 64 numbers), a dense
layer to 256 numbers, a non-linearity (GELU), a dense layer to 4,096 scores. This is the shape
of Bengio et al.'s neural language model (2003). With one token of context it can't beat the
bigram either; it is the gradient path of every layer type the transformer uses, checked
against the PyTorch twin.

All three are stuck at one token of context. Language needs more: agreement across a sentence,
who "elle" is, what the paragraph is about. Giving a model a long context — and the means to use
it — is what the transformer does.

## 8. The transformer

### The big picture

The transformer (Vaswani et al., 2017, *"Attention Is All You Need"*) processes all 256
positions of a sequence at once. Each position carries a **vector** of 384 numbers that is
progressively rewritten by a stack of 8 identical **blocks**; at the end, each position's vector
is turned into 4,096 scores, one per possible next token.

```
tokens (256 integers)
   │
   ▼
token embedding + position embedding        x: [256, 384]
   │
   ▼  ┌─────────────────────────────────────────────┐
   │  │ block (×8)                                  │
   │  │   x = x + attention(layer_norm(x))          │  positions exchange information
   │  │   x = x + mlp(layer_norm(x))                │  each position computes on its own
   │  └─────────────────────────────────────────────┘
   ▼
final layer_norm, dense head                logits: [256, 4096]
   │
   ▼
softmax → a probability for every next token, at every position
```

In the kernel (`gpt_logits`, `block`):

```
fn block<const N: i32>(x: Tensor<f32, N, WIDTH>, b: Block) -> Tensor<f32, N, WIDTH> {
    let h = layer_norm(x, b.ln1_g, b.ln1_b);
    let a = causal_attention(b.wq.dense_forward(h), b.wk.dense_forward(h), b.wv.dense_forward(h), AttentionShape::<CONTEXT, HEAD>());
    let x2 = x + b.wo.dense_forward(a);
    x2 + b.proj.dense_forward(gelu(b.fc.dense_forward(layer_norm(x2, b.ln2_g, b.ln2_b))))
}
```

The pieces, one by one.

### Embeddings: tokens become vectors

A token id is just a number; 812 is not "close" to 813. The **token embedding** is a table of
$4096 \times 384$ learned numbers: token $t$ becomes row $t$, a point in a 384-dimensional space.
Training moves these points so that tokens that behave alike end up near each other (*" soir"*
near *" matin"*, *" il"* near *" elle"*), because similar vectors lead to similar predictions.

Attention, as we'll see, treats its input as a **set**: it has no built-in notion of order. Order
is added by a **position embedding**: a second table, $256 \times 384$, whose row $k$ is added to
the vector of the token at position $k$. Both tables are learned. (This learned, absolute scheme
is GPT-2's; it also caps the context at 256. Modern models often use rotary embeddings, RoPE,
which encode relative positions instead.)

### Self-attention: positions talk to each other

Attention is how information moves between positions. To predict what follows *"Gervaise posa
son fer et elle"*, the position of *"elle"* needs to fetch information from the position of
*"Gervaise"*. Attention lets every position **look at every earlier position and decide how much
of each to take**.

Each position's vector $x_i$ is projected (by learned matrices) into three vectors:

- a **query** $q_i = x_i W_Q$: *what am I looking for?*
- a **key** $k_j = x_j W_K$: *what do I contain?* (to be matched against queries)
- a **value** $v_j = x_j W_V$: *what do I give if someone attends to me?*

Position $i$ scores every position $j$ by how well its query matches $j$'s key — a dot product
— turns the scores into weights that sum to one (softmax), and takes the weighted average of the
values:

$$s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}}, \qquad
w_{ij} = \frac{e^{s_{ij}}}{\sum_{j'} e^{s_{ij'}}}, \qquad
\text{out}_i = \sum_j w_{ij}\, v_j$$

In matrix form, for a whole sequence: $\text{softmax}(QK^\top/\sqrt{d_h})\,V$. It's a **soft,
differentiable lookup**: like a dictionary access, but every key matches a little, and the
result is a blend. Because it's differentiable, training can learn *what* to look for.

**Why divide by $\sqrt{d_h}$.** A dot product of two random vectors of dimension $d_h$ has a
variance proportional to $d_h$. Unscaled, the scores grow with the dimension, the softmax
saturates (one weight near 1, the rest near 0) and its gradient vanishes. Dividing by
$\sqrt{d_h}$ keeps the scores in a range where the softmax can still learn.

**The causal mask.** All 256 positions predict their next token at once, in parallel. Position
$i$ must not see positions after $i$ — that would be reading the answer. So the scores $s_{ij}$
for $j > i$ are set to $-\infty$ before the softmax, which makes their weights exactly zero:

```
        j →  0   1   2   3
  i = 0     ✓   ·   ·   ·
  i = 1     ✓   ✓   ·   ·
  i = 2     ✓   ✓   ✓   ·
  i = 3     ✓   ✓   ✓   ✓
```

This is what makes training efficient: one forward pass over a sequence of 256 tokens yields 256
training examples (*predict token 1 from token 0, token 2 from tokens 0–1, ...*), every one of
them honest. Training on all prefixes at once, with the true previous tokens as input, is called
**teacher forcing**.

**Multiple heads.** A single attention can only blend one way per position. The model runs
several in parallel — **heads** — each with its own $W_Q, W_K, W_V$ over a slice of the vector:
here 6 heads of $d_h = 64$ dimensions ($6 \times 64 = 384$). One head might track the subject of
the sentence, another the previous punctuation, another the speaker of a dialogue line; nobody
assigns these roles, they emerge from training. The heads' outputs are concatenated and mixed by
a last learned matrix $W_O$ (`b.wo`).

In cleave, attention is one algebra of the standard library, `CausalAttention`
(`stdlib/nn/nn.cleave`): per sequence and per head, the two matrix products go to BLAS
(`sgemm`), and its gradient is **declared** as a whole (an `adjoint` rule) rather than derived
operation by operation — it recomputes the attention weights once instead of storing them.

### The MLP: each position thinks on its own

After attention has gathered information, each position processes its vector **independently**
with a small two-layer network — the **MLP** (multi-layer perceptron), or feed-forward layer:

$$\text{mlp}(x) = \text{GELU}(x W_1 + b_1)\, W_2 + b_2$$

from 384 dimensions up to 1,536 ($4\times$) and back down to 384. GELU is a smooth version of
"keep positive values, zero out negative ones":

$$\text{GELU}(x) \approx \tfrac{1}{2}\, x \left(1 + \tanh\!\left(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right)\right)$$

Without a non-linearity, any stack of layers would collapse into a single matrix product. The
MLPs hold about two thirds of each block's parameters; interpretability research suggests they
act as a large learned **key-value memory** of patterns and facts (Geva et al., 2021), while
attention routes information between positions.

### Residual connections: the stream

Each sub-layer **adds** its output to its input (`x + attention(...)`, `x2 + mlp(...)`) rather
than replacing it. Think of the vector at each position as a **residual stream** running through
the whole network: every attention and MLP reads from it and writes a small update into it
(Elhage et al., 2021). Two consequences:

- information from early layers (the token's identity, its position) stays available to all later
  ones;
- the gradient has a direct path from the loss back to every layer (the derivative of $x + f(x)$
  always contains the identity), which is what makes deep stacks trainable at all.

### LayerNorm: keeping the numbers in range

Before each sub-layer, the vector is **normalized**: shifted to mean 0 and scaled to variance 1
across its 384 numbers, then multiplied and shifted by two learned vectors $g, b$:

$$\text{layer\_norm}(x) = g \odot \frac{x - \mu}{\sqrt{\sigma^2 + 10^{-5}}} + b$$

It keeps activations at a stable scale whatever the depth, so training doesn't diverge.
Normalizing *before* each sub-layer (**pre-LN**, as in GPT-2) rather than after leaves the
residual stream itself untouched, which trains more stably.

### The head: scores for every next token

After the 8 blocks and a final LayerNorm, a dense layer (`head`) turns each position's 384
numbers into 4,096 **logits**, which the **softmax** turns into probabilities (section 3), and
the loss at that position is the surprise $-\ln p_{\text{target}}$ (section 3). The kernel's
`sparse_cross_entropy` computes it with the max subtracted first (so `exp` never overflows) and
declares its gradient directly — a famously simple one: $p - \text{onehot}(\text{target})$. The
gradient pushes the right token's score up by how far its probability is from 1, and every other
token's down by its probability.

### Sizes and parameters

| `define` | value | meaning |
|---|---|---|
| `VOCAB` | 4,096 | tokens |
| `CONTEXT` | 256 | positions per sequence |
| `WIDTH` | 384 | the vector at each position (the *model dimension*) |
| `HEAD` | 64 | dimensions per attention head: 6 heads |
| `MLP` | 1,536 | the MLP's hidden width, $4 \times$ `WIDTH` |
| blocks | 8 | the fields `b1`..`b8` of `struct Gpt` |
| `ROWS` | 8,192 | 32 sequences × 256 positions per batch |

Counting the learned numbers:

| part | shape | parameters |
|---|---|---|
| token embedding | 4096 × 384 | 1,572,864 |
| position embedding | 256 × 384 | 98,304 |
| per block: Q, K, V, O | 4 × (384 × 384 + 384) | 591,360 |
| per block: MLP | 384 × 1536 + 1536 + 1536 × 384 + 384 | 1,181,568 |
| per block: 2 LayerNorms | 4 × 384 | 1,536 |
| 8 blocks | 8 × 1,774,464 | 14,195,712 |
| final LayerNorm | 2 × 384 | 768 |
| head | 384 × 4096 + 4096 | 1,576,960 |
| **total** | | **17,444,608** |

About 17 million parameters: around a hundredth of GPT-2's largest version, a ten-thousandth of
today's large models. Its weights take 70 MB.

## 9. Training: gradients, Adam, and the schedule

### The gradient

Training means adjusting the 17 million parameters to lower the loss. For that we need the
**gradient**: for each parameter, how much the loss would change if that parameter moved a
little. Computing it efficiently for millions of parameters is **reverse-mode automatic
differentiation** (backpropagation): run the model forward, then propagate derivatives backward
from the loss through every operation, applying the chain rule. The backward pass costs about
twice the forward pass, whatever the number of parameters.

In PyTorch, the gradient is recorded at run time (`loss.backward()`). In cleave, it is
**derived by the compiler**, from the source of the loss function:

```
gpt_grad_micro = grad(gpt_loss_micro, m);
```

`grad` produces a new function that returns, for the model `m`, a value of the same type `Gpt`
holding every parameter's gradient. Each operation of the standard library declares its own
derivative (`derivative` and `adjoint` rules in its algebra); the compiler composes them and
optimizes the result like any other code.

### Adam

The simplest update is **gradient descent**: move each parameter a small step against its
gradient, $\theta \leftarrow \theta - \eta\, g$, with $\eta$ the **learning rate**. nanoLM uses
**Adam** (Kingma & Ba, 2014), which keeps two running averages per parameter — of the gradient
($m$, momentum) and of its square ($v$) — and steps by

$$\theta \leftarrow \theta - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon}$$

with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Dividing by $\sqrt{\hat v}$ gives
each parameter its own step size: parameters with large, noisy gradients take careful steps,
rarely-updated ones (the embedding rows of rare tokens) take bolder ones. Adam's two averages
double the memory of the parameters; they are saved in the checkpoint so a resumed run continues
exactly.

### The learning-rate schedule

`learning_rate` in the kernel:

- **warm-up**: from nearly 0 up to $10^{-3}$ over the first 200 steps. At the start, the weights
  are random and Adam's averages empty; large steps then would throw the model into a bad region.
- **linear decay** down to $10^{-4}$ at step 20,000, flat after. Large steps explore early; small
  steps settle into a minimum late.

The rate depends only on the step number, so a resumed run follows the same schedule.

### Initialization

Weights start random, with a scale chosen so that signals neither explode nor vanish through the
layers: **Xavier** initialization (uniform in $\pm\sqrt{6/(\text{fan}_{in} + \text{fan}_{out})}$)
for most matrices, **He** initialization (standard deviation $\sqrt{2/\text{fan}_{in}}$) for the
MLP's first layer, biases at zero, LayerNorm gains at one (`new_block`). The loss at step 0 is
therefore close to $\ln 4096$: the model starts by guessing uniformly.

### Train and validation losses

At the end of each round, the kernel measures the loss on the last 4 training batches (just
trained on) and on 4 validation batches (never seen). Early in training they move together.
They are worth watching side by side: a widening gap means memorization. With 161 million
training tokens and one pass over them (section 12), this run is in no danger of it.

## 10. Parallelism: one step on eight cores

One training step is about $6 \times 17.4\text{M} \times 8192 \approx 0.86$ trillion
floating-point operations (the classic estimate: 2 per parameter per token forward, 4 backward),
plus attention. nanoLM spreads it over the CPU's cores by **data parallelism**: the batch's 8,192
rows are split into 8 **micro-batches** of 4 whole sequences, each micro-batch's gradient is
computed in its own task, and the 8 gradients are summed. Since the loss is a sum over rows, the
sum of the parts' gradients is exactly the batch's gradient.

In cleave, with `spawn` (`doc/plan-spawn.md`):

```
let g0 = spawn gpt_grad_micro(x0, y0, pm, m);
...
let g7 = spawn gpt_grad_micro(x7, y7, pm, m);
sync;
accumulate(accumulate(accumulate(g0, g1), accumulate(g2, g3)), accumulate(accumulate(g4, g5), accumulate(g6, g7)))
```

Each `spawn` starts a task (on the OpenMP runtime); reading `g0` waits for it. The gradients are
summed in a fixed balanced tree, so the result is the same whatever the number of threads. The
evaluation at the end of each round is parallel the same way (`parallel_loss`). The summing and
the Adam update are themselves parallel over the model's fields (`Accumulate` and `Optimizer` in
`stdlib/optim`).

On an 8-core Ryzen 7 9700X, a step takes about 1.1 seconds; 20,000 steps, about 6 hours.

## 11. Generation: writing one token at a time

`write` continues a prompt (`generate` in the kernel, `src/generate.rs` on the host side):

1. The prompt is normalized and tokenized exactly like the corpus.
2. The model computes the logits for the position after the last token.
3. One token is **drawn** at random from the softmax of these logits, divided by a
   **temperature** $\tau$: $p_t \propto e^{z_t/\tau}$, keeping only the **nucleus** of the most
   likely tokens (`sample_row_top_p`, below).
4. The token is printed and appended to the context; back to 2.

The **temperature** trades safety for variety. At $\tau \to 0$ the model always picks its top
choice (repetitive, often looping); at $\tau = 1$ it samples its true distribution (varied, more
errors); $0.7$–$0.9$ is the usual compromise. Generation is random: the same prompt with
another seed gives another text.

**Nucleus (top-p) sampling** (Holtzman et al., 2019) fixes what temperature alone can't: the
**tail**. Each of the thousands of unlikely tokens has a tiny probability, but together they can
weigh 10 or 15%: drawing from the full distribution picks one of them every few tokens, and the
model must then continue from that mistake. Top-p keeps the smallest set of most likely tokens
whose probabilities add up to $p$ (0.9 by default) and draws only among them. The set adapts to
the model's confidence: one or two tokens when it is sure (after *"Il ouvrit la"*), dozens when
many continuations fit (after *"Le soir tombait sur"*). Temperature 0.7–0.9 with top-p 0.9 is the
usual setting of text generators. (The kernel finds the nucleus without sorting the 4,096 tokens:
the smallest weight a token must have to be kept is found by bisection.)

Once the context holds 256 tokens, the oldest one slides out: the model never sees more than its
last ~900 characters. And each new token recomputes the whole context from scratch — production
systems keep the keys and values of earlier positions (a **KV cache**) to avoid that; nanoLM
doesn't yet.

## 12. Scale: how big, how long, how good

How good a model can get depends mostly on three quantities: its **parameters**, the **data** it
trains on, and the **compute** spent. The **scaling laws** (Kaplan et al., 2020; Hoffmann et
al., 2022, "Chinchilla") found the loss falls smoothly and predictably as each grows, and that for
a fixed compute budget the best results come from about **20 training tokens per parameter**.

For nanoLM:

| | |
|---|---|
| parameters | 17.4 M |
| Chinchilla-optimal data | ~350 M tokens |
| one pass over the corpus | 161 M tokens = 19,700 steps of 8,192 tokens |
| this run | 20,000 steps, ~6 hours |

The run is slightly short of the optimal data for its size — the price of fitting in a night —
and sees each token only once, so it can't memorize. A reasonable expectation is a validation
loss around 3.2–3.4 nats/token (about 1.3–1.4 bits/char, perplexity ~25–30): grammatical text,
locally coherent over a few sentences, without a plot. For comparison, the first nanoLM (0.8 M
parameters, characters, Zola alone) took about the same time to train.

For scale: GPT-2's largest version had 1.5 billion parameters; Llama 3's largest, 405 billion,
trained on 15 trillion tokens — about $3 \times 10^9$ times this run's compute, on 16,000 GPUs
for 54 days.

## 13. The PyTorch twin

`bench/nanolm-pytorch/gpt.py` is the same model written with PyTorch, in the same order, reading
the same tokens, drawing the same batches, starting from the same initial weights (which
cleave's `bench` mode writes to `gpt_init.ckpt`). Its losses must match cleave's:

```
cargo run --release -p nanolm -- bench 0 3 100
cd bench/nanolm-pytorch; poetry run python gpt.py 0 3 100
```

They agree to the fourth decimal for the first hundreds of steps, then drift by about $10^{-3}$
— float rounding, and cleave's polynomial approximation of `tanh`. It is how we know the
compiler's derived gradient is right. It's also a speed reference: on the 6-layer, width-256
version of the model, cleave took 293 ms per step and PyTorch about 420 ms on the same machine.

## 14. Files

| file | role |
|---|---|
| `src/kernel.cleave` | everything the model computes: baselines, transformer, gradient, training loop, evaluation, generation |
| `src/corpus.rs` | downloading, cleaning and encoding the books; the alphabet |
| `src/bpe.rs` | the BPE tokenizer: training, encoding, decoding |
| `src/data.rs` | batches for the kernel, the tokenized corpus |
| `src/generate.rs` | the prompt and the printing of generated tokens |
| `src/main.rs` | the modes, the round reports (nats, bits/char, timings) |
| `books.txt` | the frozen list of books |
| `build.rs` | compiles `kernel.cleave` into the Rust binary at build time (`cleave-build`) |
| `../../bench/nanolm-pytorch/` | the PyTorch twin |
| `../../stdlib/nn/nn.cleave` | layers, attention, LayerNorm, GELU, cross-entropy |
| `../../stdlib/optim/optim.cleave` | Adam, gradient accumulation |
| `../../doc/plan-nanolm.md`, `../../doc/plan-spawn.md` | design notes |

The kernel is compiled ahead of time into the host binary, like any Rust dependency: no Python,
no interpreter, no runtime beyond cleave's small one. The program that trains the model is the
program that runs it.

## 15. Glossary and further reading

**Activation** — any intermediate vector computed by the model. **Batch** — the set of examples
of one training step. **Checkpoint** — saved weights (and optimizer state). **Context** — the
tokens the model sees when predicting. **Cross-entropy** — the average surprise; the loss.
**Embedding** — a learned vector per token (or per position). **Epoch** — one pass over the
training data. **Gradient** — the derivative of the loss with respect to every parameter.
**Head** — one of the parallel attentions of a block; also, the final dense layer. **Logits** —
unnormalized scores, before the softmax. **Nat / bit** — units of surprise ($\ln$ / $\log_2$).
**Overfitting** — learning the training text instead of the language. **Parameter / weight** — a
learned number. **Perplexity** — $e^{\text{loss}}$, the effective number of choices.
**Residual stream** — the per-position vector that every block reads and updates. **Softmax** —
turns scores into probabilities. **Temperature** — divides the logits before sampling.
**Token** — a unit of text, here a BPE subword. **Validation set** — text held out to measure
generalization.

Further reading:

- C. Shannon, *Prediction and Entropy of Printed English* (1951) — language as prediction,
  entropy of text.
- Y. Bengio et al., *A Neural Probabilistic Language Model* (2003) — embeddings + MLP.
- A. Vaswani et al., *Attention Is All You Need* (2017) — the transformer.
- A. Radford et al., *Language Models are Unsupervised Multitask Learners* (2019) — GPT-2, the
  shape nanoLM follows.
- R. Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016) — BPE.
- D. Kingma, J. Ba, *Adam: A Method for Stochastic Optimization* (2014).
- J. Ba et al., *Layer Normalization* (2016); D. Hendrycks, K. Gimpel, *GELUs* (2016).
- N. Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021) — the residual
  stream view.
- M. Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* (2021).
- J. Kaplan et al., *Scaling Laws for Neural Language Models* (2020); J. Hoffmann et al.,
  *Training Compute-Optimal Large Language Models* (2022).
- A. Karpathy, *nanoGPT* and the video *Let's build GPT: from scratch, in code, spelled out* —
  the same model in PyTorch, built step by step.
