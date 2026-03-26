# Generative Recommendations: A Mechanistic Guide

---

## 1. The Broken Foundation

You run a video platform with 500 million videos. Each video gets an entry in a lookup table:

```
video_38291047 → [0.23, -0.71, 0.45, ..., 0.12]   # embedding: float32[128]
video_38291048 → [0.89, 0.14, -0.33, ..., -0.56]
```

The full table is a matrix: `embedding_table: float32[500_000_000, 128]`. Each row is learned during training. The model sees "user watched video_38291047 then clicked video_38291049" and adjusts those rows to be more compatible.

This is broken in three ways:

**No generalization.** Video_38291047 and video_38291048 might both be pasta recipes, but the model can't know that. The IDs are arbitrary database keys. Every embedding is learned independently from scratch.

**Cold start.** A new video's embedding is random noise. It needs hundreds of interactions before the model knows anything about it.

**The task is too easy.** The model predicts $P(\text{click} | \text{user}, \text{item}, \text{context})$: a binary classification. Embed features, cross them, MLP, sigmoid. A shallow network can memorize this. Adding layers doesn't help because dot-product-style feature crossing has no deep hierarchical structure. Scaling laws never emerge.

The symptom is the **one-epoch phenomenon**: train for one epoch, fine. Train for two, test performance drops. The sparse ID embeddings overfit to training-time co-occurrence patterns after a single pass. They see each ID only a handful of times, fit it to the training distribution, and can't generalize. A second epoch reinforces the overfit.

Two independent problems need fixing: **item IDs have no structure** (so the model can't generalize) and **the training task is too easy** (so bigger models don't help). Everything that follows pulls one or both of these levers.

---

## 2. Building a Vocabulary for Items

### The idea

Replace arbitrary IDs with learned hierarchical codes. Instead of `video_38291047`, assign `(42, 8, 3, 177)`: a sequence of codes from coarse to fine, where items sharing prefixes are semantically similar. The model gets structure to exploit.

### What a codebook is

A codebook is a clustering of a vector space. You have 500 million item embeddings, each a vector in some high-dimensional space. A codebook with $K = 256$ entries partitions this space into 256 regions, each with a center point (the code vector). Every item gets assigned to its nearest center.

```
codebook: float32[K, d_latent]     # K=256 code vectors
```

One codebook gives each item a 1-digit code. But 256 buckets for 500M items = ~2M per bucket. Too coarse. Making $K$ = 500M brings you back to atomic IDs.

### The residual trick

Instead of one huge codebook, stack multiple small ones. Each refines the previous one's error.

**Level 0**: Assign each item to its nearest code in codebook 0. The residual is the error:

```python
# latent: float32[d_latent]              - one item's latent vector (from encoder, explained below)
# codebook_0: float32[K, d_latent]       - 256 code vectors

distances = ||latent - codebook_0||               # float32[K]
c_0 = argmin(distances)                           # int, the assigned code
residual_1 = latent - codebook_0[c_0]             # float32[d_latent], the leftover error
```

**Level 1**: Cluster the *residual* with codebook 1:

```python
distances = ||residual_1 - codebook_1||           # float32[K]
c_1 = argmin(distances)                           # int
residual_2 = residual_1 - codebook_1[c_1]         # float32[d_latent]
```

Repeat for $m$ levels. The output is the Semantic ID: $(c_0, c_1, \ldots, c_{m-1})$.

The quantized approximation is the sum of all assigned code vectors:

```python
quantized = codebook_0[c_0] + codebook_1[c_1] + ... + codebook_{m-1}[c_{m-1}]
# float32[d_latent], approximates the original latent vector
```

With $K = 256$ and $m = 4$ levels, you get $256^4 \approx 4.3$ billion unique addresses from a vocabulary of only $4 \times 256 = 1024$ tokens. Two pasta videos share prefix `(42, 8, 3)`. A skateboarding video starts with `(17, ...)`. The hierarchy is meaningful by construction. Items sharing prefixes are nearby in embedding space.

**What Semantic IDs deliberately don't capture.** Two identical pasta tutorials, one with 10 million views and one with 100, get the same Semantic ID. So do a trending video and a stale one. Semantic IDs encode what an item *is* (content similarity), not how popular, fresh, or trending it is. This is by design: popularity changes hourly, but Semantic IDs are static (or retrained weekly at most). You can't bake a volatile signal into a code that's meant to be stable. Where real-time signals like popularity, freshness, and creator reputation enter the system is a real design tension. We'll address the mechanism in Section 4 (token construction) and the deeper tradeoffs in Section 8.

### Making embeddings clustering-friendly first

Before we quantize anything, we need to fix the input embeddings. Content embeddings from a vision-language model capture what an item *looks like*: its visual style, text, metadata. But a cooking tutorial and a kitchen gadget review might look different while serving the same user need. And two visually similar videos (both dark-lit vlogs) might serve completely different audiences.

Fine-tune the embeddings with contrastive learning *before* quantization: pull together items engaged by overlapping user sets, push apart items with disjoint audiences.

$$\mathcal{L}_{\text{collab}} = -\log \frac{\exp(\cos(\texttt{emb}_i,\ \texttt{emb}_j) / \tau)}{\sum_{k \in \text{batch}} \exp(\cos(\texttt{emb}_i,\ \texttt{emb}_k) / \tau)}$$

where $(i, j)$ are items engaged by overlapping users ("collaboratively similar") and $\tau$ is a temperature that sharpens or softens the distribution. The numerator pulls similar items together; the denominator pushes everything else apart.

After this alignment, the embedding space reflects *user behavior similarity*, not just content similarity. Items that serve similar user needs are nearby even if they look different. This step is critical because it makes the embedding space much more clustering-friendly. Everything downstream depends on it.

### Two ways to build codebooks from aligned embeddings

We now have collaboratively-aligned content embeddings. We need to assign codes. There are two approaches, and the simpler one is what's used in the largest production system.

### Approach 1: RQ-Kmeans (the simple, production approach)

Just run k-means on the aligned embeddings. Compute residuals. Run k-means on the residuals. Repeat.

```python
# RQ-Kmeans: no encoder, no decoder, no gradients for codebooks
# Input: content embeddings AFTER collaborative alignment

residuals = aligned_content_embeddings              # float32[N_items, d_content]
for level in range(num_levels):
    codebook[level] = kmeans(residuals, K=256)      # cluster centers
    codes[level] = assign_nearest(residuals, codebook[level])  # int[N_items]
    residuals = residuals - codebook[level][codes[level]]      # leftover error
```

That's it. No neural network. No gradients. No training loop. No stop-gradient tricks. You run k-means offline on your catalog, store the codebooks and codes, and you're done.

Kuaishou's OneRec uses RQ-Kmeans in production. Their technical report shows it achieves **perfect 1.0 codebook utilization** at all levels (every code is used), higher entropy (more balanced token distribution), and better reconstruction quality than the more complex alternative (RQ-VAE). The collaborative alignment step does the heavy lifting of making the space clustering-friendly. Once the space is well-shaped, plain k-means is hard to beat.

### Approach 2: RQ-VAE (the learned approach)

RQ-VAE wraps the residual quantization in a neural network: an encoder MLP before the codebooks and a decoder MLP after. The encoder learns to rearrange the embedding space to be more quantization-friendly. The decoder verifies that quantization didn't destroy important information. Everything is trained jointly with gradients.

**Where RQ-VAE came from.** RQ-VAE was invented for image generation (Lee et al., CVPR 2022), not recommendation. In images, you *need* a learned encoder-decoder because you're compressing raw pixel data into discrete codes. There's no pre-existing "collaboratively-aligned embedding space" for images. The encoder learns the compression from scratch. Google's TIGER (Rajput et al., 2023) borrowed RQ-VAE for recommendation, using it to convert item content embeddings into Semantic IDs.

**What the encoder-decoder adds over plain k-means:**
The encoder can rearrange the embedding space before quantization: pull together items that are far apart in the raw space but should share codes, spread apart items that are close but should get different codes, reshape elongated clusters into rounder ones. The decoder provides a reconstruction loss that tells the system "your quantization destroyed information about X," so the encoder adjusts to preserve it.

**When this matters:** If your collaborative alignment is weak (limited interaction data, cold-start heavy catalog) or your content embeddings are messy (high-dimensional, elongated clusters), the RQ-VAE encoder can compensate. RQ-Kmeans can't. It's stuck with whatever space you give it.

**Why we'll explain the training in detail:** RQ-VAE is the foundational method in the literature (TIGER, GRID, many others). Understanding how it trains teaches you stop-gradient and straight-through estimator, which transfer to VQ-VAE in audio, image tokenizers, and discrete latent models generally. And you need to understand the complex approach to know when the simple one is sufficient.

### RQ-VAE: the architecture

There are exactly three groups of trainable parameters:

```python
# GROUP 1: Encoder - a small MLP (2-3 layers)
# Takes a content embedding and maps it to a latent space 
# that's easier to quantize.
encoder_weights = {
    W1: float32[d_content, d_hidden],    # e.g. [768, 512]
    b1: float32[d_hidden],
    W2: float32[d_hidden, d_latent],     # e.g. [512, 256]
    b2: float32[d_latent],
}

# GROUP 2: Codebooks - just lists of vectors, one list per level
# These ARE the cluster centers. They start random and learn to 
# move to where the data is.
codebooks = {
    level_0: float32[K, d_latent],       # e.g. [256, 256] - 256 centers
    level_1: float32[K, d_latent],       # another 256 centers
    level_2: float32[K, d_latent],
    level_3: float32[K, d_latent],
}

# GROUP 3: Decoder - a small MLP (mirrors the encoder)
# Takes the quantized representation and tries to reconstruct 
# the original content embedding.
decoder_weights = {
    W1: float32[d_latent, d_hidden],
    b1: float32[d_hidden],
    W2: float32[d_hidden, d_content],
    b2: float32[d_content],
}
```

That's it. No Transformer here. RQ-VAE is a small model. The encoder and decoder are tiny MLPs. The codebooks are just matrices of vectors. Total parameter count is modest (a few million at most).

### What a single training step looks like

You sample a batch of items from your catalog and process them all in parallel:

```python
# ============================================================
# STEP 1: Sample a batch of items
# ============================================================
batch_content_embs = sample_from_catalog(batch_size=512)
# float32[512, d_content] - 512 items, each with a content embedding
# These content embeddings are PRE-COMPUTED (frozen) from a vision-language model
# that processed each item's images, title, tags, description, etc.
# RQ-VAE doesn't train the VLM - it takes its output as input.

# ============================================================
# STEP 2: Encode the whole batch
# ============================================================
batch_latents = encoder_mlp(batch_content_embs)
# float32[512, d_latent] - each item mapped to latent space
# This is just a matrix multiply + activation, applied to each item independently.
# No interaction between items in the batch.

# ============================================================
# STEP 3: Quantize each item independently
# ============================================================
# Every item finds its own nearest code at each level.
# Items don't interact - each one's codes depend only on its own latent vector.

all_codes = []                # will be [512, num_levels] - one Semantic ID per item
all_residuals = []            # saved for the loss
all_code_vectors_used = []    # saved for the loss

batch_residuals = batch_latents    # float32[512, d_latent]

for level in range(num_levels):    # e.g. 4 levels
    # Distance from each item to each code vector at this level
    distances = cdist(batch_residuals, codebooks[level])
    # float32[512, K] - distance[i, k] = ||residual_i - code_k||
    
    # Each item picks its nearest code (independently)
    codes_this_level = argmin(distances, dim=1)    # int[512]
    
    # Look up the code vectors that were selected
    selected_vectors = codebooks[level][codes_this_level]
    # float32[512, d_latent] - one code vector per item
    
    # Save for loss computation
    all_residuals.append(batch_residuals.clone())
    all_code_vectors_used.append(selected_vectors)
    all_codes.append(codes_this_level)
    
    # Update residuals: subtract the selected code vector
    batch_residuals = batch_residuals - selected_vectors
    # float32[512, d_latent] - the leftover error, to be refined at next level

# ============================================================
# STEP 4: Reconstruct
# ============================================================
batch_quantized = sum(all_code_vectors_used)    # float32[512, d_latent]
# For each item: sum of its 4 selected code vectors ≈ its latent vector

batch_reconstructed = decoder_mlp(batch_quantized)
# float32[512, d_content] - attempt to recover original content embeddings

# ============================================================
# STEP 5: Compute loss (averaged over the 512 items in the batch)
# ============================================================
loss = compute_rqvae_loss(
    batch_content_embs,        # original inputs
    batch_reconstructed,       # reconstructed outputs
    all_residuals,             # residuals at each level (for quantization loss)
    all_code_vectors_used,     # code vectors selected at each level
)

# ============================================================
# STEP 6: Backprop and update
# ============================================================
loss.backward()
optimizer.step()   # updates encoder_weights, codebooks, decoder_weights
```

Notice: every item in the batch is processed **independently**. There's no attention, no interaction between items. The batch dimension is purely for parallelism and gradient averaging. RQ-VAE is a per-item model. It maps one content embedding to one Semantic ID.

### The loss function: who gets updated by what?

Now the tricky part. We have three groups of parameters (encoder, codebooks, decoder) and we need gradients to flow to all of them. But `argmin` in the quantization step is non-differentiable. It has zero gradient almost everywhere. How does this work?

**The big picture first.** Training has a forward pass and a backward pass, and they do very different things:

**Forward pass** (what you saw in the training loop above): Each item finds its nearest code at each level. This is pure nearest-neighbor assignment, just like the assignment step in k-means. No gradients are involved. The codebooks don't move during the forward pass. You're just asking: "given where the centers currently are, which center is each item closest to?"

**Backward pass** (what we're about to explain): Three separate gradient signals update three separate parameter groups.
1. The **decoder** gets a straightforward gradient from the reconstruction loss. Nothing tricky here.
2. The **codebook centers** get updated to better represent the data assigned to them. Here's how, concretely. Suppose code vector 42 is currently at position `[0.5, 0.3]` and three items got assigned to it during this batch, with residuals at `[0.7, 0.4]`, `[0.6, 0.2]`, and `[0.8, 0.3]`. We write a loss term: `||code_42 - residual||^2` for each of these three items (with the residual frozen via stop-gradient, explained below). The gradient of `||code_42 - residual||^2` with respect to `code_42` is `2*(code_42 - residual)`, pointing *away from the residual*. The optimizer subtracts this gradient, so the code vector moves *toward* the residual. Across many batches, this is equivalent to k-means moving the center to the mean of its assigned points, except it happens incrementally via SGD rather than in one shot. The codebook centers don't move during the forward pass (that's just assignment). They move during `optimizer.step()`, after gradients have been computed.
3. The **encoder** gets two gradients: one telling it "produce latents that reconstruct well" and another telling it "produce latents that land close to their assigned code, so the assignments are stable."

The problem is that `argmin` sits between the encoder and the codebooks in the computation graph, blocking the normal gradient flow. So we need two tricks to manually route gradients around the blockage. The tricks are just plumbing. The conceptual picture above is the real content.

To solve this, we need two tricks that come up frequently in discrete/quantized systems. Let's understand each one before seeing how they're used.

### What is stop-gradient?

In normal backpropagation, when you compute a loss involving two variables $a$ and $b$, the gradient flows to *both*. If your loss is $\|a - b\|^2$, the gradient with respect to $a$ is $2(a - b)$ (pulling $a$ toward $b$) and the gradient with respect to $b$ is $2(b - a)$ (pulling $b$ toward $a$). Both move.

Sometimes you want only one to move. Stop-gradient, written $\text{sg}[\cdot]$, tells the autograd system: "treat this value as a frozen constant during backprop. It contributed to the forward pass, but don't compute or propagate gradients through it."

```python
# Normal: both a and b receive gradients
loss = (a - b) ** 2
loss.backward()    # grad_a = 2(a-b), grad_b = 2(b-a)

# Stop-gradient on b: only a receives gradients
loss = (a - sg(b)) ** 2
loss.backward()    # grad_a = 2(a-b), grad_b = 0

# Stop-gradient on a: only b receives gradients
loss = (sg(a) - b) ** 2
loss.backward()    # grad_a = 0, grad_b = 2(b-a)
```

In PyTorch, `sg(x)` is `x.detach()`. In JAX, it's `jax.lax.stop_gradient(x)`. The forward computation is identical in all three cases (the loss value is the same). The difference is purely in which parameters get updated during the backward pass.

Why is this useful? When you have two groups of parameters that both appear in a loss term but you want to update them *separately with different objectives*. You can write two loss terms using stop-gradient to route gradients to the right group.

### What is the straight-through estimator?

The quantization step does this:

```python
# Forward pass:
code = argmin(distances)                    # discrete: pick nearest code
quantized = codebook[code]                  # look up that code's vector
```

`argmin` returns an **integer index**, not a continuous value. It answers "which code is nearest?" (an integer like 42), not "how near is the nearest code?" (a smooth distance). If you nudge `latent` by a tiny epsilon, either the same code is still nearest (index stays 42, output unchanged, gradient = 0) or a different code becomes nearest (index jumps from 42 to 43, a discontinuous step, not differentiable). Then `codebook[code]` is a table lookup by that integer. You can't differentiate "look up row 42" with respect to 42. So during backprop, no gradient flows from `quantized` back to `latent`. The encoder gets no learning signal from the reconstruction loss.

The straight-through estimator is a hack: during the forward pass, use the actual quantized value. During the backward pass, pretend quantization didn't happen and copy the gradient straight through.

```python
# Forward pass: quantized = codebook[argmin(distances)]  (actual discrete lookup)
# Backward pass: grad_latent = grad_quantized             (pretend quantized == latent)
```

In PyTorch, this is typically implemented as:

```python
quantized = codebook[code]
# Trick: add and subtract latent so that the VALUE is quantized
# but the GRADIENT flows through latent
quantized_st = latent + (quantized - latent).detach()
# Forward: quantized_st == quantized  (because latent + quantized - latent = quantized)
# Backward: grad flows to latent      (because .detach() kills the quantized-latent path)
```

This is biased (the gradient is approximate, not exact), but it works well in practice. The encoder learns "if I shift my output slightly in direction $d$, the reconstruction gets better/worse by this much," even though the actual quantized value didn't shift (it snapped to a code).

### Applying both tricks to RQ-VAE

**Reconstruction loss**: did quantization destroy information?

$$\mathcal{L}_{\text{recon}} = \frac{1}{B}\sum_{i=1}^{B}\| \texttt{content\_emb}_i - \texttt{reconstructed}_i \|^2$$

Averaged over the batch. This wants to update the encoder (to produce better latent vectors), the codebooks (to approximate the latents better), and the decoder (to reconstruct better). But `argmin` blocks gradients from flowing through the quantization step. **The straight-through estimator** patches this: during backprop, gradients flow from `reconstructed` → decoder → `quantized` → (straight-through) → `latent` → encoder. The encoder and decoder both get updated. The codebooks still don't, because the straight-through sends gradients to `latent`, not to the codebook vectors.

**Quantization loss: routing gradients to the codebooks (and back to the encoder):**

The codebooks need their own gradient signal. We use **stop-gradient** to create two auxiliary losses that manually route gradients:

$$\mathcal{L}_{\text{quantization}} = \frac{1}{B}\sum_{i=1}^{B}\sum_{\text{level}} \underbrace{\| \text{sg}[\texttt{residual}_i] - \texttt{code\_vector}_i \|^2}_{\text{Term 1: moves codebook centers}} + \beta \underbrace{\| \texttt{residual}_i - \text{sg}[\texttt{code\_vector}_i] \|^2}_{\text{Term 2: stabilizes encoder output}}$$

**Term 1 in plain English**: Stop-gradient on the residual. Only the code vector receives gradients. The gradient is $2 \cdot (\texttt{code\_vector}_i - \texttt{residual}_i)$, pointing from the code vector toward the residual. The optimizer takes a step, and the code vector moves closer to the residual. This is how codebook centers migrate to where the data actually is. Same as a k-means centroid update, except via SGD across many batches.

**Term 2 in plain English**: Stop-gradient on the code vector. The gradient flows to the residual, which flows back through the encoder. The gradient is $2\beta \cdot (\texttt{residual}_i - \texttt{code\_vector}_i)$, telling the encoder: "move your output closer to the code vector you were assigned to." This prevents the encoder from being flighty. If it keeps shifting its outputs, the code assignments keep changing, and the codebooks never stabilize. $\beta = 0.25$ is typical.

**Putting it together: what each parameter group receives:**

| Parameter group | Updated by | What happens |
|----------------|------------|-------------|
| Encoder weights | Reconstruction loss (via straight-through) + Term 2 | Encoder learns to produce latents that are easy to quantize AND reconstruct well |
| Codebook vectors | Term 1 only | Cluster centers migrate toward the data they're assigned |
| Decoder weights | Reconstruction loss (directly) | Decoder learns to reconstruct from the quantized representation |

**The full loss:**

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{quantization}}$$

One `loss.backward()` call computes all the gradients. The stop-gradient operators and straight-through estimator ensure each parameter group gets exactly the right signal. One `optimizer.step()` updates everything simultaneously.

### What goes wrong with RQ-VAE: codebook collapse

Some codes get assigned to many data points early on, get frequently updated, attract more data. Others drift away and die. You end up with 20 active codes out of 256.

**First fix**: K-means initialization + EMA updates + reset dead codes periodically.

**Still collapsing?** This is one reason RQ-Kmeans wins empirically. K-means directly computes cluster centers as the mean of assigned points, which guarantees every center is near actual data. Gradient-based codebook learning (Term 1) does the same thing incrementally via SGD, but it's more susceptible to the rich-get-richer dynamic that causes collapse. RQ-Kmeans achieves perfect 1.0 codebook utilization by construction.

### Living with a changing catalog

Everything above describes building Semantic IDs once. But your catalog isn't static. New videos are uploaded every minute. Old ones are removed. User behavior shifts seasonally. The Semantic ID system must handle this continuously, and the way it handles it has cascading consequences for everything downstream.

**New items (the easy case).** A new video is uploaded. You run its content through the vision-language model to get `content_emb`, collaboratively align it if your alignment model supports incremental updates, then quantize it against the frozen codebooks. With RQ-Kmeans, this is just nearest-neighbor assignment at each level. With RQ-VAE, it's a forward pass through the frozen encoder plus nearest-neighbor assignment. Either way, no retraining needed. The new video immediately gets a meaningful ID that shares prefixes with similar existing videos. This is one of the biggest practical advantages of Semantic IDs over atomic IDs: a new video with atomic ID `video_999999999` starts with a random embedding and zero information. A new video with Semantic ID `(42, 8, 3, 215)` is instantly known to be similar to other `(42, 8, 3, *)` items. The downstream Transformer already knows how to handle that prefix.

The limitation: the codebooks were built on the old catalog's distribution. If the new video is genuinely unlike anything seen during codebook training, say your cooking platform suddenly adds gaming content, the existing codebooks may not carve the space well for it. The residual errors will be larger, the Semantic ID less precise. This degrades gracefully (the coarse codes are still meaningful, just the fine codes are noisy) but accumulates over time.

**Periodic codebook retraining (the hard case).** Eventually the catalog drifts enough that the codebooks need retraining. You rerun RQ-Kmeans (or retrain RQ-VAE) on the updated catalog. New codebooks are learned. And now every item's Semantic ID potentially changes.

This is the hard problem. The video that was `(42, 8, 3, 177)` might become `(38, 12, 7, 201)`. The downstream Transformer spent its entire training learning that `(42, 8, 3, *)` means "Italian pasta content." After rebuilding codebooks, that knowledge is invalidated. The Transformer's entire learned vocabulary is broken. So is the trie of valid IDs. So are any cached user sequence representations.

**The downstream cascade:**
- The Transformer must be retrained or fine-tuned on the new Semantic IDs. Retraining from scratch is expensive but clean. Fine-tuning on new IDs risks catastrophic forgetting. The model partially remembers old ID patterns that no longer exist, creating ghost associations.
- The valid-ID trie must be rebuilt entirely.
- All cached user representations (KV caches, precomputed encoder outputs) are stale and must be recomputed.
- If you're running A/B tests, the old and new models produce incomparable Semantic IDs. You can't mix them.

**Practical strategies to manage this:**

*Gradual codebook updates.* Instead of rebuilding codebooks from scratch, update them incrementally using EMA: $\texttt{codebook} \leftarrow \gamma \cdot \texttt{codebook} + (1 - \gamma) \cdot \texttt{new\_centers}$. With $\gamma = 0.99$, codebooks shift slowly. Most items' Semantic IDs stay the same or change by one code at the finest level. The downstream Transformer can absorb small ID perturbations without retraining. Its learned patterns at the coarse levels survive.

*Scheduled retraining with staged rollout.* Rebuild codebooks weekly or monthly on the full updated catalog. Retrain the Transformer on the new IDs. Roll out in stages: 1% of traffic on the new model, monitor metrics, scale up. Keep the old model as fallback. This is operationally complex but standard in production ML.

*Training the Transformer to be robust to ID noise.* During Transformer training, randomly perturb a small fraction of Semantic IDs (swap a fine-level code for a nearby code). This teaches the model that IDs are approximate. It learns to rely on coarse prefixes (which are stable across retrainings) more than fine suffixes (which are volatile). A form of data augmentation specific to the Semantic ID setting.

The bottom line: Semantic IDs are not a "train once and deploy" component. They're a living part of the system that requires an operational cadence: monitor codebook utilization, track how much the catalog has drifted since last retraining, schedule retraining before degradation becomes visible in online metrics. Staff-level ownership of this system means owning this lifecycle, not just the initial architecture.

---

## 3. Training Like a Language Model

### From vocabulary to language

We now have a vocabulary of ~1024 tokens (256 codes × 4 levels) and every item is a "word" of 4 tokens. A user's engagement history becomes a sentence:

```
User history (3 items watched):
  pasta video   → (42, 8, 3, 177)
  sushi video   → (42, 15, 91, 22)
  skateboarding → (17, 203, 44, 9)

Token sequence: [42, 8, 3, 177, 42, 15, 91, 22, 17, 203, 44, 9]
```

We can now train a Transformer on these sequences exactly like a language model. Given the history, predict the next item's Semantic ID, one code at a time:

$$\mathcal{L}_{\text{next-token}} = -\sum_{t=1}^{T-1} \log P(\texttt{token}_{t+1} \mid \texttt{token}_1, \ldots, \texttt{token}_t)$$

Each term asks: "at position $t$, how surprised was the model by what actually came next?" Lower surprise (higher probability assigned to the true next token) = lower loss.

### Why this fixes scaling

The discriminative task (binary click prediction) gives 1 bit of supervision per example. The generative task (predict the next token from a vocabulary of 256) gives $\log_2 256 = 8$ bits per prediction, and a sequence of length $T$ gives $T - 1$ predictions. For a user with 1000 engagements encoded as 4-code Semantic IDs, that's 3999 prediction tasks per user, each over a 256-way vocabulary. Versus 1000 binary labels in the discriminative setup.

More importantly, the task is *harder*. The model must learn the full joint distribution: the probability of seeing this *entire sequence* of items in this order:

$$P(\texttt{token}_1, \ldots, \texttt{token}_n) = \prod_{t=1}^{n} P(\texttt{token}_t \mid \texttt{token}_1, \ldots, \texttt{token}_{t-1})$$

Each factor asks: "given everything this user has done so far, what's the probability of this specific next token?" The model must understand temporal patterns, item relationships, preference evolution, and the compositional structure of codes. A single Transformer layer can't do it. Depth helps. Scaling laws emerge.

### Why depth specifically matters: a toy example

Consider this user sequence (showing only the first code of each Semantic ID for brevity):

```
[42, 42, 42, 17, 17, 42, 42, 17, ???]
```

**What a 1-layer model can learn**: First-order transitions. "After seeing code 42, code 42 is likely again (60%), code 17 is possible (40%)." It computes one round of attention over the sequence, producing a weighted mix of what it's seen. It can learn "the most recent token was 17, so maybe 17 again" or "42 is more common overall." But it computes each token's representation by looking at the raw inputs only once.

**What a 4-layer model can learn**: Higher-order patterns. Layer 1 might learn "this user alternates between 42-runs and 17-runs." Layer 2 might learn "the runs are getting shorter: started with three 42s, then two 17s, then two 42s, then one 17." Layer 3 might learn "based on the shortening pattern, the next run should be one 42." Layer 4 combines this with the user's overall preference distribution.

Each layer operates on the *output* of the previous layer, not on the raw input. So layer 2 reasons about layer 1's *conclusions*, not about raw tokens. This is compositional reasoning, the same reason deep networks outperform shallow ones on tasks with hierarchical structure. The generative task has this structure because user behavior is temporally structured (sessions, interest phases, boredom cycles). The discriminative task ("will this user click this item?") doesn't. It's a single flat prediction.

The empirical proof: a Transformer trained autoregressively on pure item ID sequences showed power-law improvement from 98K to 0.8B parameters. The same Transformer as a feature extractor for a discriminative head showed no scaling. Only difference: the training objective.

---

## 4. What Are the Input Tokens?

In Section 3, we described the user's history as a flat sequence of codes:

```
[42, 8, 3, 177, 42, 15, 91, 22, 17, 203, 44, 9]
  ← item 1 →    ← item 2 →    ← item 3 →
```

That's 12 tokens for 3 items. Each item contributes 4 codes (one per Semantic ID level). This is the right picture for understanding the training objective: the model predicts each code given all previous codes, so within a single item it predicts the fine codes conditioned on the coarse ones.

But it's not how the Transformer actually *sees* the data. Processing 4 separate codes per item means 4× the sequence length, which means 16× the attention cost. In practice, we **collapse each item's 4 codes into a single dense vector** before feeding it to the Transformer. So 3 items become 3 item tokens, not 12. (Each engagement also gets an action token, so 3 engagements = 6 tokens total, as we'll see shortly.) The autoregressive prediction of individual codes still happens at the output head. The model decodes level by level. But the input representation is one dense vector per item.

This section explains how that collapse works, and what else goes into each token beyond the item identity.

### How a Semantic ID becomes an embedding

A Semantic ID like `(42, 8, 3, 177)` is four integers. The Transformer needs a dense vector. How do you convert one to the other?

You might think: use the codebook vectors directly. After all, `codebook_0[42] + codebook_1[8] + codebook_2[3] + codebook_3[177]` reconstructs the item's vector. But those code vectors were optimized for quantization quality, not for downstream sequence prediction. The Transformer needs embeddings optimized for *its* task.

Instead, maintain **separate embedding tables for the Transformer**, one per Semantic ID level:

```python
# Separate from the codebooks - these are the Transformer's own learned embeddings
sem_id_embed_tables = {
    level_0: float32[K, d_model],    # 256 embeddings for level-0 codes
    level_1: float32[K, d_model],    # 256 embeddings for level-1 codes
    level_2: float32[K, d_model],    # 256 embeddings for level-2 codes
    level_3: float32[K, d_model],    # 256 embeddings for level-3 codes
}

# To embed item with Semantic ID (42, 8, 3, 177):
item_embedding = (sem_id_embed_tables[level_0][42] 
                + sem_id_embed_tables[level_1][8]
                + sem_id_embed_tables[level_2][3]
                + sem_id_embed_tables[level_3][177])
# float32[d_model] - one vector per item, sum of 4 level embeddings
```

Total embedding parameters: $4 \times 256 \times d_{\text{model}}$. With $d_{\text{model}} = 512$, that's ~500K parameters, versus 500M × 512 ≈ 256 billion parameters for atomic ID embedding tables. The compression is massive.

Why summation works: it mirrors the residual quantization structure (sum of code vectors approximates the item), so the Transformer's embedding space inherits the hierarchical structure. Two items sharing prefix `(42, 8, 3)` share three of four embedding components and differ only in the level-3 embedding. They start close in the Transformer's input space, which is exactly what we want.

An alternative is concatenation (`concat` instead of `+`), which gives a $4 \times d_{\text{model}}$ vector that you project down. This preserves more information (the level-0 component can't be confused with a level-3 component) but costs a projection layer. Summation is simpler and works well in practice.

### The sequence format: item tokens and action tokens

Each engagement has two pieces of information: *what* item the user saw and *what they did* with it (clicked, watched 30 seconds, skipped, purchased). Meta's HSTU (Hierarchical Sequential Transduction Unit) represents each engagement as two separate tokens in the sequence:

```
engagement_i → [Φ_i, a_i]
                 ↑          ↑
                 item        action
                 token       token

Full sequence: [Φ_0, a_0, Φ_1, a_1, ..., Φ_{n-1}, a_{n-1}]
```

Each `Φ_i` is the item embedding (sum of 4 level embeddings from above). Each `a_i` is an action embedding from a small learned table (one embedding per action type: long_watch, click, skip, purchase, etc.).

Note: each `Φ_i` is already one dense vector, not the 4 raw codes. So 3 engagements = 6 tokens (3 item + 3 action), not 12.

**This format is the key design insight.** Because item tokens and action tokens alternate in the sequence, and the model uses causal attention (each token can only see tokens before it), ranking and retrieval are both just **next-token prediction at different positions**:

**Ranking** (predict the action): The model sees `[Φ_0, a_0, ..., Φ_{t-1}, a_{t-1}, Φ_t]` and predicts `a_t`. The item `Φ_t` is the candidate. The model has seen it but hasn't seen what the user does with it yet. It predicts the action from its output at position `Φ_t`:

```python
# Sequence so far: [..., Φ_t]  (candidate item appended, action not yet generated)
# Model predicts: a_t = what will the user do with this item?
output = hstu_stack(sequence)
action_logits = output_head(output[position_of_Φ_t])  # float32[num_action_types]
# P(long_watch)=0.41, P(click)=0.22, P(skip)=0.31, P(purchase)=0.06
```

**Retrieval** (predict the next item): The model sees `[Φ_0, a_0, ..., Φ_{t-1}, a_{t-1}]` (only after positive actions) and predicts `Φ_t`. It generates the next item's Semantic ID autoregressively, one code at a time. This is the generative retrieval from Section 3.

Both tasks use the same model, the same sequence, the same attention. The only difference is *which position* you read the prediction from and *what vocabulary* you predict over (action types for ranking, Semantic ID codes for retrieval). No special `<UNK>` tokens or separate classification heads needed.

**The cost:** Sequence length doubles compared to one-token-per-engagement. Transformer attention is $O(L^2)$, so doubling $L$ quadruples compute. But you get a unified model that handles both ranking and retrieval with one architecture and one training objective.

### Ranking multiple candidates efficiently

To rank 100 candidates for one user, you'd naively run the full sequence 100 times. But the user's history is the same every time. Two optimizations:

**KV caching (standard, same as LLM inference).** Process the history once, cache the keys and values from all history positions. For each candidate, only compute that one new token's attention against the cache. One expensive forward pass for the history, then 100 cheap single-token passes.

**Microbatched parallel scoring (the M-FALCON contribution).** Even with KV caching, scoring 100 candidates one at a time means 100 sequential GPU kernel launches, each computing attention for just one token. Each launch has overhead and underutilizes the GPU.

M-FALCON's trick: append multiple candidates to the sequence simultaneously and run ONE attention computation. The attention mask prevents candidates from seeing each other:

```text
                  cached history          candidates appended together
                  h_0  h_1 ... h_L       c_0   c_1   c_2

candidate_0   [   Y    Y   ...  Y    |    Y     .     .   ]
candidate_1   [   Y    Y   ...  Y    |    .     Y     .   ]
candidate_2   [   Y    Y   ...  Y    |    .     .     Y   ]
```

Each candidate's row computes dot products against all history keys (shared, from cache) and only its own key. This is one matrix multiply, not three. The GPU processes all candidates' attention scores in a single operation because the attention matrix is just bigger, not repeated. Instead of 100 sequential kernel launches (each underutilizing the GPU), you get 10 launches of 10-candidate microbatches, each fully utilizing the GPU's parallel compute.

```python
# Process history once (expensive):
history = [Φ_0, a_0, Φ_1, a_1, ..., a_{t-1}]
cached_KV = hstu_encode(history)

# Score candidates in parallel microbatches:
for microbatch in chunk(candidates, batch_size=10):
    Φ_batch = [embed_semantic_id(c) for c in microbatch]
    # One attention kernel scores all 10 candidates simultaneously
    # Mask ensures candidates attend to history but not to each other
    action_logits_batch = hstu_predict_microbatch(Φ_batch, cached_KV)
```

KV caching alone gives you $O(L)$ per candidate instead of $O(L^2)$. Microbatching on top of that gives you GPU parallelism across candidates. Together, this is why Meta serves a model with 285× more FLOPs than the DLRM it replaced, at higher throughput.

### Where do popularity, freshness, and other dense features go?

In Section 2, we noted that Semantic IDs deliberately don't encode popularity or freshness. The tokens so far have two components: item identity (from Semantic ID embeddings) and action type. Where do dense, real-time item features enter?

**Option 1: Add them to the item token.** Bucket continuous values into discrete ranges (e.g., popularity: 0-1K, 1K-10K, 10K-100K, ...) and learn an embedding per bucket, or pass raw floats through a small MLP. Add this as a third component to the item embedding: `Φ_i = sem_id_emb[i] + dense_emb[i]`. For candidate items during ranking, the dense features reflect the candidate's *current* real-time stats.

**Option 2: Drop them entirely (Meta's approach).** HSTU uses only sparse (categorical) features and drops all dense features. The argument: if a user has engaged with an item's Semantic ID prefix 50 times in their history, the sequence itself implicitly encodes that prefix's popularity. The model learns aggregate statistics from the raw event stream without being told explicitly. Meta reports this actually outperforms traditional feature engineering in their setting.

**Option 3: Structured token types (Tencent's approach).** Instead of mixing everything into one token, define separate token types for different kinds of information. Tencent's GPR system uses User tokens (profile), Organic tokens (content engagements), Environment tokens (real-time context like ad position, placement type, trending status), and Item tokens. The environment token is refreshed in real-time at serving, carrying the latest popularity and context signals separate from the item identity.

There's a genuine philosophical split here. Meta's "drop everything, trust the sequence" is radical but works at their scale. Meituan found the opposite: dropping dense features "significantly degrades model performance, and scaling up cannot compensate for it at all." Nobody has published a clean ablation isolating popularity features specifically in a generative recommender. We'll revisit the broader question of lost features in Section 8.

---

## 5. The Transformer Doesn't Work Out of the Box

### First attempt: vanilla Transformer on rec sequences

Take a standard causal Transformer. Feed it the user's token sequence. Train with next-token prediction. What breaks?

### Problem 1: softmax forces a competition

Standard attention:

```python
# X: float32[L, d]               - L tokens, d-dimensional
Q = einsum('ld, da -> la', X, W_Q)    # float32[L, d_attn]  - each token gets a query
K = einsum('ld, da -> la', X, W_K)    # float32[L, d_attn]  - each token gets a key
V = einsum('ld, da -> la', X, W_V)    # float32[L, d_attn]  - each token gets a value

# Every query dot-products with every key: L queries × L keys = L×L scores
scores = einsum('ia, ja -> ij', Q, K)  # float32[L, L]  - score[i,j] = how much token i attends to j

weights = softmax(scores, dim=-1)      # float32[L, L]  - each ROW sums to 1
# weights[i, :] is a probability distribution over all positions, for query i

# Each token's output is a weighted sum of all value vectors, using its row of weights
output = einsum('ij, ja -> ia', weights, V)  # float32[L, d_attn]
# output[i] = sum_j weights[i,j] * V[j]  - token i's new representation
```

The `softmax(dim=-1)` forces each row of the weight matrix to sum to 1. This is a zero-sum game: for token $i$ to attend strongly to token $j$, it must attend less to everything else.

**Why this is wrong for recommendations**: A user's history might have multiple independently relevant items. If you're predicting "what will this user watch next after pasta videos and skateboarding," both interests are relevant simultaneously. Softmax forces the model to divide attention between them. Worse, softmax can never output true zero. Every position gets a small weight, and across 1000 irrelevant tokens, those small weights add up to noise.

**The fix**: Replace softmax with a pointwise nonlinearity (SiLU) applied to each score independently:

```python
scores = einsum('ia, ja -> ij', Q, K)  # float32[L, L] - same dot products
weights = silu(scores)                  # float32[L, L] - each entry INDEPENDENT
# weights[i,j] = silu(Q[i] · K[j])    - no normalization across j!
output = einsum('ij, ja -> ia', weights, V)  # float32[L, d_attn]
```

Now each attention weight is computed independently. The model can attend strongly to multiple positions (no competition), and truly ignore irrelevant positions (SiLU of a negative score ≈ 0, unlike softmax's always-positive floor).

### Problem 2: position doesn't capture time

We try the pointwise attention with standard positional encodings. Better, but something is still wrong.

Consider: a user watches 10 cooking videos in one hour (positions 50–60), goes offline for a month, then watches a skateboarding video (position 61). Positionally, 60 and 61 are adjacent. Temporally, they're a month apart. Standard positional encodings treat position 60→61 the same as 59→60. The model can't tell a session boundary from a within-session transition.

**The fix**: Don't add positional information to the token embeddings (that pollutes the content representation). Instead, add a **relative attention bias** directly to the attention scores. HSTU uses the sum of two biases, one for positional distance and one for temporal distance:

```python
scores = einsum('ia, ja -> ij', Q, K)                   # float32[L, L]
bias = rab_position[i,j] + rab_time[i,j]                # float32[L, L]
weights = silu(scores + bias)                             # float32[L, L]
output = einsum('ij, ja -> ia', weights, V)              # float32[L, d_attn]
```

Both biases use **log-scale bucketing** to keep the number of learnable parameters small. For temporal distance: compute the time gap between events $i$ and $j$ in seconds, then bucketize with a log function:

```python
bucket = floor(log(max(1, |t_j - t_i|)) / 0.301)
# 1 second → bucket 0
# 2 seconds → bucket 1  
# 10 seconds → bucket 3
# 1 minute → bucket 6
# 1 hour → bucket 12
# 1 month → bucket 21
```

Each bucket gets one learned weight. So there are only ~25 learnable parameters for temporal bias, not $L^2$. The log scale means the model has fine resolution for recent events (distinguishing "2 seconds ago" from "10 seconds ago") and coarse resolution for old events (lumping "3 weeks ago" and "4 weeks ago" together). Same bucketing scheme for positional distances.

Now the model can learn: "events close in wall-clock time are related even if far apart in the sequence" and "a month-long gap means a context switch regardless of position." The log-bucketing prevents overfitting despite the parameters being learned.

### Problem 3: different actions carry different signals

We now have pointwise attention with relative position and time biases. The model trains well, but ranking quality plateaus. Diagnosis: the model treats all engagement types the same. A click, a 30-second watch, a purchase, and a share all become attention-weighted sums of the same value vectors. But these carry fundamentally different signals. A purchase is a much stronger preference signal than a 2-second click.

This is actually two problems:

1. **Representation**: The model has no mechanism to represent "this was a purchase" differently from "this was a bounce" in its hidden state. Even if it wanted to treat them differently, it can't.
2. **Incentive**: Even with that mechanism, if the training loss treats all next-token predictions equally, and clicks are 100× more frequent than purchases, the model optimizes for predicting clicks. It won't learn to care about purchases.

We solve (1) here with an architectural fix. Problem (2) is a training objective problem that gets solved later: through loss weighting during pretraining (weight purchase predictions 10× higher than click predictions) and through alignment in Section 7 (DPO explicitly reweights outputs by a reward function that can value purchases, watch time, and satisfaction over raw clicks).

**The architectural fix for representation**: Add a gating mechanism. Project the input to four matrices instead of three:

```python
# Project input to 4 matrices instead of 3
proj = silu(einsum('ld, dh -> lh', X, W_proj))    # float32[L, 4*d_attn]
Q, K, V, U = split(proj, 4, dim=-1)                # each: float32[L, d_attn]
#                      ^-- U is the new gating vector

scores = einsum('ia, ja -> ij', Q, K)               # float32[L, L]
weights = silu(scores + bias)                        # float32[L, L]
attn_out = einsum('ij, ja -> ia', weights, V)        # float32[L, d_attn]

# Element-wise gating: each dimension scaled independently per token
gated = attn_out * U                                 # float32[L, d_attn]
# gated[i, k] = attn_out[i, k] * U[i, k]           - dimension k of token i
#                                                      gets amplified or suppressed

output = einsum('la, ad -> ld', layer_norm(gated), W_out)  # float32[L, d]
```

The gate `U` is a per-dimension volume knob on the attention output. A purchase action token and a skip action token have different input embeddings (different rows in the action embedding table), so they produce different U vectors. The gate gives the model the *capacity* to route information differently based on action type.

But capacity is not incentive. The gate doesn't know purchases are worth more to your business than clicks. It learns whatever the loss function rewards. With unweighted next-token prediction, the model learns which action types are *predictive of future tokens*, not which action types are *valuable to the business*. If clicks are 100× more frequent than purchases, the loss is dominated by click predictions. The gate will get very good at representing click patterns and mediocre at representing purchase patterns, because that's what minimizes the loss.

For the model to actually care about purchases, you need explicit intervention in the training signal. Loss weighting (weight purchase predictions 10× higher than click predictions during pretraining) is the simplest fix. Alignment in Section 7 (DPO with a reward function that values purchases, watch time, and satisfaction over raw clicks) is the more principled fix. The gate provides the representational machinery; those interventions provide the direction.

### The complete attention block

All three fixes combined:

```python
def hstu_attention(X, pos_dist, time_dist):
    """
    X: float32[L, d]          - input sequence, L tokens of dimension d
    pos_dist: int[L, L]       - pairwise position distances
    time_dist: float[L, L]    - pairwise time gaps in seconds
    returns: float32[L, d]    - output sequence, same shape
    """
    # Project to Q, K, V, U - four views of each token
    proj = silu(einsum('ld, dh -> lh', X, W_proj))  # float32[L, 4*d_attn]
    Q, K, V, U = split(proj, 4, dim=-1)              # each: float32[L, d_attn]
    
    # Attention scores: every query against every key
    scores = einsum('ia, ja -> ij', Q, K)             # float32[L, L]
    
    # Relative attention bias: log-bucketed, learned weights for position + time
    bias = rab_pos[bucket(pos_dist)] + rab_time[bucket(time_dist)]  # float32[L, L]
    
    # Pointwise activation: each weight independent (no softmax competition)
    weights = silu(scores + bias)                      # float32[L, L]
    weights = causal_mask(weights)                     # zero out j > i
    
    # Aggregate values using attention weights
    attn_out = einsum('ij, ja -> ia', weights, V)      # float32[L, d_attn]
    # attn_out[i] = sum_j weights[i,j] * V[j]
    
    # Gate: per-dimension volume knob, different for each token
    gated = layer_norm(attn_out * U)                   # float32[L, d_attn]
    # gated[i,k] = norm(attn_out[i,k] * U[i,k])
    
    # Project back to model dimension
    output = einsum('la, ad -> ld', gated, W_out)      # float32[L, d]
    return output
```

Stack multiple blocks for depth. This is the HSTU (Hierarchical Sequential Transduction Unit) block, the core building block for generative recommendation. The name reflects what it does: process hierarchical, sequential user action data through a transduction (sequence-to-sequence) architecture.

### What else changes about the Transformer?

The three fixes above (pointwise activation, relative attention bias with time, gating) modify the attention mechanism. But HSTU also simplifies and optimizes the overall Transformer block in ways worth knowing.

**No separate feed-forward network.** A standard Transformer block alternates two sub-layers: multi-head attention, then a feed-forward network (two linear projections with a nonlinearity between them). HSTU drops the FFN entirely. The gating mechanism partially absorbs its role: element-wise multiplication of `attn_out * U` followed by a linear projection is already a nonlinear transformation of the attention output, similar to what the FFN would do. Fewer parameters, lower latency, and empirically no quality loss in the recommendation setting.

```python
# Standard Transformer block: two sub-layers
x = x + attention(layer_norm(x))      # sub-layer 1: attention
x = x + ffn(layer_norm(x))            # sub-layer 2: feed-forward (2 linear + activation)

# HSTU block: one sub-layer (attention + gating replaces both)
x = x + hstu_attention(layer_norm(x)) # gating inside absorbs FFN's role
```

This means an HSTU block has roughly half the parameters and half the compute of a standard Transformer block at the same hidden dimension. You can stack twice as many layers for the same budget, which matters because Section 3 showed that depth is what unlocks scaling.

**Sparse Mixture of Experts (in encoder-decoder variants).** When you do want more capacity in the feed-forward computation (particularly in the decoder, which needs to choose among millions of possible Semantic ID codes), some systems like OneRec add a sparse MoE layer. Instead of one FFN, you have 64 expert FFNs, and a learned gating function routes each token to its top-2 experts:

```python
# Sparse MoE: each token activates only 2 of 64 experts
gate_logits = einsum('ld, de -> le', token, W_gate)     # float32[L, 64]
top2_experts = topk(gate_logits, k=2, dim=-1)            # int[L, 2]
top2_weights = softmax(gather(gate_logits, top2_experts)) # float32[L, 2]

# Only 2 experts run per token (not all 64)
expert_outputs = [experts[e](token) for e in top2_experts]  # 2 × float32[L, d]
output = sum(top2_weights * expert_outputs)                  # float32[L, d]
```

Total model capacity scales with the number of experts (64× more parameters in the FFN), but compute per token only scales with $k = 2$. This is how you get a large model that's cheap to run. The tradeoff: load balancing across experts is tricky (some experts might get all the traffic while others idle), and the routing adds implementation complexity.

**Fused attention kernels for serving.** The HSTU paper reports 5-15× speedup over FlashAttention2 on 8192-length sequences. The trick: fuse the pointwise activation, relative attention bias, causal mask, and value aggregation into a single GPU kernel. Standard FlashAttention is optimized for softmax attention (it exploits the online softmax algorithm to avoid materializing the full L×L matrix). HSTU's pointwise activation is simpler than softmax (no normalization across the row), which enables a different fusion strategy. The computation becomes memory-bound rather than compute-bound, and scales with GPU register size rather than HBM bandwidth. You don't need to understand the kernel implementation details for an interview, but knowing *why* HSTU is faster than standard Transformers (simpler activation = more fusible operations = better hardware utilization) is useful context.

These optimizations are why Meta deployed an HSTU model with 285× more FLOPs than the DLRM it replaced, using *less* inference compute. The architectural simplifications (no FFN, pointwise instead of softmax) aren't just about model quality. They directly enable the serving efficiency that makes the whole approach practical.

---

## 6. The Sequence Doesn't Fit

### The problem (and why "just use a longer context window" doesn't work)

A power user on a short-video platform generates 100K+ engagements over their lifetime. With two tokens per engagement (item + action), that's 200K+ tokens.

LLMs now handle 200K+ context windows. So why not just feed the full history into the Transformer?

**Scale.** LLMs serve one user at a time, for a few seconds, at maybe thousands of QPS. Recommendation systems serve **billions of requests per day** with **strict latency budgets** (the ranking stage alone typically gets a tens-of-milliseconds P99 budget, with the full pipeline at 100-300ms) and train on **10-100 billion examples per day**. The constraint isn't "can attention physically handle 200K tokens." It's "can you afford $O(L^2)$ attention at that length, multiplied by billions of daily requests, within your GPU budget and latency SLA." At $L = 200{,}000$, one attention layer costs $4 \times 10^{10}$ operations per request. Even with fused kernels, that's not feasible at recommendation scale.

In practice, no production system runs full attention on 100K tokens. Every system compresses. They differ in *how*.

### What systems actually do

**HSTU (Meta): Truncate + stochastic subsample.** HSTU uses sequences of 4096-8192 tokens in production, not 100K. For training, it uses **Stochastic Length (SL)**: randomly subsample the sequence with older events having exponentially lower sampling probability. At their recommended setting ($\alpha = 1.6$), a 4096-token sequence becomes ~776 tokens most of the time, removing 80%+ of tokens. The model still trains well because the stochasticity acts as data augmentation (each training step sees a different subsample of the same user's history).

```python
# Stochastic Length: subsample with decaying probability
for each token at position i in sequence of length L:
    keep_probability = (i / L) ** (alpha - 1)    # alpha=1.6 typical
    # Recent tokens (i ≈ L): keep_prob ≈ 1.0
    # Old tokens (i ≈ 0): keep_prob ≈ 0.0
    # Middle tokens: smoothly interpolated
```

The limitation: old events survive with low probability, not zero. A cooking interest from 6 months ago might be represented by 2-3 surviving events out of hundreds, or might be entirely absent in a given training step. The model gets a noisy, sparse view of long-term history, and that view changes randomly between training steps. This works surprisingly well (the stochasticity acts as regularization), but the model can't reliably learn precise long-term patterns like "this user's cooking interest peaked in March and declined in April."

**OneRec (Kuaishou): Multi-pathway compression.** The insight behind this approach: different time horizons of a user's history answer different questions, and they need different levels of detail.

Think about what you'd want to know about a user to recommend their next video:

- **Who are they?** Age, gender, location. One token is enough. This never changes.
- **What are they doing right now?** Their last 20 interactions, in full detail with action types. You need the exact sequence because order matters (they just skipped a cooking video, so maybe not another one right now). 20 tokens.
- **What do they like in general?** Their top 256 most-engaged items. You don't need the exact order or timestamps, just the set of items they've shown strong positive signal for. 256 tokens.
- **What's their lifetime taste profile?** Their full 100K interaction history. You can't afford 100K tokens, but you also can't just throw this away. Compress it heavily into a summary. 32 tokens.

Each pathway preserves exactly the level of detail that matters for its time horizon. Recent history keeps full detail. Lifetime history gets heavy compression. Static features get one token.

```python
# Pathway 1: Who are they? → 1 token
h_user = project(concat(e_uid, e_gender, e_age, ...))    # float32[1, d_model]

# Pathway 2: What are they doing right now? → 20 tokens (full detail, order preserved)
h_short = embed_and_project(last_20_items)                # float32[20, d_model]

# Pathway 3: What do they like? → 256 tokens (set of strong positives)
h_positive = embed_and_project(top_256_items)             # float32[256, d_model]

# Pathway 4: Lifetime taste profile → 32 tokens (100K items compressed)
h_lifetime = compress_lifetime(all_100K_items)             # float32[32, d_model]

# All pathways concatenated into one sequence for the encoder
H_enc = concat([h_user, h_short, h_positive, h_lifetime])  # float32[309, d_model]
```

309 tokens. Down from 200K. Manageable.

**How the lifetime compression works (Pathway 4).** This is the non-obvious part. You have 100K item embeddings and need to squeeze them into 32 tokens. Two stages:

**Stage 1: Cluster.** Run hierarchical k-means on the 100K item embeddings to get ~200 cluster centroids. Each centroid represents a neighborhood of similar items. "Cooking content" might be one cluster, "skateboarding" another, "electronics reviews" another. 100K items → 200 centroids.

**Stage 2: Summarize the clusters with learned queries.** This uses a QFormer (Querying Transformer). Initialize 32 learnable "query" vectors. These queries learn to ask questions like "how much cooking content is in this user's history?" or "what's the strongest interest cluster?" Each query attends to all 200 centroids and produces a weighted summary:

```python
def qformer(query_tokens, cluster_centroids):
    """
    query_tokens:      float32[32, d]    - 32 learnable questions about the user's history
    cluster_centroids: float32[200, d]   - 200 cluster centers from k-means on 100K items
    returns:           float32[32, d]    - 32 summary tokens encoding the full lifetime
    """
    Q = einsum('qd, da -> qa', query_tokens, W_Q)      # 32 queries
    K = einsum('ld, da -> la', cluster_centroids, W_K)  # 200 keys
    V = einsum('ld, da -> la', cluster_centroids, W_V)  # 200 values
    
    # Each query attends to all 200 cluster centers
    scores = einsum('qa, la -> ql', Q, K)               # float32[32, 200]
    weights = softmax(scores, dim=-1)                    # each query's weights sum to 1
    
    # Each query's output is a weighted mix of all cluster values
    output = einsum('ql, la -> qa', weights, V)          # float32[32, d]
    return output
```

The 32 queries are learned during training. They start random and converge to useful questions about the user's history. The model discovers what aspects of lifetime behavior are worth preserving in 32 tokens.

**Why concatenation matters.** After concatenation, the 309 tokens go through the same HSTU attention blocks from Section 5 (but bidirectional here, no causal mask, since this is the encoder processing the user's past). This means: short-term tokens can attend to lifetime tokens, positive-feedback tokens can attend to short-term tokens, and so on. The encoder discovers relationships *across* temporal scales: "this user's recent skateboarding binge (short-term) is a departure from their lifelong cooking preference (lifetime). Maybe a temporary phase." That cross-pathway attention is the whole point of concatenating rather than processing each pathway independently.

### Other approaches to long-sequence compression

**VISTA (Meta, 2025): Linear attention summarization.** Compress 100K history into a few hundred summary tokens using linear-complexity attention (avoiding the $O(L^2)$ cost), then run standard target attention from candidates against those summaries.

**ULTRA-HSTU (Meta, 2026): Deeper computation where it matters.** Three mechanisms combined. First, semi-local attention (SLA) restricts each token to a local window rather than the full sequence. This is the same idea as sliding window attention in LLMs (Longformer, Mistral). Second, and more interesting: **attention truncation**. Run the first few HSTU layers on the full long sequence, then run the remaining (deeper) layers on only the most recent segment. The insight: old history needs shallow processing to extract general taste, but recent history needs deep processing to capture current intent. You allocate more compute where it's more predictive, rather than giving every token the same depth. Third, Mixture of Transducers (MoT): process different behavioral signals (e.g., clicks vs purchases vs searches) as separate sequences with separate transducers, then fuse.

---

## 7. The Model Understands Users but Recommends Badly

### The problem

The model is trained. It predicts next tokens well. But when we serve it, the recommendations are... fine for engagement but bad for the business. It recommends clickbait (high predicted CTR, low user satisfaction). It shows 10 pasta videos in a row (accurate prediction, terrible experience). It underweights new items (Semantic IDs help with cold start, as Section 2 showed, but the model still assigns lower probability to specific code combinations it hasn't seen frequently in training data).

The generative training objective is "predict what the user will engage with next." That's not the same as "show the user what they should see." These objectives conflict.

In the old DLRM world, you'd handle this with multi-task towers, one per objective, combined with manually tuned weights. But our generative model outputs Semantic IDs. There are no towers.

### The fix: alignment (same idea as RLHF for LLMs)

**Step 1: Define what "good" means.** Combine multiple signals into a reward:

- **Preference score**: Weighted mix of engagement metrics (watch time, likes, shares, not just clicks).
- **Format reward**: Did the model generate valid Semantic IDs? (Binary. Prevents degenerate outputs.)
- **Business reward**: Diversity (not all same category), safety, monetization targets, cold-start item boost.

**Step 2: Generate contrastive pairs.** Use beam search with the current model to produce multiple candidate item lists per user. Score each list with the reward. Take the best as "chosen", the worst as "rejected".

**Step 3: Update with DPO.** Recall that our model generates Semantic IDs autoregressively, one code at a time. So the probability of a full recommendation list is just the product of all individual code probabilities:

$$P_{\text{model}}(\texttt{list} \mid \texttt{user}) = \prod_{t} P_{\text{model}}(\texttt{code}_t \mid \texttt{all previous codes}, \texttt{user history})$$

This is exactly what the model already computes at every decoding step. The DPO loss uses these probabilities:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big(\beta \log \frac{P_{\text{model}}(\texttt{chosen} \mid \texttt{user})}{P_{\text{pretrained}}(\texttt{chosen} \mid \texttt{user})} - \beta \log \frac{P_{\text{model}}(\texttt{rejected} \mid \texttt{user})}{P_{\text{pretrained}}(\texttt{rejected} \mid \texttt{user})}\Big)$$

Each log-ratio measures: "how much more likely does the current model make this list compared to the pretrained model?" The loss says: make that ratio *larger* for the chosen list and *smaller* for the rejected list.

Concretely: if the chosen list is `[(42,8,3,177), (17,203,44,9), (55,12,8,33)]` (diverse) and the rejected list is `[(42,8,3,177), (42,8,3,52), (42,8,7,88)]` (all pasta), the model learns to increase the probability of generating the diverse list and decrease the probability of the monotone one.

$\beta$ controls how far the model can drift from pretrained behavior. Too small → barely moves. Too large → forgets everything it learned in pretraining.

**Step 4: Prevent forgetting.** Train with combined loss:

$$\mathcal{L} = \mathcal{L}_{\text{next-token}} + \lambda\ \mathcal{L}_{\text{DPO}}$$

The next-token loss on ground truth keeps the model's language modeling ability. DPO steers it toward business-preferred outputs.

### What goes wrong: early instability

In the first alignment round, the model is far from optimal. The log-ratios can be huge. A concrete example: suppose the model assigns probability 0.85 to the chosen list, but the pretrained model assigned 0.001. The log-ratio is $\log(0.85 / 0.001) = 6.7$. Multiply by $\beta = 0.1$ and you get 0.67, giving a moderate gradient. But if the model assigns 0.99 and pretrained assigned 0.0001, the log-ratio is 9.2. The gradient scales linearly with this, and early in training these extreme ratios appear constantly because the model is moving fast. The updates become erratic.

**The fix (ECPO)**: Clip the log-ratios, same idea as PPO's clipped objective:

$$\text{clip}\!\left(\log \frac{P_{\text{model}}(\texttt{list} \mid \texttt{user})}{P_{\text{pretrained}}(\texttt{list} \mid \texttt{user})},\ -\epsilon,\ \epsilon\right)$$

With $\epsilon = 0.2$, that log-ratio of 9.2 gets capped at 0.2. The gradient is gentle and controlled. As training progresses and the model stabilizes, the actual log-ratios naturally stay within the clip range, so it stops activating.

### The iterative loop

This isn't a one-shot process. After updating the model with DPO:

1. The model is now better at generating diverse, business-aligned lists.
2. Use beam search with the *updated* model to generate new candidate lists.
3. These new candidates are higher quality than before. The "chosen" lists are better.
4. Score them, select new chosen/rejected pairs, update again.

Each round raises the ceiling: the model improves, so beam search explores a better region of list space, so the preference pairs are more informative, so the next update is more useful. This converges because the improvement per round shrinks. At some point the model is good enough that beam search can't find much better candidates than what it already generates. Typically 3–5 rounds.

---

## 8. What Did We Lose by Going Generative?

### The broader problem

The model works. Scaling laws are real. Alignment improves business metrics. But there are gaps. The old DLRM system was fed hundreds of features per item and per user-item pair. Our generative model has a sequence of tokens. Where did all those features go?

In Section 4, we showed how item-side dense features (popularity, freshness, creator stats) can optionally be injected as a third token component, and noted the philosophical split: Meta drops them entirely while Meituan insists they're essential. If you chose to include them via `token = sem_id_emb + action_emb + dense_emb`, item-side features are covered.

But there's a harder category of lost signal: **user-item cross-features**. These can't be handled by adding a component to every token, because they depend on the *specific pair* of user and candidate, not just the item alone.

### The cross-feature gap

In DLRM, the most predictive features were often cross-features: "user $u$'s CTR on category $c$ = 18%", "user $u$ viewed item $i$ 7 times in 30 days", "user $u$'s average session length on cooking content = 12 minutes." These directly encode the user-item *relationship*. They're handed to the model as pre-computed dense numbers.

In our generative model, the user is a sequence of Semantic IDs. The model must *implicitly discover* that the user has engaged with cooking content 7 times recently by recognizing code-prefix patterns across thousands of tokens. That's asking attention to do counting, a much harder learning problem than reading a feature that says `count = 7`.

### First attempt: just let the Transformer learn it (Meta's HSTU approach)

This is what Meta does. HSTU drops all dense and cross-features entirely and trusts the sequence. Maybe with enough data and depth, the model will learn to count. After all, LLMs can do arithmetic.

Tested: for users with 50+ interactions with a specific category, the DLRM with pre-computed cross-features achieves measurably higher AUC than the generative model. The Transformer learns rough aggregate statistics ("this user watches a lot of cooking") but not precise counts or rates ("this user's CTR on cooking is 18% vs 12% on sports"). The gap is largest for heavy users with rich per-category history. Exactly the users where personalization matters most.

### Second attempt: shove cross-features back in (Meituan's MTGR approach)

Meituan's MTGR (Meituan Generative Recommendation) found that dropping cross-features "significantly degrades model performance, and scaling up cannot compensate for it at all." Their solution: reorganize training data from one-sequence-per-user to one-sequence-per-(user, candidate):

```python
# Standard generative: one sequence per user
sequence = [user_features, history_tokens]  # predict next item

# User-item level: one sequence per (user, candidate) pair
sequence = [user_features, history_tokens, real_time_interactions, 
            cross_features_for_candidate_k, item_features_for_candidate_k]
```

Cross-features like "user-item CTR" are now just additional features on the candidate token. The model has direct access.

But this requires careful **masking** to prevent leakage. The masking uses the same HSTU attention blocks from Section 5, but with a heterogeneous mask instead of a simple causal mask. Since HSTU uses pointwise activation (SiLU) rather than softmax, masking still works the same way. Zeroed-out positions contribute nothing to the output:

```python
# Attention mask structure:
#              user   history  realtime  cand_1  cand_2  cand_3
# user         [1      1        1         1       1       1    ]
# history      [1      causal   1         0       0       0    ]
# realtime     [1      1        causal    0       0       0    ]
# cand_1       [1      1        1         self    0       0    ]
# cand_2       [1      1        1         0       self    0    ]
# cand_3       [1      1        1         0       0       self ]
```

- User features: visible to everything.
- History: causal within itself. Can see user features but not candidates.
- Candidates: **self-only**. Each candidate sees the user, history, and its own features, but not other candidates. If candidate $k$ could see candidate $k+1$'s features (including labels), it would cheat.

**The cost**: If user $u$ has $N$ candidates, you now produce $N$ sequences instead of 1. That's $N\times$ more training data. In practice, $N$ can be hundreds. This works but it's expensive.

### Third attempt: don't replace DLRM at all (Alibaba's GPSD, Netflix FM, Pinterest PinFM)

This is the hybrid approach, and honestly it's what most companies do. Alibaba's GPSD and LUM, Netflix's Foundation Model, and Pinterest's PinFM all follow this pattern. The insight: you don't have to choose between generative and discriminative. Use generative training to learn better representations, then plug them into your existing DLRM that already handles cross-features natively.

**Wait, doesn't Section 3 say "a Transformer as a feature extractor for a discriminative head showed no scaling"?** Yes, but that was training the Transformer *with* the discriminative objective. The Transformer never learned through a hard generative task, so it never developed rich representations. Making it bigger didn't help because the binary task was too easy. Here, we train the Transformer *with* the generative objective first (where scaling laws do emerge), freeze the resulting representations, and hand them to DLRM. The scaling already happened during pretraining. The DLRM doesn't need to scale; it just consumes the already-good embeddings and adds cross-features on top.

**Step 1: Pretrain**: Autoregressive Transformer on user sequences. Standard next-token prediction. This learns item embeddings that encode temporal patterns and item-item relationships, much richer than DLRM's co-occurrence embeddings. This is where scaling laws apply.

**Step 2: Transfer**: Move the pretrained item embeddings into the DLRM. **Freeze them.** Let the DLRM fine-tune everything else (dense weights, cross-feature processing, task towers) while the pretrained embeddings stay fixed.

```python
# DLRM with transferred embeddings
item_emb = pretrained_transformer.embedding_table    # FROZEN
user_emb = pretrained_transformer.user_encoder(history)  # optionally frozen

# Cross-features still work as before
cross_features = compute_cross_features(user, item)  # user-item CTR, counts, etc.

# DLRM forward pass
x = concat(item_emb, user_emb, cross_features, context_features)
logit = mlp(feature_cross(x))
p_click = sigmoid(logit)
```

**Why freezing is critical**: The whole value of generative pretraining is robust, generalizable embeddings learned from a hard task. If you unfreeze them during discriminative fine-tuning, they start overfitting to the binary labels. The one-epoch curse returns. Tested: unfrozen embeddings degrade after epoch 1. Frozen embeddings allow training for 5+ epochs with continued improvement.

The hybrid approach gives you: scaling laws from generative pretraining + cross-features from DLRM + existing infrastructure and team structure. The tradeoff: two training stages (pretrain + fine-tune) and you don't get the architectural simplicity of a single end-to-end model.

### A middle ground: fine-tune the whole pretrained model (Pinterest's PinFM)

Instead of just transferring frozen embeddings, append each candidate to the user sequence, pass through the pretrained Transformer, and fine-tune the whole model end-to-end on ranking objectives. This is what Pinterest's PinFM does: the candidate is appended to the user sequence to bring "candidate awareness," and the model is fine-tuned (not frozen) along with action predictions. Alibaba's GPSD also tested this as their "Full Transfer" strategy, though they found that freezing the sparse (embedding) parameters and only fine-tuning the dense (Transformer) parameters worked better.

This captures richer contextual information (the Transformer's sequential understanding, not just static embeddings) but risks degrading the pretrained representations. To mitigate: use a smaller learning rate for the pretrained parameters than for the DLRM parameters.

**Serving optimization**: For $C$ candidates per user, the user's sequence representation is identical for all $C$. Compute it once, cache it. For each candidate, compute only the cross-attention between that candidate and the cached representation:

```python
# Compute once per user (expensive, but only once):
user_repr = transformer_encode(user_history)          # float32[L, d]
K_user = einsum('ld, da -> la', user_repr, W_K)       # float32[L, d_attn]  - cached
V_user = einsum('ld, da -> la', user_repr, W_V)       # float32[L, d_attn]  - cached

# Per candidate (cheap - just one query against cached keys/values):
Q_cand = einsum('d, da -> a', candidate_emb, W_Q)     # float32[d_attn] - single query
scores = einsum('a, la -> l', Q_cand, K_user)          # float32[L] - attend to user history
weights = softmax(scores)                               # float32[L]
context = einsum('l, la -> a', weights, V_user)         # float32[d_attn]
logit = dlrm_head(context, cross_features)
```

Cost drops from $O(C \cdot L^2)$ to $O(L^2 + C \cdot L)$.

---

## 9. Generating Lists Instead of Scoring Candidates

### The problem with candidate-at-a-time scoring

Everything so far scores candidates independently: process user history, append candidate item token, predict action. But this means the model doesn't know what else it's recommending. It might put 5 pasta videos in the top 5 because each one independently scores high. Diversity must be enforced post-hoc by a reranking layer.

What if the model could generate a whole recommendation list at once, where each item depends on the ones before it?

### The core insight: autoregressive decoding already gives you interdependence

Think about how the model generates Semantic IDs in retrieval mode (Section 3). It predicts one code at a time, each conditioned on all previous codes. If you generate multiple items sequentially, each item is conditioned on all previously generated items. When generating recommendation 3, the model has already generated recommendations 1 and 2 in its context.

This means diversity emerges naturally from autoregressive list generation. If recommendations 1 and 2 were both pasta videos (codes starting with 42, 8), the model's context encodes "I've already output two pasta items." The next code prediction shifts probability away from prefix (42, 8) toward other categories. This isn't a hard constraint. It's a learned behavior: during training, the model sees ground truth recommendation sessions where diversity correlates with engagement.

This works with a decoder-only model. No encoder-decoder architecture needed.

### Decoder-only list generation (the simpler approach)

OneRec v2 (Kuaishou, 2025) proved this by dropping the encoder entirely from the original OneRec encoder-decoder architecture and going decoder-only. The result: computation cut significantly, model scaled larger, quality maintained.

The sequence is just the user's history followed by the generated recommendations, all processed causally:

```
[Φ_0, a_0, Φ_1, a_1, ..., a_{t-1}, | rec_1_code_0, rec_1_code_1, ..., rec_2_code_0, ...]
         ← user history →             ← generated list (autoregressive) →
```

The causal mask means each generated code can see: all the user's history (to the left) and all previously generated codes (to the left). It cannot see future codes (to the right). Standard autoregressive generation, same as an LLM generating text.

**Constrained decoding for pre-retrieved candidates.** If you've already retrieved 100 candidates and want to generate a diverse *ordering* of a subset, restrict the trie to contain only the Semantic IDs of those 100 candidates. The model autoregressively generates codes, but at each step, only codes that lead to a valid candidate are allowed. The result: an interdependent, diversity-aware ranking of your pre-retrieved set, with no encoder-decoder overhead.

```python
# Build a restricted trie containing only pre-retrieved candidates
candidate_trie = build_trie([sem_id(c) for c in retrieved_candidates])

# Autoregressively generate a list, constrained to candidates
generated_list = []
for slot in range(num_recommendations):
    codes = []
    for level in range(num_levels):
        logits = model.predict_next_code(history + generated_so_far + codes)
        logits = candidate_trie.mask_invalid(logits, prefix=codes)  # zero out invalid
        code = sample(logits)
        codes.append(code)
    generated_list.append(tuple(codes))
```

### Encoder-decoder list generation (the heavier approach)

OneRec v1 (Kuaishou, 2025) used an encoder-decoder architecture. The encoder processes the user's history with **bidirectional** self-attention (no causal mask, every past event sees every other past event). The decoder generates recommendations autoregressively, attending to the encoder's output via cross-attention.

**What the encoder buys you**: bidirectional attention over history produces a richer user representation than causal attention. In causal attention, position 50 can only see positions 0-49. In bidirectional attention, position 50 can also see positions 51-1000. This means the encoder can represent "this user's cooking interest peaked after their skateboarding phase" which requires seeing both phases. The decoder-only model can only represent "this user has done cooking and skateboarding so far."

**The leakage question**: Bidirectional attention over history is not leakage. The encoder sees only the past. The decoder generates the future. Cross-attention bridges past → future. The encoder never sees what's being recommended. The attention mask makes this explicit:

```text
                  ENCODER (past)          DECODER (recs being generated)
                  p0   p1   ...  pL      r1_c0  r1_c1  r2_c0  r2_c1 ...

ENCODER  p0    [  Y    Y    Y    Y    |   .      .      .      .     ]
         p1    [  Y    Y    Y    Y    |   .      .      .      .     ]
         ...   [  Y    Y    Y    Y    |   .      .      .      .     ]
         pL    [  Y    Y    Y    Y    |   .      .      .      .     ]
         ------+----------------------+-------------------------------
DECODER  r1_c0 [  Y    Y    Y    Y    |   Y      .      .      .     ]
         r1_c1 [  Y    Y    Y    Y    |   Y      Y      .      .     ]
         r2_c0 [  Y    Y    Y    Y    |   Y      Y      Y      .     ]
         r2_c1 [  Y    Y    Y    Y    |   Y      Y      Y      Y     ]

Y = can attend     . = masked out (zero)
```

- **Top-left** (encoder↔encoder): All Y. Bidirectional. Every past event sees every other past event.
- **Top-right** (encoder→decoder): All `.` The past NEVER sees the future recommendations.
- **Bottom-left** (decoder→encoder): All Y. Cross-attention. Every decode step reads the full history.
- **Bottom-right** (decoder↔decoder): Causal triangle. Each code sees only previously generated codes.

**The tradeoff**: Cross-attention at every decoder layer adds parameters and compute. OneRec v2 showed that dropping this and going decoder-only is worth it: you lose bidirectional history encoding but gain the ability to scale the model larger within the same compute budget. For most teams, decoder-only list generation is the right starting point.

---

## 10. End-to-End: A Concrete User Through the Full Pipeline

Let's trace one user through every component to see how they connect.

### Offline: building the Semantic ID vocabulary

Before any user is served, we've already:

1. Encoded every video's content (images, title, tags) through a vision-language model → `content_emb: float32[d_content]` per item.
2. Fine-tuned these embeddings with collaborative contrastive loss so that items engaged by similar users are nearby.
3. Built codebooks with 4 levels of 256 codes (via RQ-Kmeans or RQ-VAE). Every video now has a Semantic ID.

```
"Cacio e pepe tutorial" → content_emb → collaborative alignment → quantization → (42, 8, 3, 177)
"Carbonara recipe"      → content_emb → collaborative alignment → quantization → (42, 8, 3, 52)
"Skateboard trick tips" → content_emb → collaborative alignment → quantization → (17, 203, 44, 9)
"Kitchen gadget review" → content_emb → collaborative alignment → quantization → (42, 12, 7, 88)
```

We've also built a trie of all valid Semantic IDs for constrained decoding.

### Online: user arrives

User 7291 opens the app. Their history (last 5 engagements, simplified):

```
1. Watched "Cacio e pepe tutorial" for 8 minutes (long watch)
2. Clicked "Carbonara recipe" then bounced (2-second click)
3. Watched "Skateboard trick tips" for 12 minutes (long watch)
4. Skipped "Kitchen gadget review" (skip)
5. Watched "Cacio e pepe tutorial" again for 6 minutes (rewatch)
```

### Step 1: Build input tokens (Section 4)

First, embed each item's Semantic ID using the per-level embedding tables:

```python
# Cacio e pepe: Semantic ID (42, 8, 3, 177)
Φ_1 = sem_id_table[0][42] + sem_id_table[1][8] + sem_id_table[2][3] + sem_id_table[3][177]

# Carbonara: Semantic ID (42, 8, 3, 52) - shares first 3 codes with cacio e pepe!
Φ_2 = sem_id_table[0][42] + sem_id_table[1][8] + sem_id_table[2][3] + sem_id_table[3][52]
# Φ_1 and Φ_2 differ by only one level-3 embedding - they start close in input space

# Skateboarding: Semantic ID (17, 203, 44, 9) - completely different prefix
Φ_3 = sem_id_table[0][17] + sem_id_table[1][203] + sem_id_table[2][44] + sem_id_table[3][9]

# Kitchen gadget: Semantic ID (42, 12, 7, 88) - shares level-0 code 42 (food/cooking)
Φ_4 = sem_id_table[0][42] + sem_id_table[1][12] + sem_id_table[2][7] + sem_id_table[3][88]
```

Then build the interleaved sequence of item tokens and action tokens:

```python
# Two tokens per engagement: [item, action, item, action, ...]
X = [Φ_1, action_emb["long_watch"],   # engagement 1
     Φ_2, action_emb["bounce"],        # engagement 2
     Φ_3, action_emb["long_watch"],    # engagement 3
     Φ_4, action_emb["skip"],          # engagement 4
     Φ_1, action_emb["long_watch"]]    # engagement 5 (rewatch)
# float32[10, d_model] - 5 engagements × 2 tokens each
```

In a real system, there would be hundreds or thousands of engagements (so thousands of tokens), plus the multi-pathway compression from Section 6 for lifetime history. We use 5 here for clarity.

### Step 2: Process through HSTU attention (Section 5)

```python
# Layer 1: basic patterns
#   The item token for engagement 5 (cacio e pepe rewatch) attends strongly to 
#   engagement 1's item token (same video) and engagement 1's action token (long watch)
#   Gating amplifies "strong positive" dimensions for long_watch action tokens
#   Gating suppresses the "bounce" and "skip" action tokens

# Layer 2: higher-order patterns  
#   The representation now encodes "user repeatedly returns to pasta content"
#   and "skateboarding was engaged too but in a different session"

# Layer 3: compositional patterns
#   Combines "pasta lover who also likes skateboarding" with 
#   "most recent action was rewatching pasta" → strong pasta momentum

output = hstu_stack(X)  # float32[10, d_model]
```

### Step 3a: Ranking a candidate (the common case, Section 4)

The retrieval system has already selected 100 candidates. For each candidate, we append its item token to the history and predict the action at the next position:

```python
# Candidate: "Pasta aglio e olio recipe" - Semantic ID (42, 8, 3, 201)
Φ_candidate = sem_id_table[0][42] + sem_id_table[1][8] + sem_id_table[2][3] + sem_id_table[3][201]

# Append candidate item token to history
X_with_candidate = X + [Φ_candidate]                    # float32[11, d_model]

# In practice: reuse cached KV from the 10 history tokens, only compute attention for token 11
output = hstu_stack(X_with_candidate)                    # float32[11, d_model]

# Predict action at the candidate's position (standard next-token prediction)
action_logits = output_head(output[-1])                  # float32[num_action_types]
# P(long_watch)=0.41, P(click)=0.22, P(skip)=0.31, P(purchase)=0.06
# This candidate scores well - user has strong pasta affinity
```

Repeat for all 100 candidates. Rank by the action scores (e.g., weighted combination of P(long_watch) and P(purchase)). The KV cache means the history computation is done once, not 100 times.

### Step 3b: Generating recommendations directly (the retrieval case, Section 3)

Alternatively, if we're doing retrieval (not ranking pre-selected candidates), the model generates Semantic IDs autoregressively from the last action position (after a positive action):

```python
logits_level_0 = output_head(output[-1])   # float32[256]
# Highest probability: code 42 (cooking/food category) - makes sense given pasta momentum
# code 42 selected

logits_level_1 = output_head(...)          # float32[256], conditioned on code 42
# Highest: code 8 (Italian subcategory) - user clearly prefers Italian
# code 8 selected

logits_level_2 = output_head(...)          # float32[256], conditioned on (42, 8)
# Highest: code 3 (pasta specifically) - but code 11 (pizza) also has decent probability
# code 3 selected

logits_level_3 = output_head(...)          # float32[256], conditioned on (42, 8, 3)
# Various specific pasta videos. (42, 8, 3, 177) is familiar, but the model 
# has learned that recommending the same video again has lower engagement.
# code 201 selected - a new pasta video the user hasn't seen

# Predicted Semantic ID: (42, 8, 3, 201)
# Constrained decoding via trie: valid ✓ - maps to "Pasta aglio e olio recipe"
```

### Step 4: If alignment has been applied (Section 7)

Without alignment, the model keeps recommending pasta (accurate prediction, bad experience). After DPO alignment, the second recommendation shifts:

```
Rec 1: (42, 8, 3, 201)  - Pasta aglio e olio (pasta momentum)
Rec 2: (17, 203, 12, 55) - New skateboard trick video (diversity learned from DPO)
Rec 3: (42, 12, 7, 33)  - Kitchen gadget for pasta making (cross-interest bridging)
```

The aligned model learned that alternating interests keeps users engaged longer than a pure pasta feed.

---

## 11. Choosing Your Path

By now you can derive, justify, and sketch every component. The remaining question is: what do you actually build?

This isn't a design aesthetics question. It's a constraints question. Your answer depends on three things:

**How much disruption can you absorb?** Full generative (replacing DLRM entirely) requires restructuring your retrieval-ranking pipeline and the teams that own each stage. Hybrid (generative pretrain → DLRM fine-tune) changes the embedding layer and nothing else. Most companies start hybrid.

**How important is personalization for returning users?** If your platform is dominated by power users with rich history, cross-features matter enormously and losing them (as full generative does) is painful. If you're cold-start dominated (many new users, many new items), the generalization benefits of Semantic IDs matter more and cross-features matter less.

**What's your latency budget?** Encoder-decoder with list generation is expensive at serving time. You're running autoregressive decoding for every request. Decoder-only scoring with KV caching is much cheaper. Hybrid with frozen embeddings is cheapest. The serving path is just your existing DLRM with better embeddings.

The field is moving fast. But the design space is finite, and now you can navigate it.

---

## 12. System Design: How It All Serves in Production

An interviewer asking "how would you serve this?" wants to know you understand the physical reality, not just the model math. Here's how a generative recommender actually runs.

### The pipeline hasn't gone away

Despite the "unify everything" pitch, production systems still use a cascade. The stages are the same as classic DLRM pipelines; what changes is *what runs inside each stage*:

```
User opens app
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ RETRIEVAL (tens of ms)                              │
│  Option A: Autoregressive Semantic ID generation    │
│    → beam search, constrained by trie               │
│    → produces ~1000 candidate Semantic IDs           │
│  Option B: ANN search on pretrained embeddings      │
│    → standard two-tower, but with better embeddings │
│  Runs on: GPU (if autoregressive) or CPU (if ANN)   │
└─────────────┬───────────────────────────────────────┘
              │ ~1000 candidates
              ▼
┌─────────────────────────────────────────────────────┐
│ RANKING (tens of ms)                                │
│  HSTU processes user history once (KV cache)        │
│  M-FALCON scores all candidates in microbatches     │
│  Predicts P(action) per candidate                   │
│  Runs on: GPU (attention is the bottleneck)         │
└─────────────┬───────────────────────────────────────┘
              │ ~100 scored candidates
              ▼
┌─────────────────────────────────────────────────────┐
│ RERANKING / POLICY (single-digit ms)                │
│  Diversity enforcement, business rules, ad mixing   │
│  OR: autoregressive list generation (Section 9)     │
│  Runs on: GPU (if autoregressive) or CPU (if rules) │
└─────────────┬───────────────────────────────────────┘
              │ ~10-30 items
              ▼
         User sees feed
```

### Where the KV cache lives

In LLM serving, the KV cache is per-conversation and persists across turns. In recommendation, it's per-request and mostly ephemeral. The user's history KV is computed at the start of the ranking stage and shared across all candidates via M-FALCON. Once the request is scored, the cache is discarded. There's no persistent KV store across requests.

Exception: RelayGR (Meta, 2026) pre-computes the long-term history prefix *during the retrieval stage* and relays it to the ranking stage. This is like LLM's prefill/decode split: the expensive prefix computation happens early, and the ranking stage only computes the candidate-specific suffix. The cache must stay on the same GPU (remote fetch would blow the latency budget), so the system uses instance-affinity routing.

### Training infrastructure

Generative recommenders are trained continuously (streaming), not in static epochs. New engagement data arrives constantly. The model trains on a sliding window of the last N days. Embedding tables (Semantic ID embeddings, action embeddings) are stored in parameter servers on DRAM, not GPU HBM, because they're too large. Optimizer states use rowwise AdamW to fit in DRAM. The attention layers train on GPU with standard data parallelism. Training throughput is measured in billions of examples per day, not tokens per second.

### The hybrid serving path

If you took the hybrid approach (Section 8), serving is simpler. Your existing DLRM inference stack doesn't change. The pretrained embeddings are exported as a lookup table, loaded into the feature store alongside your other features. The DLRM model runs on CPU or GPU as before, just with better embeddings. No autoregressive decoding, no KV cache, no new serving infrastructure.

---

## 13. Interview Questions and How to Answer Them

These are the questions a staff-level interviewer will ask about generative recommendation. For each, the section reference tells you where the deep answer lives.

### Conceptual questions

**Q: Why can't DLRM models scale with compute?**
Two reasons: (1) the task is too easy (binary click prediction gives 1 bit of supervision per item, the model plateaus quickly), and (2) item IDs are atomic (no compositionality, so the model can't generalize from one item to related items). Generative training fixes both: next-token prediction over Semantic IDs is a harder task (8+ bits per item) that rewards depth, and Semantic IDs give items compositional structure. → Sections 1, 3

**Q: What are Semantic IDs and how are they created?**
Start with collaborative alignment (contrastive loss so items engaged by similar users are nearby in embedding space). Then quantize via RQ-Kmeans (pure nested k-means, no encoder/decoder, used in OneRec production) or RQ-VAE (encoder/decoder with codebook, from image generation, used in TIGER). Explain the level-by-level residual assignment. Mention that RQ-Kmeans outperforms RQ-VAE in practice. → Section 2

**Q: Walk me through the HSTU architecture and why each design choice was made.**
Three problems, three fixes: (1) softmax forces competition between positions → replace with pointwise SiLU so each attention weight is independent, (2) positional encoding can't capture time gaps → replace with log-bucketed relative attention bias (position + time, ~25 learned parameters each), (3) model can't represent different action types differently → add U gating matrix (fourth projection) for per-dimension volume control. Then: no FFN (gating absorbs it), fused kernels (simpler activation enables fusion), stochastic length for training. → Section 5

**Q: How does HSTU handle both ranking and retrieval?**
Two-token interleaved format: [item, action, item, action, ...]. Ranking = predict the action token after seeing the candidate item token. Retrieval = predict the next item token after a positive action. Both are next-token prediction at different positions in the same sequence. Same model, same training, different prediction targets. → Section 4

**Q: What's M-FALCON and why does it matter?**
Two parts: (1) KV caching (standard, shared history across candidates), and (2) microbatched parallel scoring (the actual contribution). Pack multiple candidates into one forward pass. Modify the attention mask so each candidate sees the cached history but not other candidates. One GEMM instead of N sequential ones. This is what makes the 285× more FLOPs model servable at higher throughput than the DLRM it replaced. → Section 4

### Design tradeoff questions

**Q: Should I drop dense features like Meta does?**
Meta says yes and reports it works at their scale. Meituan says "dropping dense features significantly degrades performance, and scaling up cannot compensate." Nobody has published a clean ablation reconciling these. The safe answer: it depends on your data density and user behavior diversity. If your users have very long, rich histories, the sequence may implicitly encode what dense features would tell you. If your users are sparse or your signal is noisy, you probably need the dense features. → Sections 4, 8

**Q: How do you handle the 100K+ user history?**
Name the approaches by company: HSTU truncates to 4096-8192 and uses Stochastic Length subsampling. OneRec uses multi-pathway compression (recent history at full resolution, lifetime history through k-means + QFormer). VISTA uses linear attention summarization. ULTRA-HSTU uses semi-local attention. Explain why "just use a longer context window" doesn't work: not a capability constraint, but a throughput/cost constraint at billions of daily requests. → Section 6

**Q: What do you lose by going fully generative?**
Cross-features. DLRM had "user u's CTR on category c = 18%" as a precomputed feature. The generative model must implicitly discover this from code-prefix patterns in the sequence. Three approaches: (1) trust the sequence (Meta), (2) shove cross-features back in with heterogeneous masking (Meituan's MTGR, N× training cost), (3) hybrid: generative pretrain then DLRM fine-tune (Alibaba's GPSD, Netflix FM, Pinterest PinFM). Most companies do (3). → Section 8

**Q: How do you ensure the model doesn't just optimize for clicks?**
The gating mechanism (Section 5) gives the model *capacity* to represent different actions differently, but not the *incentive* to value purchases over clicks. Three levels of intervention: (1) loss weighting during pretraining (weight purchase predictions higher), (2) multi-task heads at serving time, (3) alignment via DPO/ECPO with a reward function that values business metrics over raw engagement. → Sections 5, 7

### System design questions

**Q: How would you serve a generative recommender at scale?**
Draw the cascade pipeline: retrieval (autoregressive or ANN, GPU) → ranking (HSTU + M-FALCON, GPU) → reranking (rules or autoregressive list generation). Explain: ranking stage computes user history KV once, shares across all candidates. M-FALCON microbatches candidates for GPU parallelism. KV cache is per-request, not persistent. Latency budget: tens of ms for ranking stage, 100-300ms full pipeline. The hybrid approach is even simpler: existing DLRM stack with pretrained embeddings, no new serving infrastructure. → Section 12

**Q: How do you handle cold-start items in a generative recommender?**
Semantic IDs solve this partially. A new item with Semantic ID (42, 8, 3, 215) shares prefix (42, 8, 3) with known items, so the model immediately knows it's related to pasta content. But the specific code 215 has never appeared in training sequences, so the model underweights it. Fixes: alignment with cold-start item boost in the reward function, or constrained decoding that biases toward freshness. → Sections 2, 7

**Q: Encoder-decoder or decoder-only for list generation?**
Decoder-only is the simpler default. Autoregressive decoding already gives you interdependent list generation (each item conditioned on previously generated items). OneRec v2 proved this by dropping the encoder. Constrained decoding on pre-retrieved candidates gives you diversity within a fixed candidate set. Encoder-decoder buys you bidirectional history encoding (richer user representation) at the cost of cross-attention overhead at every decoder layer. Most teams should start decoder-only. → Section 9