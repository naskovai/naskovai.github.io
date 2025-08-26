---
title: "From RMSProp to AdamW: The Optimizer Evolution Story"
description: "Tracing the evolution of modern neural network optimizers through the lens of what each was designed to fix: gradient scale heterogeneity, mini-batch noise, and regularization interference."
date: 2025-01-26T10:00:00-00:00
tags: ["optimization", "neural networks", "machine learning", "deep learning", "adam", "rmsprop", "adamw", "gradient descent"]
categories: ["Machine Learning"]
math: true
draft: false
---

Modern neural network optimization is fundamentally about dealing with **heterogeneity**. Different parameters have vastly different sensitivities, gradient magnitudes, and update frequencies. The progression from RMSProp to AdamW represents three major breakthroughs in handling these challenges, each building on the previous while solving distinct pathologies that emerge in large-scale training.

Let's trace this evolution through the lens of what each optimizer was actually designed to fix, using consistent notation throughout.

---

## Setting the Stage: The Challenges of Large Language Model Training

Training LLMs exposes several optimization pathologies that simple SGD struggles with:

- **Gradient scale heterogeneity**: Embedding parameters receive massive, frequent updates while deep attention weights get tiny, sparse gradients
- **Mini-batch noise**: Stochastic sampling creates noisy gradient estimates that cause oscillations  
- **Non-stationary landscapes**: The loss surface changes as the model learns, shifting optimal learning rates over time
- **Regularization interference**: Traditional L2 penalty interacts poorly with adaptive learning rates

Each optimizer evolution tackles a specific subset of these problems.

---

## 1) RMSProp: Solving Gradient Scale Heterogeneity

### The Problem RMSProp Solved

Consider training a transformer where:
- Token embeddings get updated on every forward pass → large, frequent gradients
- Deep layer weights only activate for certain input patterns → small, infrequent gradients  
- Output projection weights see accumulated representations → medium, consistent gradients

**SGD with fixed learning rate η:**
- Set η large → embeddings overshoot and destabilize
- Set η small → deep layers barely update and learn slowly
- No setting works well for all parameter types simultaneously

### RMSProp's Innovation: Per-Parameter Adaptive Scaling

**Core insight**: Scale each parameter's learning rate inversely to its historical gradient variance.

**Mathematical formulation:**

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{[track gradient variance]}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot g_t \quad \text{[scale learning rate per parameter]}$$

**Symbols:**

$\quad g_t = \nabla f(\theta_t)$: current gradients

$\quad v_t$: exponentially weighted moving average of squared gradients

$\quad \beta_2 = 0.999$: variance decay rate (typical value)

$\quad \eta$: base learning rate

$\quad \epsilon = 10^{-8}$: numerical stability constant

### What RMSProp Actually Does

**For high-variance parameters** (like frequently updated embeddings), it prevents overshooting despite large gradients:

$\quad v_t \text{ grows large} \rightarrow \frac{\eta}{\sqrt{v_t + \epsilon}} \text{ becomes small} \rightarrow \text{effective learning rate decreases}$

**For low-variance parameters** (like deep layer weights), it ensures sufficient update signal despite small gradients:  

$\quad v_t \text{ stays small} \rightarrow \frac{\eta}{\sqrt{v_t + \epsilon}} \text{ stays large} \rightarrow \text{effective learning rate increases}$

**Dynamic adaptation**: As gradient patterns change during training, $v_t$ automatically adjusts

![Loss curve comparison showing SGD vs RMSProp training from identical initialization, demonstrating RMSProp's faster and more stable convergence](/images/rmsprop-adamw-evolution/F2_loss_vs_steps_linear.png)

### Why This Matters for LLMs

RMSProp enabled stable training of networks with heterogeneous parameter types for the first time. Without it, you couldn't effectively train large networks with both embeddings and deep layers using a single learning rate schedule.

![Step size analysis comparing SGD vs RMSProp across parameter types](/images/rmsprop-adamw-evolution/F1_optimizer_contours_tuned.png)

---

## 2) Adam: Adding Momentum and Bias Correction

### The Problem Adam Solved

RMSProp was a major breakthrough - it successfully handled gradient scale differences and enabled training of heterogeneous networks. However, training still suffered from additional issues that RMSProp couldn't address:

**Mini-batch noise**: Stochastic gradient estimates from small batches create high-frequency oscillations in parameter updates

**Loss landscape navigation**: Sharp valleys in the loss surface cause parameters to bounce between walls instead of moving toward the minimum

**Initialization bias**: RMSProp's $v_t$ starts at zero and builds up slowly due to the exponential moving average with $\beta_2 = 0.999$. Early in training, $v_t$ severely underestimates the true gradient variance, making the denominator $\sqrt{v_t + \epsilon}$ artificially small and thus the effective learning rate $\frac{\eta}{\sqrt{v_t + \epsilon}}$ artificially large. This can cause unstable overshooting in the first several training steps

### Adam's Innovation: Momentum + Bias Correction

**Core insight**: Combine RMSProp's adaptive scaling with exponential moving average momentum, plus bias correction for proper early-training behavior.

**Mathematical formulation:**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{[momentum accumulation]}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{[variance estimation]}$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \quad \text{[bias-corrected momentum]}$$

$$\hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{[bias-corrected variance]}$$

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad \text{[combined update]}$$

**Additional symbols:**

$\quad m_t$: exponentially weighted moving average of gradients (momentum)

$\quad \beta_1 = 0.9$: momentum decay rate (typical value)

$\quad \hat{m}_t, \hat{v}_t$: bias-corrected estimates

$\quad t$: current time step (for bias correction)

### What Each Component Does

**Momentum ($m_t$)**:
- Accumulates gradient direction over time
- **Noise smoothing**: Random gradient fluctuations cancel out
- **Valley navigation**: Builds up speed in consistent directions, dampens oscillations
- **Acceleration**: Moves faster when gradients consistently point the same way

**Bias correction ($1-\beta^t$ terms)**:
- Compensates for zero initialization of $m_t$ and $v_t$
- **Early training**: Prevents underestimated gradients from causing overshooting  
- **Later training**: Correction terms approach 1, minimal effect

### The Combined Effect

Adam preserves RMSProp's per-parameter adaptive scaling while adding:
- **Smoother trajectories** through momentum smoothing
- **Faster convergence** through momentum acceleration  
- **Proper initialization behavior** through bias correction

**Example: Comparing Optimizers in a Noisy Quadratic Valley**

We want a toy problem that is **simple enough to analyze** but **rich enough to show why optimizers behave differently**.

**1. Parameters and Loss Landscape**

We train a 3-dimensional parameter vector: $p=(p_0,p_1,p_2)$.

The loss is a quadratic bowl:
$L(p) = \frac{1}{2}\Big(w_0 p_0^2 + w_1 p_1^2 + w_2 p_2^2\Big)$

The gradient is:
$g_{\text{true}}(p) = (w_0 p_0,\; w_1 p_1,\; w_2 p_2)$

**2. Curvature Weights = Valley Stiffness**

We fix the curvature weights: $w = (1,\;12,\;3)$.

These control how "stiff" the valley is along each coordinate:
- $w_0=1$: gentle slope
- $w_1=12$: steep, sharp valley  
- $w_2=3$: moderate

👉 With learning rate $\eta=0.16$:
- Effective step sizes are $\eta w = [0.16, 1.92, 0.48]$.
- So:
  - $p_0$: smooth and slow.
  - $p_1$: right near oscillatory boundary (1.92 ≈ 2) ⇒ **zig-zag**.
  - $p_2$: moderate steps.

This ensures that **SGD will bounce** in the steep dimension.

**3. Gradient Noise**

In LLM training, optimizers **never see the exact full gradient**. Noise comes from:
- **Mini-batch sampling** (subset of data)
- **Data variance** (sequences of different lengths, difficulties, loss scales)  
- **System effects** (dropout, mixed precision rounding, async updates)

So the optimizer updates with $g_t = g_{\text{true}}(p_t) + \varepsilon_t$, where $\varepsilon_t$ is gradient noise.

In our toy example we mimic this with Gaussian noise per parameter:
- $p_0$: $\sigma=0.10$ (easy, low noise)
- $p_1$: $\sigma=0.9$ (hard, high variance)
- $p_2$: $\sigma=0.18 \to 0.8$ halfway (regime shift → noisier mid-training)

**4. Update Loop**

At each step $t$:
1. Compute true gradient: $g_{\text{true}} = \nabla L(p_t)$
2. Add noise: $g_t = g_{\text{true}} + \varepsilon_t$
3. Optimizer update: $p_{t+1} = \text{OptimizerUpdate}(p_t, g_t)$

👉 The **noise sequence is identical** for all runs, so differences come only from the optimizer rule.

**5. Results**

**5.1 Geometry: 2D Contour with Trajectories**

![Optimizer trajectories in 2D contour showing how each optimizer moves in the valley](/images/rmsprop-adamw-evolution/F1_optimizer_contours_tuned.png)

Shows how each optimizer moves in the valley:
- SGD zig-zags in steep direction
- RMSProp adapts step sizes but jitters
- Adam is smoothest

**5.2 Loss vs Steps (linear scale)**

![Loss vs steps on linear scale](/images/rmsprop-adamw-evolution/F2_loss_vs_steps_linear.png)

- SGD drops quickly but oscillates
- RMSProp appears fast, jitter less obvious here
- Adam smooth and steady

**5.3 Loss vs Steps (log scale)**

![Loss vs steps on log scale showing residual oscillations](/images/rmsprop-adamw-evolution/F3_loss_vs_steps_log.png)

Residual oscillations exposed, especially after the **regime shift** (dashed line at halfway).

**5.4 Zoomed Tail (last 100 steps)**

![Zoomed view of last 100 training steps](/images/rmsprop-adamw-evolution/F4_loss_vs_steps_tail100.png)

- RMSProp jitter stands out compared to Adam's smooth curve
- Dashed lines = moving averages

**5.5 Per-parameter Loss Contributions**

![Per-parameter loss contributions showing jitter sources](/images/rmsprop-adamw-evolution/F5_loss_parts_tail.png)

- Confirms jitter mainly from noisy $p_1$ and shifted $p_2$
- Adam damps both better

**✅ Takeaways**
- **Curvature weights $w$** define the loss geometry
- We update **parameters $p$** with **identical noisy gradients** across optimizers
- **SGD**: overshoots in steep/noisy direction
- **RMSProp**: rescales dimensions but jitters near minimum
- **Adam**: smoothest, thanks to variance scaling + momentum

This minimal design captures **exactly why adaptive momentum methods dominate in practice**: they handle anisotropy + noise + regime shifts far better than raw SGD.

---

## 3) AdamW: Fixing Weight Decay

### The Problem AdamW Solved

Adam became widely adopted, but researchers noticed a subtle issue with regularization:

**Traditional L2 regularization** adds a penalty term to the loss function:

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

When we take gradients, the regularization term contributes $\lambda\theta$ to the gradient:

$$\nabla \mathcal{L}_{\text{total}}(\theta) = \nabla \mathcal{L}(\theta) + \lambda\theta$$

So the regularized gradients become:

$$g_t^{\text{regularized}} = g_t + \lambda\theta_t$$

This regularization term then gets processed through Adam's adaptive machinery:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)(g_t + \lambda\theta_t) \quad \text{[regularization affects momentum]}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(g_t + \lambda\theta_t)^2 \quad \text{[regularization affects variance]}$$

**The problem**: Weight decay strength becomes **parameter-dependent** - but this is problematic for a different reason than gradient scaling.

**How adding $\lambda\theta$ creates weight decay**: When we add $\lambda\theta_t$ to the gradient and then subtract it (via gradient descent), we get:

$\theta_{t+1} = \theta_t - \eta \cdot \text{[something involving } (g_t + \lambda\theta_t)\text{]}$

The $\lambda\theta_t$ term, when subtracted, pulls parameters toward zero - this is the "decay."

**How Adam makes this parameter-dependent**: In standard Adam with L2 regularization:

$g_t^{\text{regularized}} = g_t + \lambda\theta_t$

This regularized gradient gets processed through Adam's adaptive scaling:

$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

where $\hat{m}_t$ and $\hat{v}_t$ now include the $\lambda\theta_t$ terms. The crucial issue is that the weight decay component $\lambda\theta_t$ gets scaled by the same adaptive factor $\frac{1}{\sqrt{\hat{v}_t} + \epsilon}$ as the gradients.

**Why this creates parameter-dependent regularization**:
- Parameters with small $\hat{v}_t$ → large $\frac{1}{\sqrt{\hat{v}_t} + \epsilon}$ → the $\lambda\theta_t$ component gets amplified → stronger weight decay
- Parameters with large $\hat{v}_t$ → small $\frac{1}{\sqrt{\hat{v}_t} + \epsilon}$ → the $\lambda\theta_t$ component gets diminished → weaker weight decay

**Why adaptive optimization works**: We want to equalize convergence behavior across parameters. Parameters with large gradients get smaller effective learning rates (preventing overshoot), while parameters with small gradients get larger effective learning rates (ensuring sufficient updates). This solves the heterogeneity problem.

**Why adaptive regularization is problematic**: Ideally, regularization should be based solely on the regularization criterion (e.g., parameter magnitude for L2), not influenced by optimization dynamics like gradient history. But when weight decay gets processed through Adam's adaptive machinery, regularization strength becomes tied to gradient history - parameters that happen to have small gradients get more regularization, while parameters with large gradients get less, regardless of their actual values or contribution to overfitting.

### AdamW's Innovation: Decoupled Weight Decay

**Core insight**: Apply weight decay directly to parameters, bypassing Adam's adaptive scaling entirely.

**Mathematical formulation:**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{[only gradients, no regularization]}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{[only gradients, no regularization]}$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \quad \text{[bias-corrected momentum]}$$

$$\hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{[bias-corrected variance]}$$

$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta\lambda\theta_t \quad \text{[separate weight decay]}$

**Key difference**: AdamW applies two separate update components:
1. **Adam update**: $-\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ (handles gradients with adaptive scaling)
2. **Weight decay**: $-\eta\lambda\theta_t$ (direct application of $\eta \times$ regularization gradient)

The weight decay is applied directly to parameters, not processed through the adaptive machinery.

### What This Achieves

**Consistent regularization**: Every parameter gets exactly $\eta\lambda\theta_t$ weight decay applied directly, regardless of its gradient history

**Clean separation**: 
- Adam handles **optimization dynamics** (convergence, stability, adaptation)
- Weight decay handles **regularization** (generalization, overfitting prevention)
- No interaction between the two

**Better hyperparameter tuning**: You can optimize $(\beta_1, \beta_2, \eta)$ for convergence speed and optimize $\lambda$ for generalization independently

### Why This Matters for LLMs

**Parameter diversity**: LLMs have embedding layers, attention weights, feedforward layers, and normalization parameters - all with vastly different gradient patterns. AdamW ensures they all get consistent regularization.

**Scale**: With billions of parameters, inconsistent regularization can cause subtle but significant performance degradation.

**Experimental Example: AdamW vs Adam+L2**

Let's examine a toy model that isolates and visualizes how **coupled** (Adam+L2) vs **decoupled** (AdamW) weight decay behaves across different parameter types.

**Model Architecture**: Softmax regression (linear classifier) - simple enough to analyze, complex enough to show realistic gradient diversity.

**Synthetic Data with Three Feature Groups**: The setup uses features with different statistical properties to mimic the diversity found in real neural networks:

$\quad$ **HighVar**: Dense features with high variance, $N(0, 3^2)$ - mimics main network weights that see consistent, noisy gradients

$\quad$ **LowVar**: Dense features with low variance, $N(0, 0.2^2)$ - mimics LayerNorm parameters and biases that are always active but with small gradients  

$\quad$ **Sparse**: Bursty features (95% zeros, else $N(0, 1^2)$) - mimics rare token embeddings and specialized attention heads

**Training Protocol**: Training runs for 1500 steps with mini-batches, allowing Adam's momentum and variance estimates to stabilize. Both optimizers use the same learning rate and weight decay coefficient.

**What Gets Measured**:

**1. Decay Coefficient (The Multiplier)**

![Decay coefficient comparison by feature group showing AdamW's uniform vs Adam+L2's inconsistent coefficients](/images/rmsprop-adamw-evolution/adamw_vs_l2_coefficient_modern.png)

This figure shows the **effective fraction of current weight** that gets subtracted each step:

$\quad$ **AdamW**: decay_coefficient = $\alpha \cdot \lambda$ (direct subtraction, same for all parameters)

$\quad$ **Adam+L2**: The decay term $\lambda \theta$ gets added to the gradient, flows through Adam's momentum and variance machinery, then gets adaptively scaled

AdamW creates a flat line across all groups - the same coefficient everywhere. Adam+L2 creates wildly inconsistent multipliers, with LowVar parameters getting effective coefficients 100× larger than intended, while HighVar parameters get effective coefficients 10× smaller than intended.

**2. Decay Amount (The Actual Subtraction)**

![Decay amount comparison by feature group showing actual weight reduction applied](/images/rmsprop-adamw-evolution/adamw_vs_l2_amount_modern.png) 

This figure shows the **actual L2 norm of weight reduction** applied each step:

$\quad$ **AdamW**: decay_amount = $\alpha \cdot \lambda \cdot \|\theta\|$ (clean, predictable)

$\quad$ **Adam+L2**: The actual amount depends on how $\lambda \theta$ interacts with accumulated momentum and variance from previous steps

Even though AdamW has uniform coefficients, the actual amounts differ because different feature groups naturally settle at different weight magnitudes. Adam+L2 compounds this with its complex momentum-variance interactions, creating decay amounts that vary by orders of magnitude.

**Why Different Groups Have Different Weight Sizes**: Each parameter group experiences gradient updates (pushing weights up to reduce loss) and weight decay (pulling weights toward zero). After many steps, these forces reach a steady-state balance. Different feature groups settle at different weight magnitudes because they need different sizes to be effective - **LowVar** has always-active but weak signals so small weights suffice, while **Sparse** features rarely fire but need large weights so rare activations still influence predictions.

**Key Insight**: **Adam+L2 (coupled)** makes weight decay vary widely through complex momentum-variance interactions. **AdamW (decoupled)** provides clean separation - Adam handles gradients with adaptive scaling, while weight decay provides uniform shrinkage pressure across all parameters. This demonstrates why keeping weight decay separate from momentum and variance tracking produces more predictable training dynamics.

---

## The Evolution in Context

Each optimizer preserved the benefits of its predecessors while solving a specific pathology:

### SGD → RMSProp
**Preserved**: Basic gradient descent principle  
**Added**: Per-parameter adaptive learning rates  
**Solved**: Gradient scale heterogeneity across parameter types

### RMSProp → Adam
**Preserved**: Per-parameter adaptive learning rates  
**Added**: Momentum smoothing and bias correction  
**Solved**: Mini-batch noise and early training instability

### Adam → AdamW  
**Preserved**: Adaptive learning rates + momentum + bias correction  
**Moved**: Weight decay from gradient processing to direct parameter application  
**Solved**: Inconsistent regularization across parameter types

### The Bigger Picture

This isn't just about mathematical elegance - each evolution addressed real problems that emerged as models scaled up:

- **RMSProp**: Made heterogeneous networks trainable
- **Adam**: Made large-scale stochastic training robust  
- **AdamW**: Made regularization work properly at scale

The key insight is that optimization algorithm development is fundamentally **empirical** and **co-evolutionary** with model architecture development. Each breakthrough emerged from trying to solve concrete training problems, not from abstract mathematical principles.