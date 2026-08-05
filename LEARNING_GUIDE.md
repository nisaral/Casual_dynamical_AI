# Learning Guide — Causal-Dynamical AI

This is the study path for the repo. Read it **alongside** the notebooks (not instead of them).  
Each section answers three questions: **what**, **why it matters for AI**, and **what to verify by running code**.

Suggested cadence: **one major notebook per sitting** (60–120 min). Do the exercises at the end of each notebook before moving on.

---

## The big picture (one paragraph)

Modern LLMs are excellent **observational pattern matchers**. A stronger agent needs:

1. **Dynamics** — a state that evolves with stable memory (ODEs / SSMs)  
2. **Causal structure** — distinguish “what I saw” from “what I caused” (PGMs / do-calculus)  
3. **Imagination + search** — roll out futures and pick actions (world model + MCTS)  
4. **Adaptation** — learn new tasks from few examples (meta-learning)

This repo builds those layers from textbook math up to agent algorithms.

```
Phase 0 Dynamics ──► Phase 1–3 Causality ──► Phase 4 Memory (SSM)
        │                    │                      │
        └──────────► Phase 5 Chaos/Fractals         │
                           │                        │
                           ▼                        ▼
                    Phase 6 Planning (MCTS) ◄── Phase 7 World Models
                                                    │
                                                    ▼
                                             Phase 8 Meta-Learning
```

---

## Phase 0 — Dynamical systems (the geometry of state)

### 0.1 Flows on the Line  
**File:** `Calculus & Dynamics/1D flows/Stability & Potential/Flows_on_the_Line.ipynb`

| Idea | Plain English | AI reading |
|------|----------------|------------|
| $\dot x = f(x)$ | Velocity field on a line | Hidden state drift |
| Fixed point $f(x^*)=0$ | State stops changing | Attractor / memory / steady belief |
| $f'(x^*)<0$ stable | Nudge → returns | Robust representation |
| Potential $V$, $f=-V'$ | Ball rolling downhill | Loss landscape analogy |

**What to internalize:** Stop solving ODEs for closed form. Draw arrows. Stability is about the **slope at zeros**.

**Run & check:** Logistic equation → trajectories go to $x=1$. Tanh bistability → two attractors when weight is large enough (memory emerges).

---

### 0.2 Bifurcations  
**File:** `.../Bifurcations/Bifurcations.ipynb`

When a parameter crosses a threshold, the **number or stability** of fixed points jumps.

| Type | Cartoon | AI intuition |
|------|---------|--------------|
| Saddle-node | Attractors appear/disappear | Memory creation/destruction |
| Transcritical | Two fixed points exchange stability | Decision switch |
| Pitchfork | Symmetric → two choices | Symmetry breaking / multi-stability |

**What to internalize:** Training hyperparameters and weight scales can cause **qualitative** changes, not just smoother loss curves.

---

### 0.3 Linear systems (2D)  
**File:** `.../Linear system/Linear_Systems.ipynb`

$\dot{\mathbf x}=A\mathbf x$. Eigenvalues of $A$ classify nodes, saddles, spirals.

**What to internalize:**  
- Re$(\lambda)<0$ → forgetting / stability  
- Pure imaginary → oscillation (center)  
- Positive real part → blow-up (bad latent dynamics)

This is the health check for **SSM matrices** later.

---

### 0.4 Phase plane (nonlinear 2D)  
**File:** `.../Phase_Plane/Phase_Plane.ipynb`

Local linearization via Jacobian. Separatrices = decision boundaries between basins.

**What to internalize:** Ambiguous prompts live near separatrices. Guardrails = keeping trajectories in safe basins.

---

### 0.5 Limit cycles  
**File:** `.../Limit Cycles/Limit_Cycles.ipynb`

Isolated closed orbits (Van der Pol). Gradient systems cannot oscillate. Poincaré–Bendixson: in 2D, trapped flow without fixed points → cycle.

**What to internalize:** Rhythm and periodic inference; also a **ceiling** — true chaos needs ≥3D continuous time (Lorenz later).

---

### 0.6 Discrete maps & chaos  
**File:** `.../chaos/Discrete_Maps_Hyperbolic.ipynb` + `fractals/fratals.ipynb`

Autoregressive generation **is** $x_{n+1}=f(x_n)$. Logistic map: period-doubling → chaos. Lyapunov exponent = sensitivity to prompt. Hyperbolic space fits trees (beam search / hierarchies).

**What to internalize:** Temperature ~ chaos knob (analogy, not equality). Fractal dimension measures attractor complexity of learned dynamics.

---

## Phase 1–3 — Causal / probabilistic structure

### 1–3A Belief Architecture  
**File:** `Probabilistic_Graphs_and_State/Belief_Architecture.ipynb`

| Concept | Formula | Why agents need it |
|---------|---------|-------------------|
| Bayes | posterior ∝ likelihood × prior | Belief update from sensors |
| BN factorization | $\prod P(x_i\mid\mathrm{Pa}_i)$ | Compact world structure |
| Markov blanket | parents+children+co-parents | Local update region |
| d-separation | graph rules for independence | What evidence blocks/opens paths |
| Collider / explaining away | observe common effect → parents compete | Not what plain Transformers guarantee |
| Variable elimination | sum-product | Exact inference (treewidth!) |
| KL forward vs reverse | $D(P\|Q)$ vs $D(Q\|P)$ | Mode covering vs mode seeking (VAE) |

**What to internalize:** Graphs encode **conditional independence**, not just “correlation arrows.”

---

### 1–3B Inference & interventions (critical)  
**File:** `.../Inference_Algorithms_and_Causality.ipynb`

| Algorithm | Use when |
|-----------|----------|
| Belief Propagation | Tree / chain latents (HMM, Kalman cousin) |
| Loopy BP | Cycles; approximate |
| Metropolis–Hastings | Any $P$; only need unnormalized density |
| Gibbs | Easy full conditionals / Markov blankets |
| **do-calculus** | Planning: force action, don’t condition on it |

**The one equation to never confuse:**

$$
P(Y\mid X=x)\;\neq\;P(Y\mid\mathrm{do}(X=x))
$$

**Run & check (Sprinkler):**  
- Observe sprinkler on → rain probability **drops** (explaining away / confounding path)  
- **do**(sprinkler on) → rain stays near base rate; wet grass rises  

That gap is why “train on text correlations” ≠ “reason about interventions.”

---

## Phase 4 — State-space models (scalable memory)

**Notebook:** `State_Space_Models/SSM_S4_Mamba_HiPPO.ipynb`  
**Study notes (annotated PDF + Markdown):**  
- `State_Space_Models/SSM_Learning_Notes.pdf` — full theory, math derivations, code, production  
- `State_Space_Models/SSM_Learning_Notes.md` — same content, easy to highlight / add your own notes  

Aligned to the SSM lecture (continuous → discretize → FFT → S4/HiPPO → Mamba) plus online theory and production practice (Mamba-2, hybrids, HF usage).

Mental model:

```
input tokens x_t  →  latent memory h_t  →  features y_t
     continuous:  h' = A h + B x ,  y = C h + D x
     discrete:    h ← Ā h + B̄ x
```

| Piece | Role |
|-------|------|
| $A$ eigenvalues | Memory lifetime (half-life) |
| HiPPO $A$ | Optimal polynomial compression of history |
| S4 | Structured LTI SSM; train as convolution |
| Mamba | $B,C,\Delta$ depend on $x_t$ → **select** what to write/forget |

**Run & check:**  
- `max|scan − conv| ≈ 0` (math identity)  
- Selective scan reacts hard to spikes, ignores noise  

**How to study:** Read notes § by § → re-derive boxed equations → run matching notebook cells → annotate the `.md` freely.

**Bridge:** A world model is often “SSM/RNN dynamics + stochastic latents + decoder.”

---

## Phase 6 — Look-ahead (planning)

**File:** `Reasoning_and_Planning/MCTS_Lookahead_Planning.ipynb`

Reflex policy: $a=\pi(s)$ in one shot.  
Deliberation: **search** simulated futures, then act.

MCTS loop: **Select → Expand → Evaluate → Backup**  
UCT balances $Q$ vs exploration; PUCT adds prior $P(a\mid s)$.

**Run & check:** Gridworld with trap — enough simulations find the detour; PUCT needs fewer sims with a goal-directed prior.

**Causal link:** Each simulated action is $\mathrm{do}(a)$ inside the model, not “what usually followed $a$ in the dataset.”

---

## Phase 7 — World models (learn the simulator)

**File:** `World_Models/Dreamer_JEPA_World_Models.ipynb` *(this phase)*

Learn $P(o_{t+1}\mid o_{\le t}, a_{\le t})$ or a latent version, then plan inside it.

| Family | Predicts | Notes |
|--------|----------|-------|
| Pixel/world models (PlaNet/Dreamer) | Latents + decode obs/reward | Imagination RL |
| MuZero | Latent dynamics + policy/value | Search in latent space |
| JEPA | Future **embeddings**, not pixels | Avoid wasteful pixel reconstruction |

---

## Phase 8 — Meta-learning (adapt fast)

**File:** `Meta_Learning/MAML_and_Fast_Adaptation.ipynb` *(this phase)*

Outer loop: learn initialization $\theta$ such that **one/few gradient steps** on a new task yield good performance.  
MAML = “learn to learn” for agents that face nonstationary worlds.

---

## How to study (method that works)

1. **Read the theory cells first** without running code (15–20 min).  
2. **Predict** what the next plot will show.  
3. **Run** the code; compare to your prediction.  
4. **Change one number** (noise, $\Delta$, $c_{\mathrm{uct}}$, temperature $r$) and re-run.  
5. Write 3 bullet “AI implications” in your own words.  
6. Only then open the next notebook.

### Red flags you’re skimming, not learning

- You can’t explain $P(Y\mid do(X))$ vs $P(Y\mid X)$ out loud  
- You can’t say what $\mathrm{Re}(\lambda)<0$ means for memory  
- You treat MCTS as “magic tree search” without Select/Expand/Eval/Backup  

### When you’re ready for Phase 7–8

You can explain this pipeline without notes:

> Encode observations → update belief/latent state → intervene with candidate actions in a model → score imagined trajectories (MCTS/MPC) → act → adapt the model or policy when the world shifts.

---

## Optional textbook pairing

| Repo phase | Book chapters |
|------------|----------------|
| 0 | Strogatz 2–3, 5–7, 9–10 |
| 1–3 | Murphy PML intro Ch. 2–4, 9–12; Pearl *Causality* 1–3 |
| 4 | S4 / HiPPO / Mamba papers (intros + figures) |
| 6 | Browne MCTS survey §1–3; AlphaZero / MuZero blogs |
| 7 | Hafner DreamerV3; LeCun JEPA path paper |
| 8 | Finn MAML; Hospedales meta-learning survey |

If a derivation in a notebook feels thin, paste the textbook section you’re on and we can deepen that cell.
