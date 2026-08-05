# Causal-Dynamical AI

## Why this repo?

I'm fascinated by a specific gap in modern AI. Most current models are incredible "statistical pattern matchers." They predict the next word or pixel with mind-bending accuracy by interpolating across massive datasets. However, as noted by researchers like Yann LeCun (Ex-Meta) and Judea Pearl, they often lack an internal "physics" of the world.

**The Shortcomings I'm exploring:**

- **Causality:** They often mistake correlation for causation (the "Causal Parrot" problem).
- **Planning:** They struggle with "Look-ahead" — reasoning through the consequences of an action before taking it.
- **Extrapolation:** They are brilliant at repeating what they've seen, but often fail when faced with structural changes that weren't in the training data.

This repo is my one unified notebook series for exploring how to fuse **Dynamical Systems** (state evolution), **PGMs** (causal logic), and **Deep Learning** (scalable training) to build World Models.

---

## What is a World Model?

In this repo, we study World Models not as a single architecture, but as a system that allows an agent to simulate reality. We implement and study the **"Big Three"** approaches:

- **Latent World Models (DreamerV3/MuZero):** Models that learn a compressed "hidden state" of the world and "imagine" future trajectories to plan actions.
- **Joint-Embedding Predictive Architectures (V-JEPA):** Meta's approach to learning by predicting missing pieces of a video or image in "concept space" rather than pixel space.
- **State Space Models (Mamba/S4):** Using the math of continuous-time physics (ODEs) to give models infinitely long, efficient memory.

---

## Prerequisites & Background

This repo assumes you have a **high-level understanding of Machine Learning, Deep Learning, and Large Language Models (LLMs).** You don't need to be a math PhD, but you should be comfortable with how neural networks are trained (gradient descent, loss functions) and what problems standard Transformer-based LLMs face today.

If you are new to the concept of **World Models**, **State Space Models**, or **Embodied AI**, here are some core concepts and papers to skim before diving in:

- **World Models:**
  - Ha & Schmidhuber's [World Models (2018)](https://arxiv.org/abs/1803.10122)
  - Yann LeCun's [A Path Towards Autonomous Machine Intelligence (JEPA)](https://openreview.net/forum?id=BZ5a1r-kVsf)
- **State Space Models (SSMs):**
  - Gu & Dao's [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
  - Gu et al.'s [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396)
- **Continuous/Dynamical Deep Learning:**
  - Hasani et al.'s [Liquid Neural Networks (LNNs)](https://arxiv.org/abs/2006.04439)
  - Chen et al.'s [Neural Ordinary Differential Equations (Neural ODEs)](https://arxiv.org/abs/1806.07366)

---

## The Roadmap (Phase-by-Phase)

| Phase | Focus | Status | Notebooks |
|-------|-------|--------|-----------|
| 0 | The Math Engine (ODEs, bifurcations, phase plane, limit cycles, chaos) | **Done** | `Calculus & Dynamics/**` |
| 1–3 | Causal Logic (Bayes nets, d-sep, VE, BP, MCMC, do-calculus) | **Done** | `Probabilistic_Graphs_and_State/**` |
| 4 | The State (SSMs: HiPPO, S4, Mamba) | **Done** | `State_Space_Models/SSM_S4_Mamba_HiPPO.ipynb` |
| 5 | Emergence (fractals, attractors, chaos) | **Done** | `Calculus & Dynamics/chaos/**` |
| 6 | Reasoning (MCTS, look-ahead, interventions) | **Done** | `Reasoning_and_Planning/MCTS_Lookahead_Planning.ipynb` |
| 7 | Embodiment / World Models (Dreamer-style + JEPA) | **Done** | `World_Models/Dreamer_JEPA_World_Models.ipynb` |
| 8 | Meta-Learning (MAML, Reptile, few-shot adaptation) | **Done** | `Meta_Learning/MAML_and_Fast_Adaptation.ipynb` |

**Study path:** start with `LEARNING_GUIDE.md` (plain-language tour of every phase), then run notebooks in order.

---

## Repository Structure

```
Causal-Dynamical-AI/
├── Calculus & Dynamics/                 # Phase 0 + 5
│   ├── 1D flows/
│   │   ├── Stability & Potential/Flows_on_the_Line.ipynb
│   │   └── Bifurcations/Bifurcations.ipynb
│   ├── 2D flows/
│   │   ├── Linear system/Linear_Systems.ipynb
│   │   ├── Phase_Plane/Phase_Plane.ipynb
│   │   └── Limit Cycles/Limit_Cycles.ipynb
│   └── chaos/
│       ├── Discrete_Maps_Hyperbolic.ipynb
│       └── fractals/fratals.ipynb
├── Probabilistic_Graphs_and_State/      # Phase 1–3
│   ├── Belief_Architecture.ipynb        # Bayes, BN, d-sep, VE, KL
│   └── Inference_Algorithms_and_Causality.ipynb  # BP, MCMC, do-calculus
├── State_Space_Models/                  # Phase 4
│   └── SSM_S4_Mamba_HiPPO.ipynb
├── Reasoning_and_Planning/              # Phase 6
│   └── MCTS_Lookahead_Planning.ipynb
├── World_Models/                        # Phase 7
│   └── Dreamer_JEPA_World_Models.ipynb
├── Meta_Learning/                       # Phase 8
│   └── MAML_and_Fast_Adaptation.ipynb
├── scripts/                             # notebook builders / executors
├── LEARNING_GUIDE.md                    # plain-language study path
├── requirements.txt
└── README.md
```

---

## Suggested Reading Order

1. **Flows on the Line** → stability as memory
2. **Bifurcations** → when rules (hyperparameters) change qualitative behavior
3. **Linear Systems** → eigenvalues of $A$ as the health of a latent state
4. **Phase Plane** → nonlinear geometry, separatrices as decision boundaries
5. **Limit Cycles** → rhythm, oscillations, inference guardrails
6. **Discrete Maps & Fractals** → autoregression as iterated maps; chaos metrics
7. **Belief Architecture** → causal graphs, d-separation, variable elimination
8. **Inference & Causality** → BP, Gibbs/MH, $P(Y\mid do(X))$ vs $P(Y\mid X)$
9. **SSM / S4 / Mamba** → scalable continuous-time memory
10. **MCTS Planning** → look-ahead using a model as an interventional simulator
11. **World Models** → learn the simulator; MPC in imagination; JEPA latents
12. **MAML / Reptile** → fast adaptation when the task/world shifts

---

## Quick Start

```bash
git clone https://github.com/Nisaral/Causal_dynamical_AI.git
cd Causal_dynamical_AI

python -m pip install -r requirements.txt
jupyter lab
```

Open any notebook above. Figures are saved next to each notebook when cells are run.

To rebuild / re-execute the Phase 1–6 continuation notebooks:

```bash
python scripts/build_inference_notebook.py
python scripts/build_ssm_notebook.py
python scripts/build_mcts_notebook.py
python scripts/execute_notebooks.py
```

---

## How I Document

- **LaTeX for Math:** Re-derive key results from Strogatz, Murphy, Pearl, Gu & Dao.
- **Manual vs. Package:** Prefer raw NumPy implementations (Euler, BP, MH, ZOH discretisation, UCT) before black-box libraries.
- **Visualizations:** Every concept gets a plot — dynamics are best understood through geometry.
- **AI bridge sections:** Each notebook ends with an explicit map to world models / LLMs / agents, carefully separated from proven textbook theorems.

---

## Resources used

- Nonlinear Dynamics and Chaos (Strogatz)
- Probabilistic Machine Learning (Kevin P. Murphy)
- Causality (Judea Pearl)
- S4 / HiPPO / Mamba papers (Gu, Dao, et al.)
- AlphaGo / AlphaZero / MuZero (Silver, Schrittwieser, et al.)

## Contributing & Chat

This is a learning journey, not a finished product. If you find a mistake in a derivation or have a better way to implement a Mamba-scan, please open an **Issue** or a **PR**!

> **Disclaimer:** This repo is for educational and research exploration purposes. I'm just an enthusiast trying to understand how we build the "whole car" of intelligence, not just the engine.
