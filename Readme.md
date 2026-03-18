# Causal-Dynamical AI

## Why this repo?

I'm fascinated by a specific gap in modern AI. Most current models are incredible "statistical pattern matchers." They predict the next word or pixel with mind-bending accuracy by interpolating across massive datasets. However, as noted by researchers like Yann LeCun (Ex-Meta) and Judea Pearl, they often lack an internal "physics" of the world.

**The Shortcomings I'm exploring:**

- **Causality:** They often mistake correlation for causation (the "Causal Parrot" problem).
- **Planning:** They struggle with "Look-ahead" — reasoning through the consequences of an action before taking it.
- **Extrapolation:** They are brilliant at repeating what they've seen, but often fail when faced with structural changes that weren't in the training data.

This repo is my one unified notebook for exploring how to fuse **Dynamical Systems** (state evolution), **PGMs** (causal logic), and **Deep Learning** (scalable training) to build World Models.

---

## What is a World Model?

In this repo, we study World Models not as a single architecture, but as a system that allows an agent to simulate reality. We will be implementing and studying the **"Big Three"** approaches:

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
  - AI21's [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)
- **Continuous/Dynamical Deep Learning:**
  - Hasani et al.'s [Liquid Neural Networks (LNNs)](https://arxiv.org/abs/2006.04439)
  - Chen et al.'s [Neural Ordinary Differential Equations (Neural ODEs)](https://arxiv.org/abs/1806.07366)

---

## The Roadmap (Phase-by-Phase)

| Phase | Focus | Key Concepts | Resources |
|-------|-------|--------------|-----------|
| 0 | The Math Engine | ODEs, Vector Calculus, Discretization | Strogatz, MIT 18.03, 3B1B |
| 1–3 | Causal Logic | Bayesian Nets, MCMC, Belief Propagation | Murphy, Koller, Judea Pearl |
| 4 | The State (SSMs) | Mamba, S4, HiPPO, RNN Physics | Gu & Dao (Mamba) |
| 5 | Emergence | Cellular Automata, Attractors, Chaos | Wolfram, Strogatz Ch. 9 |
| 6 | Reasoning | MCTS (AlphaZero style), PDDL Planning | Silver et al., SymPy |
| 7 | Embodiment | V-JEPA, World Models, Video RAG | LeCun (JEPA), Hafner |
| 8 | Meta-Learning | MAML, Hypothesis Generation | Finn (MAML), learn2learn |

---

## Repository Structure

```
Causal-Dynamical-AI/
├── 01_Calculus_and_Dynamics/
│   ├── Ch02_Flows_on_the_Line/
│   │   ├── Ch02_Flows_on_the_Line.ipynb  # Intro + Theory + Relevance + Code + Viz
│   │   
│   ├── Ch03_Bifurcations/
│   │   ├── Ch03_Bifurcations.ipynb
│   │   
│   ├── Phase_Plane/
│   │   └── Phase_Plane.ipynb
│   └── ...
├── 02_Probabilistic_Graphs/
│   ├── Ch03_Bayesian_Networks/
│   │   ├── Ch03_Bayesian_Networks.ipynb
│   │   
├── src/                                  # Reusable core modules (pgm_utils.py, etc.)
├── experiments/                          # Large-scale benchmarks (e.g., Mamba vs Transformer)
└── README.md
```

---

## Quick Start (Clone & Explore)

If you want to run my derivations and simulations locally:

**1. Clone the Repo:**
```bash
git clone https://github.com/Nisaral/Causal_dynamical_AI.git
cd Causal_dynamical_AI

```

**2. Set up the Environment:**

I recommend using a virtual environment.
```bash
pip install -r requirements.txt
# Requirements include: torch, pgmpy, mamba-ssm, matplotlib, sympy
```

**3. Run the First Discovery:**

Open the Phase 0 notebook to see how a simple 1D ODE can model "memory" in a system.
```bash
jupyter lab notebooks/phase_00_fixed_points.ipynb
```

---

## How I Document

- **LaTeX for Math:** I re-derive everything from Strogatz and Murphy in LaTeX to ensure I actually understand the "why."
- **Manual vs. Package:** For every concept, I first write a manual implementation (e.g., a raw NumPy Euler solver) before using a standard package (e.g., `scipy` or `torchdiffeq`).
- **Visualizations:** Every notebook includes interactive plots. Dynamics are best understood through movement LOL!!

---
## Resources used
- Nonlinear Dynamics and Chaos (Strogatz)
- Probabilistic Machine Learning (Kevin P. Murph)


## Contributing & Chat

This is a learning journey, not a finished product. If you find a mistake in my derivations or have a better way to implement a Mamba-scan, please open an **Issue** or a **PR**!

> **Disclaimer:** This repo is for educational and research exploration purposes. I'm just an enthusiast trying to understand how we build the "whole car" of intelligence, not just the engine.