# State-Space Models — Expanded Study Notes

**Primary deliverable (read this):** [`SSM_Learning_Notes.pdf`](SSM_Learning_Notes.pdf) — **27 pages**, full derivations, GPU theory, paper lineage, industry pipelines, code, and **14 explained figures**.

**Figures folder:** `State_Space_Models/figures/`  
**Regenerate figures:** `python State_Space_Models/scripts/gen_ssm_figures.py`  
**Notebook:** `State_Space_Models/SSM_S4_Mamba_HiPPO.ipynb`  
**Lecture:** https://www.youtube.com/watch?v=A7iAmVr0QE4

---

## What’s in the PDF (by part)

1. **Problem setup** — attention $O(L^2)$, KV-cache, complexity + memory growth graphs  
2. **Continuous SSMs** — full variation-of-parameters derivation, impulse response, transfer functions + plots  
3. **Discretization** — ZOH proof, bilinear/Tustin, half-life heatmap  
4. **Unrolling / FFT** — kernel $K_m$, multi-scale modes, scan≡FFT numerical check  
5. **HiPPO** — polynomial memory + eigenvalue spectrum figure  
6. **S4 / S4D / DPLR / Woodbury / Cauchy** — structure for computable kernels  
7. **H3 / Hyena → Mamba** — selection math, parallel scan monoid, architecture diagram  
8. **Mamba-2 / SSD** — tensor-core chunking  
9. **GPU theory** — HBM vs SRAM, roofline, fusion, IO complexity, numerics  
10. **Code** — LTI + selective educational impl + HF / `mamba-ssm` production entry points  
11. **Industry** — Codestral Mamba, Jamba, Granite hybrids, Nemotron-H-class, serving pipeline figure  
12. **Limits, exercises, glossary, references**

---

## Figure guide (same as PDF captions)

| File | What it teaches |
|------|-----------------|
| `fig01_impulse_response.png` | Real poles → forget speed |
| `fig02_oscillatory_kernel.png` | Complex poles → ringing memory |
| `fig03_discretization.png` | Continuous fade vs token samples |
| `fig04_multiscale_kernel.png` | Diagonal SSM = sum of geometrics |
| `fig05_scan_vs_fft.png` | LTI identity: scan = FFT conv |
| `fig06_selective_vs_lti.png` | Why selection helps on spikes-in-noise |
| `fig07_complexity.png` | $L^2$ vs $L\cdot N$ compute growth |
| `fig08_memory_growth.png` | KV-cache vs flat SSM state |
| `fig09_gpu_hierarchy.png` | Why fused kernels exist |
| `fig10_parallel_scan.png` | Blelloch scan tree |
| `fig11_hippo_eigs.png` | Stable multi-scale HiPPO spectrum |
| `fig12_mamba_block.png` | Expand→Conv→SSM→Gate→Project |
| `fig13_halflife.png` | $m_{1/2}=\ln2/(a\Delta)$ heatmap |
| `fig14_industry_pipeline.png` | Long context → serve path |

---

## How to study

Open the **PDF** and read top to bottom. For each figure: read the caption, then the paragraphs above it. Re-derive boxed/numbered equations on paper. Run matching notebook cells. Use this Markdown only as a map / quick figure index while annotating.

**Rebuild PDF after edits:**
```bash
cd State_Space_Models
python scripts/gen_ssm_figures.py
pdflatex SSM_Learning_Notes.tex
pdflatex SSM_Learning_Notes.tex
```
