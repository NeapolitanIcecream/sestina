# Related-Work Audit For Sestina

Date: 2026-05-02

Workflow: `sestina-related-work-audit`

## Scope

This audit grounds the current Sestina historical arXiv pilot conclusion against
related work on active ranking, noisy top-K aggregation, dueling bandits,
crowdsourced top-K selection, low-budget active-learning failure modes, and
Bradley-Terry / Plackett-Luce implementation libraries.

No Sestina paid LLM calls were made for this audit. The current empirical
evidence remains:

- 8 historical arXiv buckets, 634 papers, 40 future-citation top-K positives.
- Historical random + posterior top-K: Recall@K 0.375, nDCG@K 0.412096, AP
  0.407579.
- Exact-pool random + posterior top-K: Recall@K 0.375, nDCG@K 0.404687, AP
  0.381836.
- Complete active/candidate/decision arms recover 13/40 or 14/40 positives;
  historical random and exact-pool random recover 15/40.
- Negative or inconclusive Sestina arms so far: original active, revised active,
  posterior top-K EVSI, sequential EVSI, CCTD-GF, expanded-pool random,
  targeted-outsider random, degree-aware posterior shrinkage, and soft-strength
  calibration.

## Bottom Line

The literature supports keeping random or exact-pool random plus posterior
top-K as the default small-budget baseline. Low-budget active methods can be
variance-sensitive, random controls need to be strong, and sparse noisy pairwise
graphs can make posterior/model tweaks insufficient.

The literature also challenges a stronger conclusion that active scheduling is
hopeless. Several active ranking and dueling-bandit papers obtain large
comparison savings, but under conditions Sestina has not yet demonstrated:
stochastic and stationary pairwise preferences, identifiable gaps around the
top-K boundary, enough repeated comparisons to build confidence intervals, and
fixed-confidence or PAC stopping rules rather than a single fixed budget with
one label per pair.

The next Sestina step should be a no-paid design gate: implement and replay a
confidence-interval top-K partition/elimination scheduler with a randomized
coverage floor against existing cached labels and retrospective diagnostics. Do
not buy new labels unless the offline gate beats exact-pool random across
multiple seeds and preserves the weak-bucket oracle caps.

## What The Literature Supports

- Strong random controls are not optional. CrowdTopK and Active Evaluation both
  compare active/heuristic methods against uniform or broad baselines; low-budget
  active-learning work explicitly shows that active methods can fail to beat
  random when the budget is too small.
- Sparse pairwise evidence limits model-level fixes. Heckel et al. show that
  Bradley-Terry-Luce style parametric assumptions provide at most logarithmic
  sample-complexity gains for the stochastic comparison setting they analyze,
  which is consistent with Sestina's negative degree-shrinkage and soft-strength
  calibration results.
- Low-budget selection needs coverage before fine uncertainty. Hacohen et al.,
  ProbCover, and Uncertainty Herding all argue that the right query strategy
  depends on budget regime; low-budget regimes favor typicality/coverage, while
  uncertainty becomes safer later.
- Dueling-bandit and active-evaluation papers reinforce the need for confidence
  accounting. Their wins are framed around high-probability identification,
  annotation complexity, or regret. Sestina's current active arms are not yet
  using a comparable stopping/sample-complexity gate.

## What The Literature Challenges

- Active ranking can work. Jamieson and Nowak, Mohajer et al., Falahatgar et al.,
  RUCB, and Sparse Dueling Bandits all show comparison savings under assumptions
  on geometry, stochastic transitivity, Borda/Condorcet structure, or pairwise
  probability gaps.
- Sestina's one-seed, 8-bucket evidence is weak as a general empirical claim.
  The observed random advantage is only 1-2 positives versus several arms. It is
  enough to block a larger paid run, but not enough to reject active scheduling
  as a class.
- Sestina may be using the wrong active-learning frame. The literature often
  treats top-K identification as accepting/rejecting items with confidence
  intervals. Sestina's EVSI and CCTD-GF arms mainly rank candidate pairs inside a
  fixed budget and then score posterior membership; they do not certify that
  outsiders are separated from the K-th item.
- Single judgments per pair are a mismatch for many guarantees. Several
  algorithms assume repeated independent comparisons or stable pairwise
  probabilities. Sestina's LLM judge gives one label per scheduled pair in the
  current small-budget design.

## Related-Work Matrix

The machine-readable version of this matrix is
`artifacts/backtest-arxiv-related-work/related-work-matrix.json`.

| Source | Year / venue | Problem setting | Key assumptions | Algorithmic idea | Baselines / random controls | Relevance to Sestina | Supports or challenges current conclusion | Concrete implication |
|---|---|---|---|---|---|---|---|---|
| [Jamieson & Nowak, Active Ranking using Pairwise Comparisons](https://papers.nips.cc/paper/4427-active-ranking-using-pairwise-comparisons) | 2011, NeurIPS | Recover a full ranking from adaptively selected pairwise comparisons. | Items embed in low-dimensional Euclidean space and ranking reflects distance from a reference point; robust variant assumes comparisons are probably correct. | Adaptive comparison design exploits geometric structure; random comparisons can need nearly all pairs under their model. | Explicit analytical contrast with random pair selection. | Directly adjacent to active pair scheduling, but Sestina does not currently have a validated low-dimensional ranking geometry. | Challenges: active can beat random if structure holds. | Add a preflight check for any claimed structure; do not rely on active gains unless candidate features predict boundary errors. |
| [Mohajer, Suh & Elmahdy, Active Learning for Top-K Rank Aggregation from Noisy Comparisons](https://proceedings.mlr.press/v70/mohajer17a.html) | 2017, ICML | Top-K sorting and top-K partitioning from noisy pairwise comparisons. | General stochastic model covering SST, BTL, and uniform noise; active sequential design. | Active algorithm for top-K sorting/partitioning with sample-complexity gains over passive ranking. | Compares active and passive sample complexity. | Direct match to Sestina's top-K decision objective, but assumes a clean stochastic ranking model. | Challenges: active top-K can work. | Prototype a top-K partition confidence scheduler before another EVSI score tweak. |
| [Heckel et al., Active Ranking from Pairwise Comparisons and when Parametric Assumptions Do Not Help](https://arxiv.org/abs/1606.08842) | 2019, Annals of Statistics; arXiv 2016 | Active ranking/top-K partitioning from noisy pairwise probabilities. | Items ranked by probability of beating a random item; stochastic comparisons; confidence intervals over win counts. | Sequential count-and-confidence algorithm decides whether to stop or compare another pair. | Theoretical lower bounds; contrasts nonparametric guarantees with parametric BTL/Thurstone assumptions. | Explains why BT/PL posterior tweaks may not solve acquisition failure by themselves. | Supports model-tweak caution; challenges scheduler frame. | Implement a no-paid confidence-interval partition replay and require evidence that it moves false negatives, not only increases degree. |
| [Falahatgar et al., Maximum Selection and Ranking under Noisy Comparisons](https://proceedings.mlr.press/v70/falahatgar17a.html) | 2017, ICML | PAC maximum selection and ranking via pairwise comparisons. | Strong stochastic transitivity and stochastic triangle inequality; repeated independent comparisons. | Knockout-style maximum selection and noisy-binary-search ranking with near-optimal comparison complexity. | Compares to prior adaptive algorithms; not a Sestina-style random control. | Suggests a top-K/best-item scheduler should reason about pairwise gaps and confidence, not just posterior EVSI. | Challenges: fixed-budget one-label scheduling may miss the PAC framing. | Gate next paid work on simulated gap/confidence behavior around the K-th boundary. |
| [Zoghi et al., Relative Upper Confidence Bound for the K-Armed Dueling Bandit Problem](https://proceedings.mlr.press/v32/zoghi14.html) | 2014, ICML | Dueling bandits with relative feedback between arms. | Preference matrix has a suitable winner class; finite-time regret analysis. | RUCB estimates pairwise probabilities and selects promising duels using upper confidence bounds. | Empirical comparison to state of the art on information retrieval data. | Relevant to pair choice and confidence bounds; less direct because Sestina wants top-K papers, not regret-minimizing online ranker choice. | Challenges: UCB-style pair selection is a plausible missing frame. | If implemented, compare RUCB/RMED-like scheduling to exact-pool random offline before paid calls. |
| [Yue & Joachims, Beat the Mean Bandit](https://icml.cc/2011/papers/200_icmlpaper.pdf) | 2011, ICML | Dueling bandits with noisy preference feedback. | Relaxes strong transitivity assumptions; targets good arms under relative preferences. | Eliminates arms by comparing against the empirical mean of surviving arms. | Empirical comparisons in retrieval-like settings. | Points toward survivor-set elimination rather than ranking all candidate pairs by EVSI. | Challenges: active elimination can help under suitable preference structure. | Consider survivor elimination only if Sestina can define stable paper-level win probabilities. |
| [Jamieson et al., Sparse Dueling Bandits](https://arxiv.org/abs/1502.00133) | 2015, AISTATS | Pure exploration to find a Borda winner from noisy pairwise comparisons. | Borda score objective; possible sparsity in pairwise comparison matrix. | Successive Elimination with Comparison Sparsity exploits informative comparisons. | Compares against standard dueling algorithms on synthetic and real data. | Sestina diagnostics already include graph degree and topology; sparse-dueling framing explains why topology alone is insufficient unless it targets informative entries. | Mixed: supports graph diagnostics but challenges naive graph floors. | Add "informative gap entries found" diagnostics, not only component size or degree. |
| [Zhang, Li & Feng, Crowdsourced Top-k Algorithms](https://www.vldb.org/pvldb/vol9/p612-zhang.pdf) and [CrowdTopK project](https://dbgroup.cs.tsinghua.edu.cn/ligl/crowdtopk/) | 2016, PVLDB | Crowdsourced top-K from noisy pairwise or rating data. | Human workers can be wrong; pair selection and inference are separable steps. | Systematic evaluation of more than twenty selection and inference algorithms, including Borda, Copeland, Rank Centrality, Elo, TrueSkill, CrowdBT, HodgeRank, and top-K heuristics. | Broad experimental comparison across synthetic, real, and real-crowd data; includes code/datasets. | Strongly relevant to Sestina's pair selection plus inference split and noisy judge setting. | Supports: compare many arms against strong controls before scaling. | Reuse their pair-selection/inference taxonomy in Sestina reports; keep pair selection and posterior decision failures separate. |
| [Mohankumar & Khapra, Active Evaluation](https://aclanthology.org/2022.acl-long.600.pdf) and [DuelNLG](https://github.com/akashkm99/duelnlg) | 2022, ACL | Identify the top-ranked NLG system with few pairwise human comparisons. | Stationary pairwise preference distribution; dueling-bandit assumptions such as Condorcet/Copeland variants may not be verifiable a priori. | Evaluates 13 dueling-bandit algorithms; model-based variants combine automatic metrics with human comparisons. | Uniform exploration is the main annotation-complexity baseline. | Closest project analogue for LLM/NLG pairwise evaluation and dueling-bandit code. | Mixed: active can reduce annotations, but only with assumptions and many seeds. | Borrow the "annotation complexity over many seeds" evaluation before any paid Sestina run. |
| [Attenberg & Provost, Inactive Learning?](https://fosterprovost.com/publication/inactive-learning-difficulties-employing-active-learning-in-practice/) | 2010, SIGKDD Explorations | Practical difficulties deploying active learning. | Real applications face cold start, biased samples, model-selection uncertainty, and evaluation friction. | Essay-style audit of under-discussed practical challenges. | Random sampling is the default practical reference point. | Relevant because Sestina is exactly in a cold-start, budget-limited, high-imbalance setting. | Supports caution: active learning can underdeliver in practice. | Keep explicit random controls and require deployment-style gates, not only algorithmic novelty. |
| [Hacohen, Dekel & Weinshall, Active Learning on a Budget](https://proceedings.mlr.press/v162/hacohen22a.html) | 2022, ICML | Active learning strategy choice as a function of label budget. | Query strategy effectiveness changes with budget; typical examples help at low budget, uncertainty later. | TypiClust selects typical cluster representatives for low budgets. | Compares against active-learning baselines and random sampling. | Matches Sestina's low-pairwise-budget regime and explains why uncertainty-heavy EVSI can underperform. | Supports: low-budget uncertainty can be wrong default. | Next scheduler should have a coverage/typicality/random floor and only use uncertainty after coverage criteria pass. |
| [Yehuda et al., Active Learning Through a Covering Lens](https://openreview.net/forum?id=u6MpfQPx9ck) | 2022, NeurIPS | Low-budget active learning as probability coverage. | Good representations expose data geometry; low-budget selections should cover dense regions. | ProbCover maximizes probability coverage instead of chasing only uncertainty. | Evaluates against random and active-learning baselines. | Supports Sestina's diagnostic that touching more papers or higher degree is not enough; coverage has to be decision-relevant. | Supports random/coverage floor. | Add coverage metrics over pointwise-false-negative neighborhoods, not just unique papers touched. |
| [Bae, Sutherland & Oliveira, Uncertainty Herding](https://openreview.net/forum?id=UgPoHhYQ2U) | 2025, ICLR | Active learning across low and high label budgets. | Budget regime is problem-dependent; uncertainty-only methods can fail at low budgets. | Uncertainty coverage interpolates between coverage and uncertainty. | Empirical validation against active-learning baselines. | Reinforces that Sestina should not trust posterior uncertainty alone under a tiny pairwise budget. | Supports current conclusion. | A Sestina arm should predeclare when it switches from coverage/random floor to uncertainty-heavy acquisition. |
| [choix](https://choix.lum.li/en/stable/) | Project docs, Python library | Inference for Luce-choice models: Bradley-Terry, Plackett-Luce, top-1 choices, network choice. | Luce choice axiom; regularization may be needed; graph/data support matters. | Provides LSR, MM, Rank Centrality, and approximate Bayesian inference via expectation propagation. | Not an experiment; implementation reference. | Useful for checking Sestina's BT/PL aggregation against standard algorithms. | Implementation check, not direct evidence. | Compare Sestina posterior top-K on cached data against `choix` MM/RankCentrality/EP as a no-paid sanity check. |
| [Crowd-Kit pairwise aggregation](https://crowd-kit.readthedocs.io/en/latest/pairwise/) | Project docs, Python library | Pairwise crowdsourcing aggregation. | Bradley-Terry implementation needs a strongly connected comparison graph; NoisyBradleyTerry models worker skill and bias. | Implements BT and noisy BT aggregators over pairwise comparisons. | Not an experiment; implementation reference. | Relevant to Sestina's sparse graph diagnostics and possible judge-noise modeling. | Supports graph-connectivity caution. | Before adopting BT-like changes, report strong-connectivity and low-degree failure modes per bucket. |
| [Mattos & Ramos, Bayesian paired comparison with the bpcs package](https://davidissamattos.github.io/bpcs/) | 2021 article / R package | Bayesian paired-comparison models in Stan. | Bayesian BT/Davidson extensions; priors help when MLE is unstable or nonexistent; can model ties/extensions. | Stan/NUTS posterior inference for paired-comparison data and rank distributions. | Not an acquisition benchmark. | Useful for posterior uncertainty and tie modeling, but not a pair scheduler by itself. | Mixed: supports posterior diagnostics, not paid acquisition. | If label interpretation is revisited, test Bayesian tie/uncertain-label models offline first. |
| [Turner et al., PlackettLuce R package](https://arxiv.org/abs/1810.12068) | 2020, Journal of Statistical Software; arXiv 2018 | Model ranking data with Plackett-Luce and extensions. | Ranking data can be represented as paired comparisons; connectivity/regularization issues matter. | R package for Plackett-Luce ranking models, including ties and item worth estimates. | Not an acquisition benchmark. | Relevant to implementation checks for ranking/posterior alternatives. | Implementation check. | Use as a cross-check for rank-model behavior, not as justification for new paid labels. |

## Project And Library Shortlist

- `choix`: best Python candidate for quick BT/PL, Rank Centrality, LSR, MM, and
  approximate Bayesian sanity checks on cached comparisons.
- Crowd-Kit: useful if Sestina wants a worker/judge reliability model. Its docs
  explicitly flag strong connectivity for basic Bradley-Terry, which should stay
  in Sestina diagnostics.
- `bpcs`: useful for Bayesian BT/Davidson tie modeling in R/Stan if we need a
  higher-fidelity posterior check. It is heavier than a scheduler gate.
- PlackettLuce R package: useful for independent PL/ranking-model checks,
  especially ties and worth estimates.
- CrowdTopK: useful taxonomy and legacy code for pair-selection and inference
  algorithms, but the project code is old Python 2 and should be treated as a
  reference, not a direct dependency.
- DuelNLG: useful implementation reference for dueling-bandit evaluation,
  repeated seeds, and annotation-complexity reporting.

## Recommended Next Action

Do not spend on another acquisition tweak yet. Implement a no-paid design gate:

1. Add an offline replay harness over existing historical/cached pairwise labels.
   It should run at least 20 seeds for each candidate scheduler and exact-pool
   random with the same feasible pools.
2. Prototype a confidence-interval top-K partition/elimination scheduler:
   maintain paper-level win/Borda-style estimates; repeatedly compare only
   unresolved candidate-vs-boundary or candidate-vs-outsider pairs; accept or
   reject papers when confidence bounds separate them from the K-th boundary.
   Keep a randomized coverage floor so low-budget behavior cannot collapse into
   over-focused uncertainty sampling.
3. Gate with retrospective metrics that do not use future labels for scheduling:
   Recall@K, nDCG@K, AP, pointwise-plus-touched oracle cap,
   positive-negative-pair oracle cap, weak-bucket deltas, unique positives
   touched, graph connectivity, and confidence-bound unresolved count.
4. Proceed to paid labels only if the offline gate beats exact-pool random
   across seeds on Recall@K/nDCG@K or materially improves weak-bucket oracle
   caps without losing the exact-pool random floor.

Preconditions for the scheduler to be credible:

- Pairwise labels must be treated as noisy stochastic evidence, not deterministic
  truth.
- The design must state how repeated comparisons are represented under the
  current one-label-per-pair cache. If it needs repeated independent labels, that
  requirement must be explicit before any paid run.
- The candidate set must have enough pointwise recall. If candidate recall is
  poor in a bucket, no pairwise scheduler can recover the missing positives
  unless it creates model-visible outsider challengers.
- All paid artifacts must store `scheduled_pair` diagnostics for reused and new
  labels so future audits do not depend on reconstructing schedules after code
  changes.

## Limitations

- This is a focused audit, not a complete survey of ranking theory or preference
  learning.
- Sources were restricted to primary papers, proceedings pages, arXiv pages, and
  official project/library docs where available.
- No external project code was run.
- Sestina's current empirical evidence remains one seed over 8 historical
  buckets; the literature raises enough uncertainty that this should be treated
  as a stop-and-gate result, not a broad negative theorem about active ranking.
