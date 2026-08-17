# V4d — Robustness and Held-Out Generalization

## Motivation

By the end of V4c, the policy itself was no longer changing. Inventory control used

$$
center_t = S_t - k q_t,
$$

and toxicity control acted directly on estimated adverse markout:

$$
h_t = \frac{1}{\kappa} + \lambda \max(\hat m_t, 0).
$$

The remaining question was not whether another mechanism could improve the model, but whether the existing policy generalized beyond the environments and seeds used while developing it.

V4d therefore imposed a strict workflow:

```text
DEV
→ VALIDATION
→ FREEZE
→ HELD-OUT TEST
→ STRESS
```

The fixed architecture was

```text
k = 0.04
N = 50
h_warmup = 1.08
```

and the only development degree of freedom was

$$
\lambda \in \{0.5, 1.0, 1.5\}.
$$

The primary environment remained

```text
sigma = 0.3
A = 0.4
kappa = 1.0
p = 0.002
n_steps = 4500
```

Fresh seed blocks were reserved before running the experiments:

```text
DEV         3000–3499
VALIDATION  3500–3999
TEST        4000–6999
STRESS      7000–7999
```

No policy parameter was changed after the freeze.

---

## Experiment 1 — Development, Validation, and Freeze

### Question

Does the final direct-markout policy require any retuning before held-out evaluation?

### Hypothesis

The existing default \(\lambda=1\) was not assumed to remain best. Any change, however, had to be selected using DEV only and confirmed on VALIDATION before TEST was opened.

No finer grid or additional candidates were allowed.

### Design

The 500 DEV seeds compared

```text
lambda = 0.5
lambda = 1.0
lambda = 1.5
```

using mean terminal PnL as the selection criterion.

If the DEV winner differed from \(\lambda=1\), that candidate would be compared with the default on the 500 validation seeds using a paired difference. A non-default value would only be accepted if its validation improvement remained clearly positive.

### Results

```text
lambda=0.5   mean PnL = 584.81
lambda=1.0   mean PnL = 586.35
lambda=1.5   mean PnL = 585.42
```

The DEV winner was

$$
\boxed{\lambda^*=1.0}.
$$

Because the default itself won, no validation contest against a non-default candidate was needed.

The final policy was frozen as

```text
lambda = 1.0
k = 0.04
N = 50
h_warmup = 1.08
```

before the held-out TEST seeds were opened.

### Conclusion

The original direct-markout rule survived development selection without retuning.

---

## Experiment 2 — Held-Out Test

### Question

Do the main results from the earlier V4 experiments replicate on untouched seeds under hidden stochastic toxicity?

### Hypothesis

Three findings should survive:

1. inventory skew should sharply reduce inventory and terminal-PnL dispersion;
2. toxicity adaptation should retain a small positive economic effect;
3. inventory and toxicity control should remain approximately additive rather than strongly interacting.

A further prediction from V4a was that inventory control should reduce the sampling noise of the toxicity treatment effect.

### Design

The 3000 TEST seeds were opened once after the policy freeze.

A full \(2\times2\) ablation compared:

```text
                 Toxicity OFF       Toxicity ON
Inventory OFF    F                  T
Inventory ON     I                  IT
```

with

```text
F   k=0,    h=1.08
I   k=.04,  h=1.08
T   k=0,    direct-markout adaptation
IT  k=.04,  direct-markout adaptation
```

The adaptive policies used the frozen \(\lambda=1\).

All four cells used same-seed common random numbers. Reported metrics were terminal PnL, PnL standard deviation, 5th-percentile PnL, RMS inventory, mean maximum absolute inventory, and fills.

Paired treatment effects were computed for

$$
I-F,\quad T-F,\quad IT-I,\quad IT-T,
$$

with standard errors and 95% confidence intervals.

The factorial interaction was

$$
\Gamma = IT-I-T+F.
$$

### Results

```text
       mean PnL   std PnL     p05    RMS inv   max|q|   fills
F       599.55    351.13     -6.31   15.400    30.58    611.2
I       585.62     77.09    458.73    3.461    10.04    617.2
T       600.83    345.96      4.75   15.224    30.07    590.1
IT      587.87     76.69    460.72    3.460    10.01    596.0
```

#### Inventory control

Without toxicity adaptation,

$$
I-F=-13.936,
$$

with

$$
SE=6.028,
\qquad
95\%CI=[-25.751,-2.120].
$$

With toxicity adaptation,

$$
IT-T=-12.954,
$$

with

$$
SE=5.923,
\qquad
95\%CI=[-24.563,-1.346].
$$

RMS inventory fell from roughly

$$
15.4 \rightarrow 3.46,
$$

a reduction of about 77.5%.

Terminal-PnL standard deviation fell from roughly 350 to 77, while 5th-percentile PnL moved from around zero to about 460.

The held-out sample therefore reproduced the earlier inventory trade-off: much lower inventory and downside dispersion at a modest cost in mean PnL.

#### Toxicity adaptation

Without inventory control,

$$
T-F=+1.273,
$$

with

$$
SE=1.757,
\qquad
95\%CI=[-2.171,+4.717].
$$

With inventory control,

$$
IT-I=+2.255,
$$

with

$$
SE=0.389,
\qquad
95\%CI=[+1.492,+3.018].
$$

The point effects were of similar scale, but the precision was very different.

The paired standard-error ratio was

$$
\frac{0.389}{1.757} \approx 0.221.
$$

Inventory control therefore reduced the standard error of the toxicity treatment effect by about 78%.

This closely reproduced the V4a result. Inventory control did not materially amplify the toxicity effect; it suppressed inventory-driven PnL noise and made the smaller effect easier to measure.

#### Interaction

The factorial interaction was

$$
\Gamma=+0.981,
$$

with

$$
SE=1.762,
\qquad
95\%CI=[-2.472,+4.435].
$$

There was no statistically resolved evidence of a material positive or negative interaction.

### Conclusion

The main V4a findings replicated on untouched stochastic held-out seeds.

Inventory and toxicity controls remained approximately modular. Inventory control again removed most inventory-driven dispersion, while the smaller toxicity-adaptation effect became statistically well resolved once that noise was suppressed.

---

## Experiment 3 — Stress Tests

### Question

Where does the frozen policy begin to weaken when the observation problem becomes harder?

### Hypothesis

The policy should not be expected to perform equally well under every environment.

Two stresses directly challenge the fixed \(N=50\) estimator:

- faster hidden-state switching;
- fewer fills.

Higher volatility instead increases the scale of adverse markout, so the direct-markout policy may become more economically valuable.

No policy parameter was changed under stress.

### Design

Each pre-specified stress used 1000 fresh seeds.

Only two policies were compared:

```text
I   inventory-only
IT  frozen integrated policy
```

This isolates the incremental value of toxicity adaptation after inventory risk is already controlled.

The stress environments were:

```text
A: faster switching   p = 0.004
B: higher volatility  sigma = 0.4
C: lower fill rate    A = 0.25
```

Reported metrics were paired \(IT-I\) PnL, RMS inventory, estimator RMSE, transition-resolution rate, and fills.

### Results

#### A — Faster switching

$$
IT-I=+1.631,
$$

with

$$
SE=0.647,
\qquad
95\%CI=[+0.363,+2.900].
$$

```text
IT estimator RMSE       = 0.4700
IT resolution rate      = 0.538
IT fills                = 595
RMS inventory: I / IT   = 3.471 / 3.469
```

Estimator quality deteriorated as expected. Faster regime changes reduced the probability that the rolling window could complete its response before the next switch.

The adaptive PnL benefit became smaller but remained positive and statistically resolved at this stress level.

#### B — Higher volatility

$$
IT-I=+5.508,
$$

with

$$
SE=1.003,
\qquad
95\%CI=[+3.543,+7.474].
$$

```text
IT estimator RMSE       = 0.4086
IT resolution rate      = 0.684
IT fills                = 575
RMS inventory: I / IT   = 3.500 / 3.491
```

The incremental value of toxicity adaptation increased substantially.

This is consistent with the direct-markout policy responding naturally to a larger adverse-selection cost scale, although the stress test alone does not establish that mechanism.

#### C — Lower fill intensity

$$
IT-I=+0.626,
$$

with

$$
SE=0.597,
\qquad
95\%CI=[-0.545,+1.796].
$$

```text
IT estimator RMSE       = 0.4487
IT resolution rate      = 0.576
IT fills                = 373
RMS inventory: I / IT   = 3.447 / 3.446
```

With fewer observations, the fixed \(N=50\) estimator required more calendar time to update.

Tracking quality deteriorated and the incremental toxicity benefit became smaller and statistically unresolved.

### Conclusion

The stress tests exposed sensible limits rather than structural failures.

The adaptive component weakened when latent toxicity changed too quickly or informative fills arrived too slowly relative to the estimator's memory. Inventory control remained stable throughout.

---

## Conclusion

V4d evaluated the final market-making policy under a strict development, freeze, and held-out testing workflow.

The original \(\lambda=1\) direct-markout rule survived development selection without retuning. On 3000 untouched TEST seeds, the main earlier findings replicated: inventory skew reduced inventory and terminal-PnL dispersion by roughly 78% at a modest cost in mean PnL, inventory and toxicity controls remained approximately additive, and suppressing inventory-driven noise again reduced the standard error of the toxicity treatment effect by roughly 78%.

The stress tests exposed clear operating limits without motivating further tuning. Faster toxicity switching and lower fill intensity degraded estimator tracking and reduced the incremental value of adaptation. Higher volatility increased the value of reacting to adverse markout. Across all three stresses, inventory control remained stable.

The final policy is therefore not uniformly optimal across every observation regime. Its adaptive advantage depends on toxicity persisting long enough, and fills arriving frequently enough, for the rolling estimator to extract useful information.

No parameters were retuned after the policy was frozen.

**V4d closes the V4 research cycle.**
