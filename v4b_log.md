# V4b — Hidden Stochastic Toxicity

## 1. Motivation and Research Question

V3b introduced online toxicity estimation, but its environment was deliberately simple:

$$
\phi: 0 \rightarrow 1 \rightarrow 0.
$$

That deterministic schedule was useful for validating estimator response, but it made regime changes unusually clean and persistent.

V4b asks the next question:

> **Does the same simple rolling-markout estimator remain useful when toxicity becomes genuinely hidden and stochastic?**

The estimator is deliberately kept unchanged. The purpose of V4b is not to introduce a more advanced estimator, but to make the environment harder and test whether the existing mechanism still works.

The hidden toxicity state becomes

$$
\phi_t \in \{0,1\},
$$

following a symmetric two-state Markov process:

$$
P(\phi_{t+1}\neq\phi_t)=p.
$$

Primary environment:

```text
sigma = 0.3
A = 0.4
kappa = 1
p = 0.002
mean regime length ≈ 500 ticks
N = 50
n_steps = 4500
```

V4b asks three questions:

1. Can a valid stochastic hidden-toxicity environment be constructed?
2. Can the existing `N=50` rolling estimator still track it?
3. Does the same estimator continue to work inside the full inventory-controlled maker?

---

## 2. Experiment 1 — Hidden Stochastic Toxicity Environment

### Question

Can the deterministic toxicity schedule be replaced by a genuinely stochastic hidden-state process without changing the rest of the market model?

### Hypothesis

A symmetric two-state Markov process with

$$
p=0.002
$$

should generate persistent but randomly timed regimes, with expected duration

$$
E[L]=\frac{1}{p}=500
$$

ticks.

This should create a harder environment than V3b while still leaving enough time for the existing `N=50` estimator to respond in many regimes.

### Design

Generate

$$
\phi_t\in\{0,1\}
$$

using an independent Markov state process.

The environment sanity checks verify that:

- empirical switching behavior is consistent with the specified transition probability;
- regime durations are sensible for a geometric process;
- paths are reproducible by seed;
- the maker never observes the hidden state directly.

No estimator comparison or PnL analysis is introduced at this stage.

### Result

The Markov environment passed the sanity checks and generated randomly timed, persistent hidden-toxicity regimes consistent with the intended switching process.

### Conclusion

> The deterministic experimental schedule can be replaced by a reproducible hidden Markov toxicity process, providing a harder environment for testing the existing estimator.

---

## 3. Experiment 2 — Can the Existing Rolling Estimator Track Stochastic Toxicity?

### Question

The V3b estimator was calibrated under long deterministic regimes. Does the same frozen estimator remain informative when regime changes occur randomly?

The estimator is unchanged:

$$
\hat m_t
=
\text{mean(last 50 filled markouts)},
$$

$$
\hat\phi_t
=
\operatorname{clip}
\left(
\frac{\hat m_t}{\sigma\sqrt{2/\pi}},
0,1
\right).
$$

### Hypothesis

The rolling estimator has finite memory.

Therefore:

1. sufficiently persistent regimes should still be tracked;
2. successful transitions should require approximately

$$
N/2 \approx 25
$$

new fills to cross the midpoint;
3. some short regimes should end before the estimator finishes adapting.

The third outcome is not a bug. It is the expected limitation of a finite-memory estimator in a stochastic environment.

### Design

Estimation is first isolated from inventory effects:

```text
k = 0
N = 50
sigma = 0.3
p = 0.002
n_steps = 4500
n_seeds = 300
```

A fixed quote is used, and `phi_hat` is reconstructed from realized filled markouts.

Primary metrics:

- full-path tracking RMSE;
- RMSE conditional on low/high hidden state;
- response lag among resolved transitions;
- resolution rate:

$$
\frac{\text{switches resolved before next switch}}
{\text{eligible switches}}.
$$

Response lags are pooled across individual transitions so that every resolved transition carries equal weight.

### Results

```text
tracking RMSE (all)   = 0.3888
tracking RMSE (low)   = 0.3900
tracking RMSE (high)  = 0.4235

resolved response lag = 23.0 fills
predicted N/2         = 25 fills

resolution rate       = 0.694
                       = 1334 / 1921 eligible switches

mean fills            = 661.8
```

The resolved response lag of `23.0` fills is close to the pre-specified `N/2 = 25` prediction.

However, only `69.4%` of eligible transitions persist long enough for the estimator to cross the midpoint before the next hidden-state switch.

The full-path RMSE should not be compared directly with the steady-state RMSE reported in V3b. V3b excluded transition periods from its steady-state calculation, whereas V4b deliberately evaluates a path containing repeated stochastic transitions.

### Conclusion

> The frozen `N=50` estimator remains informative under stochastic hidden toxicity. When a regime persists, it responds with the expected roughly `N/2`-fill lag. Its main limitation is now visible: sufficiently short regimes can end before the rolling window completes its adjustment.

The results do not motivate a more complex estimator in the current model.

---

## 4. Experiment 3 — Does the Same Estimator Survive Full Integration?

### Question

V4a showed that inventory skew does not materially contaminate the fill-based toxicity signal under deterministic regimes.

Does that result continue to hold when toxicity itself switches stochastically?

### Hypothesis

Putting the same estimator into the `k=.04` IntegratedMaker should:

1. leave toxicity-tracking quality approximately unchanged;
2. preserve the large reduction in inventory exposure;
3. require no change to `N`, `k`, or the estimator.

### Design

Use the same hidden Markov environment:

```text
sigma = 0.3
p = 0.002
N = 50
n_steps = 4500
n_seeds = 300
```

Compare only:

```text
T  = Toxicity-only, k=0
IT = Integrated, k=.04
```

No new estimator and no parameter tuning are introduced.

Primary metrics:

- toxicity-estimation RMSE;
- resolved transition lag;
- resolution rate;
- RMS inventory;
- fills.

Terminal PnL is reported descriptively, but V4b does not reopen the inventory-economics question studied in V4a.

### Results

```text
                       T            IT
phi RMSE              0.4010        0.4003
resolved lag         22.3          22.4 fills
resolution rate       0.671         0.677
RMS inventory        15.539         3.455
fills               592.7         598.4
mean PnL            626.97        589.94
```

Estimator behavior is essentially unchanged:

$$
0.4010 \rightarrow 0.4003
$$

for RMSE,

$$
22.3 \rightarrow 22.4
$$

fills for response lag, and

$$
0.671 \rightarrow 0.677
$$

for transition-resolution rate.

Meanwhile, RMS inventory falls from

$$
15.539 \rightarrow 3.455,
$$

a reduction of approximately `77.8%`.

The paired PnL comparison is:

$$
IT-T=-37.030,
$$

with

$$
SE=18.485,
$$

and

$$
95\%CI=[-73.261,-0.800].
$$

This difference is reported honestly but is not used to reopen the inventory-cost analysis from V4a. The relevant V4b result is that the estimator and inventory-control mechanisms continue to function normally together.

### Conclusion

> Inventory control does not materially degrade toxicity tracking in the stochastic hidden-state environment. The rolling estimator shows nearly identical RMSE, response lag, and resolution rate with and without inventory skew, while the integrated strategy retains the large reduction in inventory exposure.

Thus the modularity result from V4a survives the harder stochastic environment.

---

## 5. Overall Conclusion

V4b replaces the deterministic toxicity schedule with a genuinely hidden two-state Markov process while keeping the existing `N=50` rolling-markout estimator unchanged.

The estimator remains useful: persistent regimes are tracked with the expected roughly `N/2`-fill response, while short regimes expose its natural finite-memory limitation and can end before adaptation is complete.

Importantly, putting the same estimator back into the inventory-controlled maker causes almost no deterioration in RMSE, response lag, or transition-resolution rate, while preserving the large reduction in inventory exposure.

> **The evidence therefore does not motivate a more complex estimator in the current model.**

V4b establishes that the existing estimator is adequate under stochastic hidden toxicity. The next question is therefore not how to estimate toxicity more elaborately, but how much of that latent-state estimation is actually necessary for the quoting decision.
