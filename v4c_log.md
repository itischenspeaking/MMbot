# V4c — From Estimation to Decision

## Motivation

V3b and V4b expressed adaptive quoting through a latent toxicity estimate:

$$
\hat{\phi}_t
=
\operatorname{clip}
\left(
\frac{\hat{m}_t}{\sigma\sqrt{2/\pi}},
0,1
\right),
$$

followed by

$$
h_t
=
\frac{1}{\kappa}
+
\hat{\phi}_t \sigma\sqrt{2/\pi},
$$

where \(\hat m_t\) is the rolling estimate of adverse markout per fill.

Substituting the estimator into the quoting rule gives

$$
h_t
=
\frac{1}{\kappa}
+
\operatorname{clip}
\left(
\hat m_t,
0,
\sigma\sqrt{2/\pi}
\right).
$$

The normalization by \(\sigma\) therefore cancels when estimation is converted back into a decision.

This motivates the V4c question:

> **Is the latent toxicity parameter \(\phi\) actually needed for quoting, or can the maker act directly on estimated adverse markout?**

V4c keeps the existing estimator and market-making architecture fixed:

```text
N = 50
k = 0.04
kappa = 1
h_warmup = 1.08
Markov p = 0.002
```

No new estimator or parameter tuning is introduced.

---

## Experiment 1 — Decision Equivalence

### Question

Is the existing \(\phi\)-based quoting policy exactly equivalent to a direct-markout policy with the same clipping rule?

### Hypothesis

Given

$$
\hat{\phi}_t
=
\operatorname{clip}
\left(
\frac{\hat m_t}{\sigma\sqrt{2/\pi}},
0,1
\right),
$$

the existing policy

$$
h_t
=
\frac{1}{\kappa}
+
\hat{\phi}_t\sigma\sqrt{2/\pi}
$$

should be exactly equivalent to

$$
\boxed{
h_t
=
\frac{1}{\kappa}
+
\operatorname{clip}
\left(
\hat m_t,
0,
\sigma\sqrt{2/\pi}
\right)
}
$$

provided that both implementations share the same rolling \(N=50\) markout history, warm-up rule, causal update timing, and clipping.

### Design

A direct-markout implementation was added using

$$
h_t
=
\frac{1}{\kappa}
+
\operatorname{clip}
\left(
\hat m_t,
0,
\sigma\sqrt{2/\pi}
\right).
$$

Same-seed regression checks compared it against the existing \(\phi\)-based policy on:

- quotes,
- fills,
- inventory,
- terminal PnL.

This was an implementation-equivalence test rather than an economic Monte Carlo experiment.

### Results

The two implementations matched exactly under the regression checks.

### Conclusion

The latent-\(\phi\) parameterization is useful for interpreting the hidden environment, but it is not mathematically necessary for the quoting decision.

The existing adaptive policy can be written directly in terms of estimated adverse markout.

The only remaining structural difference is the upper cap implied by

$$
\phi \le 1,
$$

which becomes

$$
\hat m_t
\le
\sigma\sqrt{2/\pi}.
$$

---

## Experiment 2 — Is the Upper Cap Economically Necessary?

### Question

Once the explicit \(\phi\) parameterization is removed, does the inherited upper cap materially affect economic performance?

The capped policy is

$$
h_t
=
\frac{1}{\kappa}
+
\operatorname{clip}
\left(
\hat m_t,
0,
\sigma\sqrt{2/\pi}
\right),
$$

while the uncapped direct policy is

$$
\boxed{
h_t
=
\frac{1}{\kappa}
+
\max(\hat m_t,0)
}
$$

The only difference is removal of the upper cap.

### Hypothesis

No directional PnL improvement was assumed in advance.

The pre-specified expectations were:

1. if the cap rarely binds, the two policies should behave almost identically;
2. if the cap binds frequently, the uncapped policy should quote slightly wider when estimated markout is unusually high;
3. any economic difference must come from the upper clipping itself, since the \(\sigma\)-normalization has already cancelled algebraically.

### Design

The experiment used the stochastic hidden-toxicity environment from V4b:

```text
sigma = 0.3
p = 0.002
N = 50
k = 0.04
n_steps = 4500
n_seeds = 1000
```

The two otherwise identical integrated policies were compared under same-seed common random numbers.

Primary metrics were:

- mean terminal PnL,
- paired Direct − Capped PnL difference with SE and 95% CI,
- RMS inventory,
- mean half-spread,
- upper-cap binding rate.

The upper-cap binding rate was defined as the fraction of post-warmup **decision ticks** for which

$$
\hat m_t
>
\sigma\sqrt{2/\pi}.
$$

At the baseline volatility,

$$
\sigma\sqrt{2/\pi}
=
0.2394.
$$

### Results

The upper cap bound on

$$
13.63\%
$$

of post-warmup decision ticks.

```text
             capped     direct
mean PnL      586.57     586.46
RMS inventory   3.454      3.454
mean h           1.1170     1.1196
```

Removing the cap increased average half-spread only slightly:

$$
1.1196 - 1.1170
=
0.0026.
$$

The paired terminal-PnL difference was

$$
Direct-Capped
=
-0.108,
$$

with

$$
SE
=
0.150,
$$

and

$$
95\%CI
=
[-0.401,\,+0.186].
$$

No economically detectable PnL difference was resolved, and inventory exposure was effectively unchanged.

### Conclusion

The result is stronger than simply saying that the cap rarely binds.

The structural upper cap is active on a nontrivial fraction of decision ticks, yet removing it has almost no effect on average spread, inventory exposure, or PnL at the baseline parameters.

The simpler direct-markout rule is therefore sufficient:

$$
\boxed{
h_t
=
\frac{1}{\kappa}
+
\max(\hat m_t,0)
}
$$

The reason for preferring it is not higher PnL. The experiment does not support such a claim.

The advantage is simplification: the final policy no longer needs either

1. normalization into a latent \(\phi\), or
2. the upper cap inherited from the structural assumption \(\phi \le 1\).

---

## Conclusion

V4c simplifies the adaptive decision rule without materially changing its economic behavior.

The latent toxicity parameter \(\phi\) remains useful for describing the simulated environment, but it is unnecessary for the quoting decision itself. The original \(\phi\)-based rule is exactly equivalent to a clipped direct-markout rule because the normalization by \(\sigma\) cancels algebraically.

Removing the remaining structural upper cap also produces almost no economic change. Despite binding on approximately 13.6% of post-warmup decision ticks, the uncapped policy has nearly identical average spread, inventory exposure, and PnL.

The final quoting rule is therefore

$$
\boxed{
h_t
=
\frac{1}{\kappa}
+
\max(\hat m_t,0)
}
$$

which acts directly on estimated adverse-selection cost.

> **Estimate the economic quantity that matters, then act directly on it.**

V4c therefore closes the model-design stage. The remaining question is whether this frozen policy generalizes beyond the environments and seeds used during development, which is the focus of V4d.
