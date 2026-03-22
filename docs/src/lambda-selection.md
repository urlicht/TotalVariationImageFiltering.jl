# Lambda Selection

`TotalVariationImageFiltering.jl` includes two ROF-specific parameter selection tools:

- `select_lambda_discrepancy`
- `select_lambda_sure`

Both operate on `N`-D arrays (for example, 2D images and 3D volumes).

## 1) Discrepancy Principle

`select_lambda_discrepancy` finds `lambda` such that:

```math
\|u_\lambda - f\|_2^2 \approx \text{target\_scale}\cdot N \sigma^2,
```

where `N = length(f)`.

Search strategy in code:

1. evaluate at `lambda_min` and `lambda_max`;
2. expand upper bracket if needed;
3. bisection (`max_bisect`) on the bracket.

Example:

```julia
using TotalVariationImageFiltering

selection = TotalVariationImageFiltering.select_lambda_discrepancy(
    noisy,
    TotalVariationImageFiltering.ROFConfig();
    sigma = 0.12,
    lambda_min = 0.0,
    lambda_max = 0.2,
    rtol = 0.05,
)

lambda_hat = selection.lambda
u = selection.u
```

Returned diagnostics include residual norm, target norm, mismatch, evaluations,
and final search bracket.

## 2) Monte-Carlo SURE

`select_lambda_sure` minimizes SURE on a provided grid:

```math
\mathrm{SURE}(\lambda)
= -N\sigma^2 + \|u_\lambda-f\|_2^2 + 2\sigma^2\,\operatorname{div}(u_\lambda(f)).
```

Divergence is estimated with Monte-Carlo finite differences:

```math
\operatorname{div}(u_\lambda(f))
\approx \frac{1}{\epsilon} b^\top\!\left(u_\lambda(f+\epsilon b)-u_\lambda(f)\right),
\quad b \sim \mathcal{N}(0,I).
```

Example:

```julia
using Random
using TotalVariationImageFiltering

selection = TotalVariationImageFiltering.select_lambda_sure(
    noisy,
    TotalVariationImageFiltering.ROFConfig();
    sigma = 0.12,
    lambda_grid = [0.0, 0.03, 0.06, 0.1, 0.16],
    mc_samples = 2,
    rng = MersenneTwister(1),
)

lambda_hat = selection.lambda
u = selection.u
```

Returned diagnostics include selected SURE, per-grid SURE values, residuals,
divergence estimates, `epsilon`, and total number of solves.

## Practical Notes

- Both selectors are implemented for the ROF path (`ROFConfig`).
- `warm_start=true` reuses solver state between lambda evaluations.
- Smaller `rtol` (discrepancy) and larger `mc_samples` (SURE) increase runtime.

## References

1. V. A. Morozov, *Methods for Solving Incorrectly Posed Problems*, 1984. [DOI:10.1007/978-1-4612-5280-1](https://doi.org/10.1007/978-1-4612-5280-1)
2. Y. Wen and R. H. Chan, "Parameter selection for total-variation based image restoration using discrepancy principle," *IEEE TIP* 21(4):1770-1781, 2012. [DOI:10.1109/TIP.2011.2181401](https://doi.org/10.1109/TIP.2011.2181401)
3. S. Ramani, T. Blu, M. Unser, "Monte-Carlo SURE: A black-box optimization of regularization parameters for general denoising algorithms," *IEEE TIP* 17(9):1540-1554, 2008. [DOI:10.1109/TIP.2008.2001404](https://doi.org/10.1109/TIP.2008.2001404)
4. Y. Lin, B. Wohlberg, H. Guo, "UPRE method for total variation parameter selection," *Signal Processing* 90(8):2546-2551, 2010. [DOI:10.1016/j.sigpro.2010.02.025](https://doi.org/10.1016/j.sigpro.2010.02.025)
5. C.-A. Deledalle et al., "Stein Unbiased GrAdient estimator of the Risk (SUGAR) for multiple parameter selection," 2014. [HAL:hal-00987295](https://hal.science/hal-00987295)
