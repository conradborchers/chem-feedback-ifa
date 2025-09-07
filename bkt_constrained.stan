data {
  int<lower=1> K;                              // number of sequences for this skill
  int<lower=1> N;                              // total length across sequences
  array[K] int<lower=1> T;                     // lengths per sequence
  array[N] int<lower=0, upper=1> y;            // concatenated 0/1 correctness
}
parameters {
  real<lower=0, upper=1> prior_raw;
  real<lower=0, upper=1> learn_raw;
  real<lower=0, upper=1> g_raw;
  real<lower=0, upper=1> s_raw;
}
transformed parameters {
  // Map to constrained ranges with 0.01 floor
  real prior = 0.01 + prior_raw * (0.90 - 0.01);   // [0.01, 0.90]
  real learn = 0.01 + learn_raw * (0.90 - 0.01);   // [0.01, 0.90]
  real guess = 0.01 + g_raw   * (0.50 - 0.01);     // [0.01, 0.50]

  // enforce guess + slip ≤ 1
  real s_cap = fmin(0.50, 1.0 - guess - 1e-6);
  real slip  = 0.01 + s_raw   * (s_cap - 0.01);    // [0.01, s_cap]
}
model {
  // weakly-informative priors
  prior_raw ~ beta(2, 2);
  learn_raw ~ beta(2, 2);
  g_raw     ~ beta(2, 8);
  s_raw     ~ beta(2, 8);

  // forward algorithm over K sequences (2-state BKT; forget=0)
  {
    real p_c_known   = 1.0 - slip;   // P(y=1 | known)
    real p_c_unknown = guess;        // P(y=1 | not-known)
    int pos = 1;

    for (k in 1:K) {
      vector[2] alpha;
      alpha[1] = log1m(prior);   // not-known
      alpha[2] = log(prior);     // known

      for (t in 1:T[k]) {
        int yt = y[pos];

        real ll0 = yt == 1 ? log(p_c_unknown) : log1m(p_c_unknown);
        real ll1 = yt == 1 ? log(p_c_known)   : log1m(p_c_known);

        alpha[1] += ll0;
        alpha[2] += ll1;

        real next0 = log_sum_exp(alpha[1] + log1m(learn),
                                 negative_infinity());
        real next1 = log_sum_exp(alpha[1] + log(learn),
                                 alpha[2]);

        alpha[1] = next0;
        alpha[2] = next1;
        pos += 1;
      }
      target += log_sum_exp(alpha);
    }
  }
}

