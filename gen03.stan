data {
  int<lower=0> N;         // Number of observations
  vector[N] x;            // Continuous predictor
  int<lower=0, upper=1> y[N]; // Binary outcome
  real<lower=0> k;        // Scaling factor for the distance between means
}

parameters {
  real m50;               // Median transition point
  real<lower=0> sigma_x;  // Standard deviation of x for both classes
}

transformed parameters {
  real mu0;
  real mu1;

  // Means based on m50 and sigma_x
  mu0 = m50 - k * sigma_x;
  mu1 = m50 + k * sigma_x;
}

model {
  // Priors
  m50 ~ normal(1300, 100);
  sigma_x ~ normal(0, 100);

  // Likelihood
  for (n in 1:N) {
    if (y[n] == 0) {
      x[n] ~ normal(mu0, sigma_x);
    } else {
      x[n] ~ normal(mu1, sigma_x);
    }
    y[n] ~ bernoulli(Phi((x[n] - m50) / sigma_x));
  }
}

generated quantities {
  real log_lik[N];
  for (n in 1:N)
    log_lik[n] = bernoulli_lpmf(y[n] | Phi((x[n] - m50) / sigma_x));
}
