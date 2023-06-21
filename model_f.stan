
data {
  int<lower=0> N;                    // Number of data points
  vector[N] x;                       // Predictor variable
  int<lower=0, upper=1> y[N];         // Binary response variable (0 or 1)
  real<lower=0> prior_tr_mu;
  real<lower=0> prior_tr_sd;
  // real<lower=0> prior_sigma;
}

parameters {
  real<lower=0> transition;           // Transition point
  real<lower=0> sigma;                // Scale parameter
}

model {
  // Priors
  transition ~ normal(prior_tr_mu, prior_tr_sd); // Prior for the transition point
  sigma ~ normal(20, 5);     // Prior for the scale parameter

  // Likelihood
  for (n in 1:N) {
    real eta = (x[n] - transition) / sigma;
    if (y[n] == 0)
      target += log1m(Phi_approx(eta));      // Likelihood for y=0
    else
      target += log(Phi_approx(eta));    // Likelihood for y=1
  }
}
