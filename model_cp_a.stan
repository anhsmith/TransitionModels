data {
  int<lower=0> N; // Number of data points
  real x[N]; // Values of x
  int<lower=0,upper=1> y[N]; // Values of y (0 or 1)
  real mu_guess; // Initial guess for mu
  real<lower=0> sd_mu;
}

parameters {
  real<lower=min(x), upper=max(x)> mu; // Change-point threshold
  real<lower=0, upper=1> p; // Probability of y being 1
}

model {
  mu ~ normal(mu_guess, sd_mu); // Prior for mu
  p ~ beta(1, 1); // Prior for p

  for (i in 1:N) {
    if (x[i] <= mu)
      y[i] ~ bernoulli(1 - p); // y follows Bernoulli distribution with probability 1-p
    else
      y[i] ~ bernoulli(p); // y follows Bernoulli distribution with probability p
  }
}

generated quantities {
  int y_pred[N]; // Predicted values of y

  for (i in 1:N) {
    if (x[i] <= mu)
      y_pred[i] = bernoulli_rng(1 - p); // Generate predicted value of y based on probability 1-p
    else
      y_pred[i] = bernoulli_rng(p); // Generate predicted value of y based on probability p
  }
}
