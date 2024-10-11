functions {

  real plateau_gaussian_normalising_constant(real delta, real sigma1, real sigma2) {
    real sqrt_2pi = sqrt(2 * pi());
    return (sqrt_2pi * sigma1 / 2) + (sqrt_2pi * sigma2 / 2) + 2 * delta;
  }

  vector plateau_gaussian_log_density(vector y, real mu, real delta, real sigma1, real sigma2) {
    // Precompute repeated values
    real sigma1_sq = square(sigma1);
    real sigma2_sq = square(sigma2);

    // Initialize the log density vector
    vector[rows(y)] log_density;

    // Compute log densities for each element in y
    for (i in 1:rows(y)) {
      if (y[i] < mu - delta) {
        log_density[i] = -0.5 * (square(y[i] - (mu - delta)) / sigma1_sq);
      } else if (y[i] <= mu + delta) {
        log_density[i] = 0;
      } else {  // y > mu + delta
        log_density[i] = -0.5 * (square(y[i] - (mu + delta)) / sigma2_sq);
      }
    }

    real C = plateau_gaussian_normalising_constant(delta, sigma1, sigma2);

    // Return the unnormalised log density
    return log_density - log(C);
  }

}

data {
  int<lower=0> N; // Number of data points
  vector[N] y;      // Observed values
}

parameters {
  real mu;              // Mean for the left distribution
  real<lower=0> delta; // Difference between left mean and right mean
  real<lower=0> sigma1; // Standard deviation for left tail
  real<lower=0> sigma2; // Standard deviation for right tail
}

model {
  // Priors
  mu ~ normal(1300, 100);
  delta ~ normal(0, 300);
  sigma1 ~ exponential(0.5);
  sigma2 ~ exponential(0.5);

  // Vectorized Likelihood
  vector[N] log_density = plateau_gaussian_log_density(y, mu, delta, sigma1, sigma2);
  target += sum(log_density);
}
