functions {

  real uniform_gaussian_normalising_constant(real delta, real sigma) {
    real sqrt_2pi = sqrt(2 * pi());
    return (sqrt_2pi * sigma / 2) + delta;
  }

  real uniform_left_gaussian_log_density_unnormalised(real y, real mu, real sigma, real L) {

    // Precompute repeated values
    real sigma_sq = square(sigma);

    // Declare the density variable
    real log_density_unnormalised;

    // Conditional structure for the piecewise density function
    if (y <= mu) {
      log_density_unnormalised = 0;
    } else {
      log_density_unnormalised = -0.5 * ( square(y - mu) / sigma_sq );
    }

    // Return the normalised density
    return log_density_unnormalised ;
  }

  real uniform_right_gaussian_log_density_unnormalised(real y, real mu, real sigma, real U) {

    // Precompute repeated values
    real sigma_sq = square(sigma);

    // Declare the density variable
    real log_density_unnormalised;

    // Conditional structure for the piecewise density function
    if (y >= mu) {
      log_density_unnormalised = 0;
    } else {
      log_density_unnormalised = -0.5 * ( square(y - mu) / sigma_sq );
    }

    // Return the normalised density
    return log_density_unnormalised ;
  }

}

data {
  int<lower=0> N; // Number of data points
  real L; // lower truncation
  real U; // upper truncation
  real<lower=L, upper=U> x[N]; // Observed values
  int<lower=0, upper=1> y[N]; // response values
  real<lower=0> prior_m50_mu;
  real<lower=0> prior_m50_tau;
  real<lower=0> prior_d_mu;
  real<lower=0> prior_d_tau;
  real<lower=0> prior_sigma_mu;
  real<lower=0> prior_sigma_tau;
}

parameters {
  real<lower=L,upper=U> m50; // Transition point
  real<lower=0> d;       // Difference between upper immature mean and lower mature mean
  real<lower=0> sigma;  // Standard deviation
}

transformed parameters{
  // Means based on m50
  real<lower=L> mu_0 = m50 - d / 2;
  real<upper=U> mu_1 = m50 + d / 2;
  // Compute the normalisation constants C
  real C_0 = uniform_gaussian_normalising_constant(mu_0-L, sigma);
  real C_1 = uniform_gaussian_normalising_constant(U-mu_1, sigma);
}

model {
  // Priors
  m50 ~ normal(prior_m50_mu, prior_m50_tau); // prior_m50_mu 1300; prior_m50_tau 100;
  d ~ normal(prior_d_mu, prior_d_tau);    // prior_d_mu 0; prior_d_tau 100;
  sigma ~ normal(prior_sigma_mu, prior_sigma_tau);  // prior_sigma_mu 0; prior_sigma_tau 100;

  // Likelihood
  for (i in 1:N) {
    if (y[i] == 0) {
      target += uniform_left_gaussian_log_density_unnormalised( x[i], mu_0, sigma, L ) - log(C_0);
      } else {
      target += uniform_right_gaussian_log_density_unnormalised( x[i], mu_1, sigma, U ) - log(C_1);
      }
    }
}

