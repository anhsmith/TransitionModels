functions {

  real plateau_gaussian_normalising_constant(real delta, real sigma1, real sigma2) {
    real sqrt_2pi = sqrt(2 * pi());
    return (sqrt_2pi * sigma1 / 2) + (sqrt_2pi * sigma2 / 2) + 2 * delta;
  }

  real plateau_gaussian_log_density_unnormalised(real y, real mu, real delta, real sigma1, real sigma2) {

    // Precompute repeated values
    real sigma1_sq = square(sigma1);
    real sigma2_sq = square(sigma2);

    // Declare the density variable
    real log_density_unnormalised;

    // Conditional structure for the piecewise density function
    if (y < mu - delta) {
      log_density_unnormalised = -0.5 * ( square(y - (mu - delta)) / sigma1_sq );
    } else if (y <= mu + delta) {
      log_density_unnormalised = 0;
    } else {  // y > mu + delta
      log_density_unnormalised = -0.5 * ( square(y - (mu + delta)) / sigma2_sq );
    }

    real C = plateau_gaussian_normalising_constant(delta, sigma1, sigma2);

    // Return the normalised density
    return log_density_unnormalised ;
  }

}

data {
  int<lower=0> N; // Number of data points
  real x[N];      // Observed values
  int<lower=0, upper=1> y[N]; // response values
}

parameters {
  real m50;              // Transition point

  real<lower=0> d;       // Difference between upper immature mean and lower mature mean

  real<lower=0> delta_0; // Difference between left mean and right mean
  real<lower=0> delta_1;

  real<lower=0> sigma_lo_0;  // Standard deviation for left tail of immatures
  real<lower=0> sigma_up_0;  // Standard deviation for right tail of immatures
  // real<lower=0> sigma_lo_1;  // Same as sigma_up_0
  real<lower=0> sigma_up_1;  // Set as same as sigma_lo_1
}

transformed parameters{

  real<lower=0> sigma_lo_1 = sigma_up_0;

  real middle_0;
  real middle_1;

  real mu_up_0;
  real mu_up_1;
  real mu_lo_0;
  real mu_lo_1;

  // Means based on m50 and sigma_x
  mu_up_0 = m50 - d / 2;
  middle_0 = mu_up_0 - delta_0 / 2;
  mu_lo_0 = mu_up_0 - delta_0;

  mu_lo_1 = m50 + d / 2;
  middle_1 = mu_lo_1 + delta_1 / 2;
  mu_up_1 = mu_lo_1 + delta_1;

  // Compute the normalization constant C
  real C_0 = plateau_gaussian_normalising_constant(delta_0, sigma_lo_0, sigma_up_0);
  real C_1 = plateau_gaussian_normalising_constant(delta_1, sigma_lo_1, sigma_up_1);

}

model {
  // Priors
  m50 ~ normal(1300, 100);

  d ~ normal(100, 100);

  delta_0 ~ normal(200, 100);
  delta_1 ~ normal(200, 100);

  sigma_lo_0 ~ normal(100,50);
  sigma_up_1 ~ normal(100,50);
  sigma_up_0 ~ normal(100,50);

  // Likelihood
  for (i in 1:N) {
    if (y[i] == 0) {
      target += plateau_gaussian_log_density_unnormalised( x[i], middle_0, delta_0, sigma_lo_0, sigma_up_0 ) - log(C_0);
      } else {
      target += plateau_gaussian_log_density_unnormalised( x[i], middle_1, delta_1, sigma_lo_1, sigma_up_1 ) - log(C_1);
      }
    }
}
