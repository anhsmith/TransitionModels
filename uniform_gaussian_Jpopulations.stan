functions {

  real uniform_gaussian_normalising_constant(real delta, real sigma) {
    real sqrt_2pi = sqrt(2 * pi());
    return (sqrt_2pi * sigma / 2) + delta;
  }

  real uniform_left_gaussian_log_density_unnormalised(real y, real mu, real sigma, real L) {

    // Declare the density variable
    real log_density_unnormalised;

    // Conditional structure for the piecewise density function
    if (y <= mu) {
      log_density_unnormalised = 0;
    } else {
      log_density_unnormalised = -0.5 * ( square(y - mu) / square(sigma) );
    }

    // Return the normalised density
    return log_density_unnormalised ;
  }

  real uniform_right_gaussian_log_density_unnormalised(real y, real mu, real sigma, real U) {

    // Declare the density variable
    real log_density_unnormalised;

    // Conditional structure for the piecewise density function
    if (y >= mu) {
      log_density_unnormalised = 0;
    } else {
      log_density_unnormalised = -0.5 * ( square(y - mu) / square(sigma) );
    }

    // Return the normalised density
    return log_density_unnormalised ;
  }

}

data {
  int<lower=0> N; // Number of data points
  int<lower=1> J; // number of populations
  real L; // lower truncation
  real U; // upper truncation
  real<lower=L, upper=U> x[N]; // Observed values: lengths
  int<lower=0, upper=1> y[N]; // Response: 0 = immature, 1 = mature
  int<lower=1, upper=J> pop[N]; // population index for each data point
  real<lower=0> prior_mu_m50_mu;
  real<lower=0> prior_mu_m50_tau;
  real<lower=0> prior_d_mu;
  real<lower=0> prior_d_tau;
  real<lower=0> prior_sigma_x_tau;
  real<lower=0> prior_sigma_alpha_tau;
}

parameters {
  real<lower=L,upper=U> mu_m50; // Mean transition point
  real<lower=0> d;       // Difference between upper immature mean and lower mature mean
  real z[J]; // z scores of population effects on m50_global (non-centred parameterisation)
  real<lower=0> sigma_x;  // Standard deviation of lengths (variation in transition)
  real<lower=0> sigma_alpha;  // Standard deviation of population transition points
}

transformed parameters{

  real alpha[J];
  real m50_pop[J];
  real mu0_pop[J];
  real mu1_pop[J];
  real C0_pop[J];
  real C1_pop[J];

  for(j in 1:J) {

    alpha[j] = z[j] * sigma_alpha;  // non-centred parameterisation
    m50_pop[j] = mu_m50 + alpha[j];
    mu0_pop[j] = m50_pop[j] - d / 2;
    mu1_pop[j] = m50_pop[j] + d / 2;

    // Compute the normalisation constants C
    C0_pop[j] = uniform_gaussian_normalising_constant(mu0_pop[j]-L, sigma_x);
    C1_pop[j] = uniform_gaussian_normalising_constant(U-mu1_pop[j], sigma_x);
  }

}

model {
  // Priors
  mu_m50 ~ normal(prior_mu_m50_mu, prior_mu_m50_tau); // 1300, 50;
  d ~ normal(prior_d_mu, prior_d_tau);    // prior_d_mu 0; prior_d_tau 100;
  sigma_x ~ normal(0, prior_sigma_x_tau);  // prior_sigma_mu 0; prior_sigma_tau 100;
  z ~ normal(0, 1);
  sigma_alpha ~ normal(0, prior_sigma_alpha_tau); // 0, 50

  // Likelihood
  for (i in 1:N) {
    if (y[i] == 0) {
      target += uniform_left_gaussian_log_density_unnormalised( x[i], mu0_pop[pop[i]], sigma_x, L ) - log(C0_pop[pop[i]]);
      } else {
      target += uniform_right_gaussian_log_density_unnormalised( x[i], mu1_pop[pop[i]], sigma_x, U ) - log(C1_pop[pop[i]]);
      }
    }
}


