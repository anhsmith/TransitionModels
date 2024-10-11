// functions {
//   real plateau_gaussian_density(real y, real mu, real d, real sigma1, real sigma2) {
//
//     // Compute the normalization constant C
//     real C = (sqrt(2 * pi() * pow(sigma1, 2)) / 2) + (sqrt(2 * pi() * pow(sigma2, 2)) / 2) + 2 * d;
//
//     // Declare the density variable
//     real density;
//
//     // Conditional structure for the piecewise density function
//     if (y < mu - d) {
//       density = exp( (-1 / (2 * pow(sigma1, 2))) * pow( y - (mu - d), 2) );
//     } else if (y >= mu - d && y <= mu + d) {
//       density = 1;
//     } else {  // y > mu + d
//       density = exp( (-1 / (2 * pow(sigma2, 2))) * pow( y - (mu + d), 2) );
//     }
//
//     // Return the normalized density
//     return density / C;
//   }
// }
//
// The following is twice as fast...
//
// real plateau_gaussian_density_unnormalised(real y, real mu, real delta, real sigma1, real sigma2) {
//
//   // Precompute repeated values
//   real sigma1_sq = square(sigma1);
//   real sigma2_sq = square(sigma2);
//
//
//   // Declare the density variable
//   real density;
//
//   // Conditional structure for the piecewise density function
//   if (y < mu - delta) {
//     density = exp(-0.5 * (square(y - (mu - delta)) / sigma1_sq));
//   } else if (y <= mu + delta) {
//     density = 1;
//   } else {  // y > mu + delta
//     density = exp(-0.5 * (square(y - (mu + delta)) / sigma2_sq));
//   }
//
//   // Return the unnormalised density
//   return density;
// }

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
  real y[N];      // Observed values
}

parameters {
  real mu;              // Mean for the left distribution
  real<lower=0> delta;      // Difference between left mean and right mean
  real<lower=0> sigma1; // Standard deviation for left tail
  real<lower=0> sigma2; // Standard deviation for right tail
}

transformed parameters{
  // Compute the normalization constant C
  real C = plateau_gaussian_normalising_constant(delta, sigma1, sigma2);
}

model {
  // Priors
  mu ~ normal(1300, 100);
  delta ~ normal(0, 300);
  sigma1 ~ exponential(0.5);
  sigma2 ~ exponential(0.5);

  // Likelihood
  for (i in 1:N) {
    target += plateau_gaussian_log_density_unnormalised( y[i], mu, delta, sigma1, sigma2 ) - log(C);
  }
}
