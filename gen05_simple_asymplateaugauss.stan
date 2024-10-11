functions {
  real plateau_gaussian_density(real y, real mu, real delta, real sigma1, real sigma2) {

    // Precompute repeated values
    real sigma1_sq = square(sigma1);
    real sigma2_sq = square(sigma2);
    real sqrt_2pi = sqrt(2 * pi());

    // Compute the normalization constant C
    real C = (sqrt_2pi * sigma1 / 2) + (sqrt_2pi * sigma2 / 2) + 2 * delta;

    // Declare the density variable
    real density;

    // Conditional structure for the piecewise density function
    if (y < mu - delta) {
      density = exp(-0.5 * (square(y - (mu - delta)) / sigma1_sq));
    } else if (y >= mu - delta && y <= mu + delta) {
      density = 1;
    } else {  // y > mu + delta
      density = exp(-0.5 * (square(y - (mu + delta)) / sigma2_sq));
    }

    // Return the normalized density
    return density / C;
  }
}

data {
  int<lower=0> N; // number of data points
  real<lower=0> x[N]; // predictor values
  int<lower=0, upper=1> y[N]; // response values
}

parameters {
  real m50;
  real<lower=0> d;
  real<lower=0> sigma; // common standard deviation
}

transformed parameters {
  real mu0;
  real mu1;

  // Means based on m50 and sigma_x
  mu0 = m50 - d / 2;
  mu1 = m50 + d / 2;
}

model {
  m50 ~ normal(1300, 100);
  d ~ normal(200, 50);
  sigma ~ normal(0, 50);

  for (i in 1:N) {
    if (y[i] == 0) {
      target += normal_lpdf(x[i] | mu0, sigma);
    } else {
      target += normal_lpdf(x[i] | mu1, sigma);
    }
  }
}
//
//
// generated quantities {
//
//   // Calculate posterior probabilities
//   real<lower=0, upper=1> p_m0_given_x[N];
//   real<lower=0, upper=1> p_m1_given_x[N];
//   for (i in 1:N) {
//     real p_x_given_m0 = exp(normal_lpdf(x[i] | mu0, sigma));
//     real p_x_given_m1 = exp(normal_lpdf(x[i] | mu1, sigma));
//     real denom = p_x_given_m0 + p_x_given_m1;
//     p_m0_given_x[i] = p_x_given_m0 / denom;
//     p_m1_given_x[i] = p_x_given_m1 / denom;
//   }
// }
