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
