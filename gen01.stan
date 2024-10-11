data {
  int<lower=0> N; // number of data points
  real<lower=1100, upper=1500> x[N]; // predictor values
  int<lower=0, upper=1> m[N]; // response values
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
    if (m[i] == 0) {
      target += normal_lpdf(x[i] | mu0, sigma);
    } else {
      target += normal_lpdf(x[i] | mu1, sigma);
    }
  }
}

