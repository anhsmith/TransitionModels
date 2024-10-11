data {
  int<lower=0> N;          // number of data points
  real<lower=1100, upper=1500> x[N]; // predictor values
  int<lower=0, upper=1> m[N];  // response values
  int<lower=1> J;          // number of populations
  int<lower=1, upper=J> pop[N]; // population index for each data point
}

parameters {
  real mu0[J];  // means for m=0 for each population
  real mu1[J];  // means for m=1 for each population
  real<lower=0> sigma; // common standard deviation
  real<lower=1000, upper=1600> mu0_global; // global mean for m=0
  real<lower=1200, upper=1800> mu1_global; // global mean for m=1
  real<lower=0> tau; // common population level standard deviation
}

model {
  // Priors
  mu0_global ~ normal(1300, 100);
  mu1_global ~ normal(1300, 100);
  tau ~ normal(0, 50);
  sigma ~ normal(0, 50);

  for (j in 1:J) {
    mu0[j] ~ normal(mu0_global, tau);
    mu1[j] ~ normal(mu1_global, tau);
  }

  for (i in 1:N) {
    if (m[i] == 0) {
      x[i] ~ normal(mu0[pop[i]], sigma) T[1100, 1500];
    } else {
      x[i] ~ normal(mu1[pop[i]], sigma) T[1100, 1500];
    }
  }
}

generated quantities {
  real m50[J];
  real global_m50;
  real sum_m50 = 0;

  for (j in 1:J) {
    m50[j] = (mu0[j] + mu1[j]) / 2;
    sum_m50 += m50[j];
  }
  global_m50 = sum_m50 / J;

  // Calculate posterior probabilities
  real p_m0_given_x[N];
  real p_m1_given_x[N];
  for (i in 1:N) {
    real p_x_given_m0 = exp(normal_lpdf(x[i] | mu0[pop[i]], sigma));
    real p_x_given_m1 = exp(normal_lpdf(x[i] | mu1[pop[i]], sigma));
    real denom = p_x_given_m0 + p_x_given_m1;
    p_m0_given_x[i] = p_x_given_m0 / denom;
    p_m1_given_x[i] = p_x_given_m1 / denom;
  }
}
