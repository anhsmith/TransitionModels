
// The input datdata {
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
  real mu0_global; // global mean for m=0
  real mu1_global; // global mean for m=1
  real<lower=0> tau0; // population level standard deviation for m=0
  real<lower=0> tau1; // population level standard deviation for m=1
}

model {
  // Priors
  mu0_global ~ normal(1200, 100);
  mu1_global ~ normal(1400, 100);
  tau0 ~ normal(0, 50);
  tau1 ~ normal(0, 50);
  sigma ~ normal(0, 50);

  for (j in 1:J) {
    mu0[j] ~ normal(mu0_global, tau0);
    mu1[j] ~ normal(mu1_global, tau1);
  }

  for (i in 1:N) {
    if (m[i] == 0) {
      target += normal_lpdf(x[i] | mu0[pop[i]], sigma) - normal_lccdf(1500 | mu0[pop[i]], sigma) + normal_lcdf(1100 | mu0[pop[i]], sigma);
    } else {
      target += normal_lpdf(x[i] | mu1[pop[i]], sigma) - normal_lccdf(1500 | mu1[pop[i]], sigma) + normal_lcdf(1100 | mu1[pop[i]], sigma);
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
}
