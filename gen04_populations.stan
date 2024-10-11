data {
  int<lower=0> N; // number of data points
  real<lower=0> x[N]; // predictor values
  int<lower=0, upper=1> y[N]; // response values
  int<lower=1> J;          // number of populations
  int<lower=1, upper=J> pop[N]; // population index for each data point

  int<lower=0> prior_mu_m50_mu;
  int<lower=0> prior_mu_m50_sigma;

  int<lower=0> prior_d_mu;
  int<lower=0> prior_d_sigma;

  int<lower=0> prior_sigma_x_sigma;
  int<lower=0> prior_sigma_alpha_sigma;
}

parameters {
  real mu_m50;
  real<lower=0> d;
  real<lower=0> sigma_x; // common standard deviation
  real z[J]; // z scores of population effects on m50_global
  real<lower=0> sigma_alpha; // standard deviation of population effects
}

transformed parameters {

  real alpha[J];
  real m50_pop[J];
  real mu0_pop[J];
  real mu1_pop[J];


  for(j in 1:J) {
    alpha[j] = z[j] * sigma_alpha;
    m50_pop[j] = mu_m50 + alpha[j];
    mu0_pop[j] = m50_pop[j] - d / 2;
    mu1_pop[j] = m50_pop[j] + d / 2;
  }

}

model {
  mu_m50 ~ normal(prior_mu_m50_mu, prior_mu_m50_sigma); // 1300, 100;
  d ~ normal(prior_d_mu, prior_d_sigma); // 200, 50
  sigma_x ~ normal(0, prior_sigma_x_sigma); // 0, 50
  z ~ normal(0, 1);
  sigma_alpha ~ normal(0, prior_sigma_alpha_sigma); // 0, 50

  for (i in 1:N) {
    if (y[i] == 0) {
      target += normal_lpdf(x[i] | mu0_pop[pop[i]], sigma_x);
    } else {
      target += normal_lpdf(x[i] | mu1_pop[pop[i]], sigma_x);
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
