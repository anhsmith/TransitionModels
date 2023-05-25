// In this modified model, we no longer include separate parameters for the
// means of the two groups. Instead, we have a single transition point mu that 
// represents the location where the transition from 0 to 1 occurs.

// We use the normal_lcdf function to calculate the cumulative distribution 
// function (CDF) of the standard normal distribution for class 0 (0 to 1 transition). 
// Similarly, we use the normal_lccdf function to calculate the complementary 
// cumulative distribution function (CCDF) for class 1 (1 to 0 transition).

// The log_sum_exp function is used to calculate the log-sum of the two terms 
// representing the probabilities of the two transitions, weighted by the quantile 
// parameter theta.

// The remaining structure of the model, including the prior specification, 
// remains the same as the previous version.

data {
  int<lower=0> N;                    // Number of data points
  vector[N] X;                       // Predictor variable
  int<lower=0, upper=1> y[N];         // Binary response variable (0 or 1)
}

parameters {
  real<lower=0, upper=1> theta;       // Quantile parameter
  real mu;                           // Transition point
  real<lower=0> sigma;                // Scale parameter
}

model {
  // Priors
  theta ~ beta(1, 1);                 // Prior for quantile parameter
  mu ~ normal(0, 10);                 // Prior for the transition point
  sigma ~ cauchy(0, 5);               // Prior for the scale parameter
  
  // Likelihood
  for (n in 1:N) {
    real eta = (X[n] - mu) / sigma;
    target += log_sum_exp(
      log1m(theta) + normal_lcdf(0 | eta, 1),
      log(theta) + normal_lccdf(0 | eta, 1)
    );
  }
}
