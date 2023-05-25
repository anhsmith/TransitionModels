// In this updated model, we no longer include the transition probability p as a parameter. 
// Instead, we model the likelihood directly based on the cumulative probit using 
// the standard normal cumulative distribution function Phi.

// In the likelihood calculation, if y[n] is 0, we use log1m(Phi(eta)), 
// which corresponds to the probability of staying at 0. 
// If y[n] is 1, we use log(1 - Phi(eta)), which corresponds to the probability 
// of transitioning from 0 to 1.
// By directly using Phi(eta) and its complement 1 - Phi(eta) in the 
// likelihood calculation, we avoid sensitivity to unbalanced samples since the 
// model does not explicitly estimate the class proportions or transition probability.


data {
  int<lower=1> N;                // Number of individuals
  int<lower=0, upper=1> y[N];    // Binary labels (0 or 1) indicating the transition
  vector[N] X;                   // Values of X
  int<lower=0> n_0;              // Sample size of 0s
  int<lower=0> n_1;              // Sample size of 1s
}

parameters {
  real<lower=min(X), upper=max(X)> threshold;  // Threshold value
}

model {
  // Priors
  threshold ~ <prior_distribution>;  // Prior distribution for the threshold
  
  // Likelihood
  for (n in 1:N) {
    if (y[n] == 0)
      target += log1m(Phi((X[n] - threshold)));  // Contribution from 0s
      else
      target += log(Phi((X[n] - threshold)));    // Contribution from 1s
  }
  
  // Adjust for different sample sizes
  target += n_0 * log1m(Phi(-threshold)) + n_1 * log(Phi(threshold));
}


