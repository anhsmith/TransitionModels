# Precompute the constant for uniform Gaussian density
c_uniform_gaussian_density <- function(delta, sigma) {
  sqrt_2pi <- sqrt(2 * pi)
  return((sqrt_2pi * sigma / 2) + delta)
}

# Vectorized left Gaussian density function
uniform_left_gaussian_density <- function(y, mu, sigma, L) {
  # Precompute repeated values
  sigma_sq <- sigma^2
  C <- c_uniform_gaussian_density(abs(mu - L), sigma)

  # Initialize density vector
  density <- numeric(length(y))

  # Vectorized conditions
  below_L <- y < L
  below_mu <- (y >= L) & (y < mu)
  above_mu <- y >= mu

  # Assign density values based on conditions
  density[below_L] <- 0
  density[below_mu] <- 1
  density[above_mu] <- exp(-0.5 * ((y[above_mu] - mu)^2 / sigma_sq))

  # Normalize density
  density <- density / C

  return(density)
}

# Vectorized right Gaussian density function
uniform_right_gaussian_density <- function(y, mu, sigma, U) {
  # Precompute repeated values
  sigma_sq <- sigma^2
  C <- c_uniform_gaussian_density(abs(mu - U), sigma)

  # Initialize density vector
  density <- numeric(length(y))

  # Vectorized conditions
  above_U <- y > U
  above_mu <- (y <= U) & (y > mu)
  below_mu <- y <= mu

  # Assign density values based on conditions
  density[above_U] <- 0
  density[above_mu] <- 1
  density[below_mu] <- exp(-0.5 * ((y[below_mu] - mu)^2 / sigma_sq))

  # Normalize density
  density <- density / C

  return(density)
}
