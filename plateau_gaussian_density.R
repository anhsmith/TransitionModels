
# From https://link.springer.com/article/10.1007/s40300-022-00235-y

c_plateau_gaussian_density <- function(delta, sigma1, sigma2) {

  sqrt_2pi <- sqrt(2 * pi)

  return(
    (sqrt_2pi * sigma1 / 2) + (sqrt_2pi * sigma2 / 2) + 2 * delta
  )
}

plateau_gaussian_density <- function(y, mu, delta, sigma1, sigma2) {

  # Precompute repeated values
  sigma1_sq <- sigma1^2
  sigma2_sq <- sigma2^2

  # Compute the normalization constant C
  C <- c_plateau_gaussian_density(delta, sigma1, sigma2)

  # Conditional structure for the piecewise density function
  if (y < (mu - delta)) {
    density <- exp(-0.5 * ((y - (mu - delta))^2 / sigma1_sq))
  } else if (y <= (mu + delta)) {
    density <- 1
  } else {  # y > mu + delta
    density <- exp(-0.5 * ((y - (mu + delta))^2 / sigma2_sq))
  }

  # Return the normalized density
  return(density / C)
}

##

c_uniform_gaussian_density <- function(delta, sigma) {

  sqrt_2pi <- sqrt(2 * pi)

  return(
    (sqrt_2pi * sigma / 2) + delta
  )
}


uniform_left_gaussian_density <- function(y, mu, sigma, L) {

  # Precompute repeated values
  sigma_sq <- sigma^2

  # Compute the normalization constant C
  C <- c_uniform_gaussian_density(abs(mu-L), sigma)

  # Conditional structure for the piecewise density function
  density <-
    ifelse(
      y < L, 0,
    ifelse(
      y < mu, 1,
    exp(-0.5 * ((y - mu)^2 / sigma_sq))
    ))

  # Return the normalized density
  return(density / C)
}



uniform_right_gaussian_density <- function(y, mu, sigma, U) {

  # Precompute repeated values
  sigma_sq <- sigma^2

  # Compute the normalization constant C
  C <- c_uniform_gaussian_density(abs(mu-U), sigma)

  # Conditional structure for the piecewise density function
  density <-
    ifelse(
      y > U, 0,
      ifelse(
        y > mu, 1,
        exp(-0.5 * ((y - mu)^2 / sigma_sq))
      ))

  # Return the normalized density
  return(density / C)
}
