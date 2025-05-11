def compute_option_payoff(all_paths, strike_temp):
    """
    Compute the payoff of a European-style call option based on average temperature.

    Parameters:
    - all_paths: ndarray, simulated temperature paths of shape (nr_paths, nr_days)
    - strike_temp: float, strike temperature of the option

    Returns:
    - payoffs: ndarray, payoff for each simulated path
    """
    average_temps = np.mean(all_paths, axis=1)
    payoffs = np.maximum(average_temps - strike_temp, 0)
    return payoffs


def calculate_option_price(payoffs):
    """
    Calculate the price of the option as the average payoff.

    Parameters:
    - payoffs: ndarray, payoff for each simulated path

    Returns:
    - option_price: float, estimated price of the option
    """
    option_price = np.mean(payoffs)
    return option_price


# Example parameters
nr_days = 365
nr_paths = 1000
params_all = (10.0, 0.01, 5.0, -3.0)
kappa = 0.2
sigma = 2.0
ini_temp = 15.0
strike_temp = 18.0  # Example strike temperature

# Simulate multiple temperature paths
all_paths = multiple_paths(nr_days, params_all, kappa, sigma, ini_temp, nr_paths)

# Compute option payoffs
payoffs = compute_option_payoff(all_paths, strike_temp)

# Calculate option price
option_price = calculate_option_price(payoffs)

print(f"Estimated option price: {option_price:.2f}")
