import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    """ Class describing a probability density function.
    """
    def __init__(self, x, y):
        """ Constructor.
        """
        InterpolatedUnivariateSpline.__init__(self, x, y)
        # Compute function's CDF.
        ycdf = np.array([self.integral(x[0], xcdf) for xcdf in x])
        self.cdf = InterpolatedUnivariateSpline(x, ycdf)
        # Remove duplicates from array y and their respective in x array.
        x_new, index = np.unique(ycdf, return_index=True)
        y_new = x[index]

        # Compute function's PPF.
        self.ppf = InterpolatedUnivariateSpline(x_new, y_new)

    def prob(self, x1, x2):
        """ Return the probability for the random variable to be included
        between x1 and x2.
        """
        return self.cdf(x2) - self.cdf(x1)

    def rnd(self, size=1000):
        """ Return an array of random values from the pdf.
        """
        return self.ppf(np.random.uniform(size=size))

def test_fancy():
    """ Test ProbabilityDensityFunction class with a fancy function.
    """
    x = np.linspace(0., 1., 1000)
    y = np.zeros(x.shape)
    y[x < 0.25] = 1
    mask1 = (x >= 0.25) & (x < 0.5)
    y[mask1] = -8 * x[mask1] + 4
    mask2 = (x >= 0.5) & (x < 0.75)
    y[mask2] = 8 * x[mask2] -4
    y[x >= 0.75] = 1

    pdf = ProbabilityDensityFunction(x,y)

    plt.figure('pdf')
    plt.plot(x, pdf(x))
    plt.xlabel('x')
    plt.ylabel('pdf(x)')

    plt.figure('cdf')
    plt.plot(x, pdf.cdf(x))
    plt.xlabel('x')
    plt.ylabel('cdf(x)')

    plt.figure('ppf')
    q = np.linspace(0., 1., 250)
    plt.plot(q, pdf.ppf(q))
    plt.xlabel('q')
    plt.ylabel('ppf(q)')

    plt.figure('Sampling')
    rnd = pdf.rnd(1000000)
    plt.hist(rnd, bins=200)


if __name__ == "__main__":
    test_fancy()
    plt.show()