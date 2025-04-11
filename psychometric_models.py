#sklearn style interface for fitting psychometric behavior models. We start with a psychometric
#function 

import numpy as np
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score
from scipy.special import erf

class PsychometricFunction(BaseEstimator, ClassifierMixin):
    def __init__(self, function, init_state=None):
        self.function = function
        self.init_state = init_state
        self.state = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
       # self.random_state_ = check_random_state(self.random_state)
        self.popt, self.pcov = curve_fit(self.function, X, y, p0=self.init_state, maxfev = 5000 )
        self.state = self.popt
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'popt', 'pcov'])
        # Input validation
        #X = check_array(X)
        return self.function(X, *self.popt)

    def score(self, X, y, sample_weight=None):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'popt', 'pcov'])
        # Input validation
        X = check_array(X)
        y = check_array(y)
        return r2_score(y, self.predict(X), sample_weight=sample_weight)

    def get_params(self, deep=True):
        return {'function': self.function, 'state': self.state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __repr__(self):
        return "PsychometricFunction(function=%s, state=%s)" % (self.function, self.state)

    def __str__(self):
        return "PsychometricFunction(function=%s, state=%s)" % (self.function, self.state)
    
    def __getstate__(self):
        return {'function': self.function, 'state': self.tate, 'popt': self.popt, 'pcov': self.pcov}
    
    def __setstate__(self, state):
        self.function = state['function']
        self.state = state['state']
        self.init_state = state['init_state']
        self.popt = state['popt']
        self.pcov = state['pcov']
        self.random_state_ = check_random_state(self.random_state)

    def __eq__(self, other):
        return self.function == other.function and self.random_state == other.random_state

    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return hash((self.function, self.random_state))

class ErfPsych(PsychometricFunction):
    def __init__(self, init_state=None):

        cdf = lambda x, bias, slope, gamma1, gamma2: gamma1 + (1 - gamma1 - gamma2) * (erf((x - bias) / slope) + 1) / 2

        super(ErfPsych, self).__init__(cdf, init_state)
    
class LogisticPsych(PsychometricFunction):
    def __init__(self, init_state=None):
        log_cdf = lambda x, a, b: 1 / (1 + np.exp(-a * (x - b)))
        super(LogisticPsych, self).__init__(log_cdf, init_state)

class WeibullPsych(PsychometricFunction):
    def __init__(self, init_state=None):
        super(WeibullPsych, self).__init__(lambda x, a, b: 1 - np.exp(-np.power(x / b, a)), init_state)

class GumbelPsych(PsychometricFunction):
    def __init__(self, init_state=None):
        super(GumbelPsych, self).__init__(lambda x, a, b: 1 - np.exp(-np.exp(-a * (x - b))), init_state)

class GaussianPsych(PsychometricFunction):
    def __init__(self, init_state=None):
        super(GaussianPsych, self).__init__(lambda x, a, b: 0.5 * (1 + erf((x - b) / (a * np.sqrt(2)))), init_state)
