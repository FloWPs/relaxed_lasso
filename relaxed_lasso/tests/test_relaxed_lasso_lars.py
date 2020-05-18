import pytest
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLars, LassoLarsCV, LinearRegression
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_raises
from sklearn.model_selection import KFold
from sklearn.datasets import make_regression

from relaxed_lasso import RelaxedLassoLars, RelaxedLassoLarsCV, relasso_lars_path


# Create highly colinear dataset for regression.
mu = np.repeat(0, 100)
dists = np.arange(100)
powers = [[np.abs(i-j) for j in dists] for i in dists]
r = np.power(.5, powers)
X = np.random.multivariate_normal(mu, r, size=50)
y = 7*X[:, 0] + \
    5*X[:, 10] + \
    3*X[:, 20] + \
    1*X[:, 30] + \
    .5*X[:, 40] + \
    .2*X[:, 50] + \
    np.random.normal(0, 2, 50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)

lasso = LassoLarsCV(cv=5).fit(X_train, y_train)
alpha = lasso.alpha_

# For testing when X input has a single feature
Xa, ya = make_regression(n_samples=50,
                         n_features=1,
                         random_state=0,
                         coef=False)

# For testing when y output vector is multidimensionnal
Xb, yb = make_regression(n_samples=50,
                         n_features=10,
                         n_informative=3,
                         n_targets=2,
                         noise=2,
                         random_state=0,
                         coef=False)


def test_theta_equal_1():
    # Validate that Relaxed asso with theta=1 is equivalent to Lasso.
    relasso = RelaxedLassoLars(alpha, 1).fit(X_train, y_train)
    lasso_prediction = lasso.predict(X_test)
    relasso_prediction = relasso.predict(X_test)
    assert_array_equal(lasso_prediction, relasso_prediction)


def test_theta_equal_0():
    # Validate that Relaxed asso with theta=0 is equivalent to OLS.
    relasso = RelaxedLassoLars(alpha, 0).fit(X_train, y_train)
    mask = lasso.active_
    lr = LinearRegression().fit(X_train[:, mask], y_train)
    ols_prediction = lr.predict(X_test[:, mask])
    relasso_prediction = relasso.predict(X_test)
    assert_array_almost_equal(ols_prediction, relasso_prediction)


def test_simple_vs_refined_algorithm() :
    # Test the consistency of the results between the 2 versions of the agorithm.
    
    alpha = 1.0
    theta = 1.0
    
    # Simple Algorithm (2 steps of Lasso Lars)
    lasso1 = LassoLars(alpha=alpha)
    lasso1.fit(X_train, y_train)
    coef1 = pd.Series(lasso1.coef_)
    X1 = pd.DataFrame(X_train.copy())
    X1.loc[:, coef1[coef1==0].index] = 0
    
    lasso2 = LassoLars(alpha=alpha*theta)
    lasso2.fit(X1, y_train)
    pred_simple = lasso2.predict(X_test)
    
    # Refined Algorithm
    relasso = RelaxedLassoLars(alpha=alpha, theta=theta)
    relasso.fit(X_train, y_train)
    pred_refined = relasso.predict(X_test)
        
    assert_array_almost_equal(pred_simple, pred_refined)
    assert_array_almost_equal(lasso2.coef_, relasso.coef_)
    assert_almost_equal(lasso2.score(X_test, y_test), relasso.score(X_test, y_test))
    

def test_relaxed_lasso_lars():
    # Relaxed Lasso regression convergence test using score.
    
    # With more samples than features
    X1, y1 = make_regression(n_samples=50,
                             n_features=10,
                             n_informative=3,
                             noise=2,
                             random_state=0,
                             coef=False)
    
    relasso = RelaxedLassoLars()
    relasso.fit(X1, y1)
    
    assert relasso.coef_.shape == (X1.shape[1], )
    assert relasso.score(X1, y1) > 0.9
    
    # With more features than samples
    X2, y2 = make_regression(n_samples=50,
                             n_features=100,
                             n_informative=3,
                             noise=2,
                             random_state=0,
                             coef=False)
    
    relasso = RelaxedLassoLars()
    relasso.fit(X2, y2)
    
    assert relasso.coef_.shape == (X2.shape[1], )
    assert relasso.score(X2, y2) > 0.9


@pytest.mark.parametrize("X, y", [(X, y), (Xa, ya), (Xb, yb)])
def test_shapes(X, y):
    # Test shape of attributes.
    relasso = RelaxedLassoLars()
    relasso.fit(X, y)
    
    # Multi-targets
    if type(y[0]) == np.ndarray:
        n_alphas = len(relasso.active_[0])
        assert len(relasso.alphas_) == y.shape[1]
        assert relasso.alphas_[0].shape == (n_alphas + 1,)
        assert relasso.coef_.shape == (y.shape[1], X.shape[1])
        assert len(relasso.coef_path_) == y.shape[1]
        if len(relasso.alphas_[0]) > 1 :
            assert relasso.coef_path_[0].shape == (X.shape[1], len(relasso.alphas_[0]), len(relasso.alphas_[0]) - 1)
        else :
            assert relasso.coef_path_[0].shape == (X.shape[1], 1, 1)
        assert relasso.intercept_.shape == (y.shape[1],)

    # 1-target
    else:
        n_alphas = len(relasso.active_)
        assert relasso.alphas_.shape == (n_alphas + 1,)
        assert relasso.coef_.shape == (X.shape[1],)
        if len(relasso.alphas_) > 1 :
            assert relasso.coef_path_.shape == (X.shape[1], len(relasso.alphas_), len(relasso.alphas_) - 1)
        else :
            assert relasso.coef_path_.shape == (X.shape[1], 1, 1)


@pytest.mark.parametrize("X, y", [(X, y), (Xa, ya)])
def test_relaxed_lasso_lars_cv(X, y):
    # Idem for RelaxedLassoLarsCV
    relasso_cv = RelaxedLassoLarsCV()
    relasso_cv.fit(X, y)
    assert relasso_cv.coef_.shape == (X.shape[1],)
    assert type(relasso_cv.intercept_) == np.float64

    cv = KFold(5)
    relasso_cv.set_params(cv=cv)
    relasso_cv.fit(X, y)
    assert relasso_cv.coef_.shape == (X.shape[1],)
    assert type(relasso_cv.intercept_) == np.float64


def test_x_none_gram_none_raises_value_error():
    # Test that relasso_lars_path with no X and Gram raises exception.
    Xy = np.dot(X.T, y)
    assert_raises(ValueError, relasso_lars_path, None, y, Gram=None, Xy=Xy)


def test_no_path():
    # Test that the 'return_path=False' option returns the correct output.
    alphas_, _, coef_path_ = relasso_lars_path(X, y)
    alpha_, _, coef = relasso_lars_path(X, y, return_path=False)
    
    # coef_path : array, shape (n_features, n_alphas + 1, n_alphas)
    assert_array_almost_equal(coef, coef_path_[:,-1,-1]) 
    assert alpha_ == alphas_[-1]


def test_no_path_precomputed():
    # Test that the 'return_path=False' option with Gram remains correct.
    G = np.dot(X.T, X)
    alphas_, _, coef_path_ = relasso_lars_path(
        X, y, method='lasso', Gram=G)
    alpha_, _, coef = relasso_lars_path(
        X, y, method='lasso', Gram=G, return_path=False)

    assert_array_almost_equal(coef, coef_path_[:,-1,-1])
    assert alpha_ == alphas_[-1]


def test_no_path_all_precomputed():
    # Test that the 'return_path=False' option with Gram and Xy remains correct.
    G = np.dot(X.T, X)
    Xy = np.dot(X.T, y)

    alphas_, _, coef_path_ = relasso_lars_path(
        X, y, method='lasso', Xy=Xy, Gram=G, alpha_min=0.9)
    alpha_, _, coef = relasso_lars_path(
        X, y, method='lasso', Gram=G, Xy=Xy, alpha_min=0.9, return_path=False)

    assert_array_almost_equal(coef, coef_path_[:,-1,-1])

    assert alpha_ == alphas_[-1]


def test_relasso_lars_path_length():
    # Test that the path length of the RelaxedLassoLars is right.
    
    relasso = RelaxedLassoLars(alpha=0.2)
    relasso.fit(X, y)
    relasso2 = RelaxedLassoLars(alpha=relasso.alphas_[2])
    relasso2.fit(X, y)

    assert_array_almost_equal(relasso.alphas_[:3], relasso2.alphas_)
    
    # Also check that the sequence of alphas is always decreasing
    assert np.all(np.diff(relasso.alphas_) < 0)


def test_singular_matrix():
    # Test when input is a singular matrix.
    X1 = np.array([[1, 1.], [1., 1.]])
    y1 = np.array([1, 1])
    _, _, coef_path = relasso_lars_path(X1, y1)
    
    assert_array_almost_equal(coef_path.T[-1, :], [[0, 0], [1, 0]])
