from distutils.core import setup

setup(name='swiftt', version='1.0',
      description='Library implementing the Taylor Differential Algebra. This covers automatic multivariate, '
                  'high-order differentiation as well as differentiation of implicit functions and a differentiating '
                  'operator on Taylor expansions.',
      author='Romain Serra', author_email='serra.romain@gmail.com', packages=['swiftt', 'swiftt.taylor'])
