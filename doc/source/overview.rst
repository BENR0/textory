=========================
Overview
=========================

Calculating textures
=====================

Variogram
----------

.. math::
   \gamma(h) = \frac{1}{2n(h)} \sum_{i=1}^{n(h)} (v(x_{i}) - v(x_{i}+h))^{2}

Madogram
----------

.. math::
   \gamma(h) = \frac{1}{2n(h)} \sum_{i=1}^{n(h)} |v(x_{i}) - v(x_{i}+h)|

Madogram
----------

.. math::
   \gamma(h) = \frac{1}{2n(h)} \sum_{i=1}^{n(h)} \sqrt{|v(x_{i}) - v(x_{i}+h)|}

Cross Variogram
----------------

.. math::
   \gamma(h) = \frac{1}{2n(h)} \sum_{i=1}^{n(h)} (v(x_{i}) - v(x_{i}+h))*(w(x_{i}) - w(x_{i}+h))

Pseudo Cross Variogram
-----------------------

.. math::
   \gamma(h) = \frac{1}{2n(h)} \sum_{i=1}^{n(h)} (v(x_{i}) - w(x_{i}+h))^{2}
