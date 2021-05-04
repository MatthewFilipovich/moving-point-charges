
# Space-Time Computation and Visualization of the Electromagnetic Fields and Potentials Generated from Moving Point Charges
**moving-point-charges** is an open-source Python library for simulating the full 3D electromagnetic fields and potentials generated from moving point charges in arbitrary motion with varying speeds (see [American Journal of Physics](https://doi.org/10.1119/10.0003207) for paper).

<p align="center">
  <img width="300" src="https://github.com/MatthewFilipovich/moving-point-charges/blob/master/Paper_Figures/Animations/readme.gif">
</p>

## Getting Started
The *MovingChargesField* class calculates the electromagnetic fields and potentials at each grid point by first determining the delayed time of each point charge. The *Charge* abstract class is the parent class used to specify the trajectory of a point charge in three dimensions, and *MovingChargesField* is instantiated with a list of objects derived from the *Charge* class. The *MovingChargesField* object can then be used to calculate the *x*, *y*, and *z* components of the electromagnetic fields and potentials in the simulation at specified grid points. The *MovingChargesField* object can also calculate the individual Coulomb and radiation terms of the generated electric and magnetic fields.

Several classes that correspond to moving point charges are included in the library, including *OscillatingCharge*, *OrbittingCharge*, *LinearAcceleratingCharge*, *LinearDeceleratingCharge*, and *LinearVelocityCharge*. Visualization and animation examples using matplotlib are shown in the *Paper_Figures* folder. 

## A Brief Physics Background

The vector and scalar potentials of a moving point charge <img src="https://render.githubusercontent.com/render/math?math=q"> in the Lorenz gauge, known as the [Liénard–Wiechert potentials](https://en.wikipedia.org/wiki/Li%C3%A9nard%E2%80%93Wiechert_potential), at the position <img src="https://render.githubusercontent.com/render/math?math=\mathbf{r}_p(t)"/> with velocity <img src="https://render.githubusercontent.com/render/math?math=c\boldsymbol{\beta}"/> are given by

<img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{\Phi}(\mathbf{r}, t) = \frac{q}{4\pi\epsilon_0}\left[ \frac{1}{\kappa R}\right]_{t\mathrm{'}},">

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}(\mathbf{r}, t) = \frac{\mu_0 q}{4\pi}\left[ \frac{\boldsymbol{\beta}}{\kappa R}\right]_{t\mathrm{'}},">

where <img src="https://render.githubusercontent.com/render/math?math=R=\left| \mathbf{r}-\mathbf{r}_p\left(t\mathrm{'}\right) \right|">, <img src="https://render.githubusercontent.com/render/math?math=\kappa=1-\mathbf{n}(t\mathrm{'})\cdot \boldsymbol{\beta}(t\mathrm{'})"/> such that <img src="https://render.githubusercontent.com/render/math?math={\mathbf{n}=(\mathbf{r}-\mathbf{r}_p(t\mathrm{'}))/R}"/> is a unit vector from the position of the charge to the field point, and the quantity in brackets is to be evaluated at the delayed time <img src="https://render.githubusercontent.com/render/math?math=t\mathrm{'}"/> given by

<img src="https://render.githubusercontent.com/render/math?math=t\mathrm{'}=t-\frac{R(t\mathrm{'})}{c}."/>

The electric and magnetic fields produced from a moving point charge can be calculated directly from their scalar and vector potentials:

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{E}\left(\mathbf{r}, t\right) = \frac{q}{4\pi\epsilon_0} \Bigg[ \frac{\left( \mathbf{n}-\boldsymbol{\beta} \right)\left(1-\beta^2\right)}{\kappa^3 R^2}%2B\frac{\mathbf{n}}{c\kappa^3 R} \times \left[ \left(\mathbf{n}-\boldsymbol{\beta}\right) \times \boldsymbol{\dot{\beta}} \right] \Bigg]_{t\mathrm{'}},"/>

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{B} = \frac{1}{c} \left[ \mathbf{n} \times \mathbf{E} \right]_{t\mathrm{'}},"/>

where <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{\dot{\beta}}"/> is the derivative of <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{\beta}"/> with respect to <img src="https://render.githubusercontent.com/render/math?math=t\mathrm{'}"/>. The first term in the electric field equation is known as the "Coulomb field" and is independent of acceleration, while the second term is known as the "radiation field" and is linearly dependent on <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{\dot{\beta}}"/>.



