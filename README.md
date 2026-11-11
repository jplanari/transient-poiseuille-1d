# transient-poiseuille-1d

Solver transitori per al flux axisimètric en una barra cilíndrica de radi constant,
amb velocitat radial `u(eta, t)` a cada volum de control axial, i verificació
contra la solució analítica de Poiseuille i el seu arrencament transitori.

## Model

Per a cada cel·la axial (longitud `dx`), amb `eta = r/R` in (0, 1):

```
rho * (du/dt + u du/dx) = -G + (mu / R^2) * (1/eta) d/deta (eta du/deta)
du/deta(0) = 0,   u(1) = 0,   G = dp/dx
```

- Discretització radial: volums finits cèl·lula-centrada en `eta`.
- Temps: esquema theta (0.5 = Crank–Nicolson, 1 = Euler implícit).
- Convectiu `u du/dx`: opcional, explícit. **Encara no està verificat.**
- Dos modes:
  - `pressure_drop`: `G = -dp/L` imposat. Mode usat per a la verificació.
  - `flow`: velocitat de colada `V_c` i pressió d'entrada `p_in` imposades. Amb `R` constant
    i sense fonts, la continuïtat dona `Q_i = pi R^2 V_c` a cada cel·la; `G_i` surt de la
    solució radial i la pressió s'integra des de `p_in`.

## Verificació

Amb viscositat constant i pèrdua de càrrega donada, el solver recupera:

- El perfil de Poiseuille estacionari `u_s = -(G R^2 / 4 mu)(1 - eta^2)` i el cabal `Q = -pi R^4 G / (8 mu)`.
- L'arrencament transitori des del repòs, comparat amb la sèrie de Fourier–Bessel
  (`J0(alpha_n eta) exp(-alpha_n^2 t / tau)`, amb `tau = rho R^2 / mu`).
- Convergència de malla radial.
- En mode `flow`: `G = -8 mu V_c / R^2` i `p_out = p_in - 8 mu V_c L / R^2`.

```bash
python -m pytest -q                 # tests
python examples/verify_poiseuille.py  # informe numèric
```

## Instal·lació

```bash
pip install -r requirements.txt
```

## Pendent

- Afegir l'equació de l'energia i la solidificació (`A(x,t)` i terme font `S`).
- Radi variable `R(x,t)`: el codi actual suposa radi constant, així que `dR/dt` i `dR/dx` no hi són.
- Verificar el terme convectiu contra un cas amb `R(x)` lentament variable.
