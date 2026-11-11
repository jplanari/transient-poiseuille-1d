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

## Ús amb fitxer de configuració

Tots els paràmetres es passen en un fitxer JSON (vegeu `configs/`):

```bash
python -m transient_poiseuille configs/copper_pressure_drop.json
python -m transient_poiseuille configs/copper_flow.json -o results/flow.npz
python -m transient_poiseuille configs/copper_pressure_drop.json --plot results/velocity.png
```

`--plot` escriu un PNG amb el camp u(x, r) (mirall de tot el diàmetre) i els perfils radials, amb les solucions analítiques com a línies discontínues quan s'aplica (μ constant, R constant, pressió imposada). Requereix `matplotlib`.

Estructura:

```json
{
  "geometry": {"R": 0.05, "L": 1.0},
  "fluid":    {"mu": 0.004, "rho": 8000.0},
  "drive":    {"mode": "pressure_drop", "dp": 0.001},
  "numerics": {"Nx": 1, "Nr": 80, "dt": 250.0, "t_end": 25000.0, "theta": 0.5, "convection": false},
  "output":   {"file": "../results/run.npz", "save_times": [500.0, 2500.0]}
}
```

- `drive.mode = "pressure_drop"` requereix `drive.dp`; `"flow"` requereix `drive.V_c` (i opcionalment `p_in`).
- Les claus desconegudes i els valors no vàlids (negatius, `theta` fora de [0,1], etc.) es rebutgen amb un missatge clar.
- Les rutes de sortida relatives es resolen respecte al fitxer de configuració.
- El fitxer `.npz` conté `u_final`, `Q_history`, `G_final`, `p_final`, `p_out` i els instantanis de `save_times`.

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

## Viscositat variable al llarg de x

`fluid.mu` pot ser un número o una llista amb un valor per cel·la axial (`Nx` valors). Vegeu `configs/copper_mu_gradient.json`.

- Cada cel·la és una secció de Poiseuille amb la seva μ_i. El flux és comú (R constant) i `G_i = (a_i - Q) / b_i`.
- Mode `flow`: `G_i = -8 μ_i V_c / R^2`.
- Mode `pressure_drop`: `Q = π R^4 Δp / (8 Σ μ_i Δx)`, i.e. una sèrie de resistències. El gradient es reparteix proporcionalment a μ_i.
- Amb R constant, la velocitat mitjana és la mateixa a totes les cel·les: només canvia el gradient de pressió, no el perfil de velocitats.
- Les proves estacionàries usen θ = 1 (Euler implícit). Amb Crank–Nicolson, l'arrencada excita modes radials ràpids que no s'amortitzen, i queda una dispersió d'uns 10⁻⁵ en G/μ amb dt = 250 s.

## Pendent

- Afegir l'equació de l'energia i la solidificació (`A(x,t)` i terme font `S`).
- Radi variable `R(x,t)`: el codi actual suposa radi constant, així que `dR/dt` i `dR/dx` no hi són.
- μ que depèn de (x, r), per exemple amb temperatura radial o zona pastosa.
- Verificar el terme convectiu contra un cas amb `R(x)` lentament variable.
