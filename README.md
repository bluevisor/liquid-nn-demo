# Liquid Neural Network — from scratch

### ▶ Live demo: **[liquid-nn-demo.vercel.app](https://liquid-nn-demo.vercel.app)**

A [Liquid Time-Constant (LTC) network](https://arxiv.org/abs/2006.04439) (Hasani et al., AAAI 2021)
implemented from scratch in PyTorch, with a set of self-contained web pages that make the idea
tangible — including a digit recognizer that runs the trained model's ODE solver **entirely in your
browser**, no server-side inference.

Try it live: [draw a digit](https://liquid-nn-demo.vercel.app) or read the
[illustrated explainer](https://liquid-nn-demo.vercel.app/how_liquid_neurons_work.html).

The one idea behind a liquid neuron: it's a leaky integrator whose *reaction speed* is itself a
learned function of the input, so its effective time constant flows with the data.

## What's here

| File | What it is |
|------|------------|
| **`index.html`** | Browser digit recognizer. A 256-neuron LTC trained on row-sequential MNIST reads a hand-drawn digit **one row at a time** (28 timesteps), integrating its state down the image. Visualizes the hidden state over time, the per-neuron liquid gates, and the belief trajectory. ~98.7% MNIST test accuracy, with test-time augmentation for hand-drawn robustness. |
| **`how_liquid_neurons_work.html`** | An illustrated explainer — from a leaky bucket up to the τ_eff equation — with a live single-neuron bench you can drive. |
| **`liquid_lab.html`** | Interactive lab for poking at one liquid neuron's dynamics. |
| **`liquid_demo.py`** | The LTC neuron built from scratch, plus an LTC-vs-GRU experiment on irregularly-sampled signal prediction (liquid net generalizes to sampling rates it never trained on). |
| **`train_lnn_mnist.py`** | Trains the row-sequential MNIST model and exports `lnn_weights.json`. |
| **`lnn_weights.json`** | The trained weights the recognizer fetches at startup. |

## Running the demo

The pages fetch `lnn_weights.json`, so they must be **served over http** (browsers block `fetch()`
on `file://`). From this directory:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/> for the recognizer, or `/how_liquid_neurons_work.html` for the
explainer. No build step, no dependencies — the model runs in plain JavaScript.

## Retraining

```bash
pip install torch torchvision matplotlib
python train_lnn_mnist.py     # downloads MNIST, trains, writes lnn_weights.json
```

The recognizer reads the network's size and solver settings from the JSON, so after retraining you
only need to reload the page — the weights are external, not embedded.

To reproduce the LTC-vs-GRU figures:

```bash
python liquid_demo.py         # writes liquid_neuron.png, liquid_vs_gru.png
```

## The model, briefly

Each neuron integrates the ODE `dx/dt = −[1/τ + f(u,x)]·x + f(u,x)·A`, where the gate `f` is a small
learned sigmoid. Grouping the decay terms gives the effective time constant
`τ_eff = 1 / (1/τ + f(u,x))` — the "liquid" quantity that shrinks when a strong signal arrives
(neuron reacts fast) and relaxes back to its slow baseline when the input is quiet. Integration uses
a stable semi-implicit Euler step:

```
x_next = (x + Δt·f·A) / (1 + Δt·(1/τ + f))
```

Because Δt sits directly in the update, the network handles irregular time sampling natively and
generalizes across timescales — which is exactly what `liquid_demo.py` demonstrates against a GRU.
