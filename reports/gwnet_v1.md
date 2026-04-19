# Graph WaveNet v1 — training report (SMOKE)

- **mode:** `dummy`
- **accelerator:** `cpu`
- **train time:** 14.9s (epochs ran: 3)
- **params:** 29,336
- **best ckpt:** `/home/divyansh/Downloads/hackathon/energy/notebooks/ckpt_local/gwnet-epochepoch=01-valval_loss=0.2489.ckpt`

## Metrics (p50 on original scale)

| metric | value |
|---|---|
| MAE  | 1.3836 |
| RMSE | 1.8749 |
| MAPE | 1.2653 |
| pinball (overall) | 0.2814 |
| 80%-interval coverage | 0.786 |

Per-quantile pinball: {'0.1': 0.19, '0.5': 0.4265, '0.9': 0.2278}

## Plots

![loss curve](figures/gwnet_v1_losscurve.png)

![reliability](figures/gwnet_v1_reliability.png)

![sample ribbon](figures/gwnet_v1_ribbon.png)

## Notes

Run config:

```json
{
  "seed": 42,
  "horizon": 24,
  "input_len": 168,
  "num_quantiles": 3,
  "quantile_levels": [
    0.1,
    0.5,
    0.9
  ],
  "batch_size": 4,
  "epochs": 2,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "hidden_dim": 16,
  "num_blocks": 3,
  "kernel_size": 2,
  "dilation_growth": 2,
  "dropout": 0.3,
  "input_dim": 5,
  "num_nodes": 20,
  "num_timesteps": 2000,
  "mode": "dummy"
}
```

## Integrated Gradients (3 random points)

![ig sanity](figures/gwnet_v1_ig_sanity.png)

- idx=9 node=15 hstep=15: top=[('dow_sin', 2.038746424659621e-05), ('hour_sin', 1.9336099285283126e-05), ('temp_scaled', 1.3087155821267515e-05), ('hour_cos', 1.1410761544539127e-05), ('dow_cos', 3.0811565920885187e-06)]
- idx=47 node=8 hstep=20: top=[('temp_scaled', 3.504312189761549e-05), ('dow_sin', 1.2516071365098469e-05), ('hour_sin', 1.043004340317566e-05), ('dow_cos', 3.51568291989679e-06), ('hour_cos', 3.3499870824016398e-06)]
- idx=9 node=13 hstep=4: top=[('dow_sin', 9.678806236479431e-06), ('hour_cos', 6.718501026625745e-06), ('hour_sin', 6.6859879552794155e-06), ('temp_scaled', 3.242283355575637e-06), ('dow_cos', 2.2125809664430562e-06)]
