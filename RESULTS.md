# Experimental Results

## CEBaB 100-Epoch Comparison

Results below use the existing 100-epoch CEBaB artifacts for seeds 123 and
2024. Task and concept accuracies are independently recomputed checkpoint
metrics. EU correlations are taken from the corresponding saved test results.

| Encoder | Schedule | Task accuracy | Concept accuracy | ρ(EU, AU) | ρ(EU, Err) | ρ(AU, H), review mean | ρ(AU, H), pooled |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | Ordinary | **0.776** | **0.821** | -0.025 | **0.224** | **0.184** | **0.137** |
| RoBERTa | Phased | **0.784** | 0.807 | -0.086 | 0.127 | 0.160 | 0.067 |
| ModernBERT | Ordinary | 0.697 | 0.772 | -0.145 | 0.142 | 0.088 | 0.098 |
| ModernBERT | Phased | 0.697 | 0.766 | -0.207 | 0.089 | 0.102 | 0.101 |
| DeBERTa-v3 | Ordinary | 0.767 | 0.794 | -0.128 | 0.143 | 0.046 | 0.037 |
| DeBERTa-v3 | Phased | 0.772 | 0.794 | -0.536 | 0.105 | 0.102 | 0.047 |
| DistilBERT | Ordinary | 0.761 | 0.807 | -0.227 | 0.182 | 0.133 | 0.105 |
| DistilBERT | Phased | 0.767 | 0.797 | -0.150 | **0.206** | **0.166** | 0.115 |

The review-mean statistic first averages AU and annotator entropy over the four
aspects for each review, then computes Spearman correlation across reviews. The
pooled statistic computes Spearman correlation over all review-aspect pairs
`(i, k)`.

### Aspect-level ρ(AU, H)

| Encoder | Schedule | Food | Service | Ambiance | Noise |
|---|---:|---:|---:|---:|---:|
| RoBERTa | Ordinary | **0.299** | 0.081 | **0.076** | 0.043 |
| RoBERTa | Phased | 0.224 | 0.064 | -0.029 | 0.013 |
| ModernBERT | Ordinary | 0.210 | 0.031 | -0.028 | **0.044** |
| ModernBERT | Phased | 0.196 | 0.054 | -0.010 | 0.016 |
| DeBERTa-v3 | Ordinary | 0.075 | 0.035 | -0.033 | -0.001 |
| DeBERTa-v3 | Phased | 0.198 | 0.051 | -0.017 | -0.043 |
| DistilBERT | Ordinary | 0.244 | 0.055 | 0.049 | 0.035 |
| DistilBERT | Phased | 0.276 | **0.089** | 0.040 | -0.036 |

Food supplies most of the measured AU-worker-entropy relationship. Service is
modest, while ambiance and noise are generally close to zero.

### RoBERTa seed-level uncertainty results

| Schedule | Seed | ρ(EU, AU) | ρ(EU, Err) | ρ(AU, H), review mean |
|---|---:|---:|---:|---:|
| Ordinary | 123 | -0.006 | 0.236 | 0.165 |
| Ordinary | 2024 | -0.045 | 0.213 | 0.204 |
| Phased | 123 | -0.123 | 0.087 | 0.145 |
| Phased | 2024 | -0.049 | 0.167 | 0.175 |
| **Ordinary mean** | — | **-0.025** | **0.224** | **0.184** |
| **Phased mean** | — | **-0.086** | **0.127** | **0.160** |

### Legacy AU proxy versus worker entropy

| Metric | Value | Valid worker-entropy correlation? |
|---|---:|---|
| Historical AU-threshold proxy | approximately 0.74 | No |
| RoBERTa ordinary review-mean ρ(AU, H) | 0.184 | Yes |
| RoBERTa ordinary pooled ρ(AU, H) | 0.137 | Yes |
| RoBERTa ordinary food ρ(AU_food, H_food) | 0.299 | Yes |

The historical value near 0.74 was based on a target constructed from AU:

```text
ρ(AU, 1[AU > 0.5])
```

It is circular and must not be reported as `ρ(AU, H_worker)`. Valid values use
annotator entropy calculated independently from the CEBaB aspect-label
distributions.

### Summary

- Highest task accuracy: phased RoBERTa, 0.784.
- Highest concept accuracy: ordinary RoBERTa, 0.821.
- Strongest ρ(EU, Err): ordinary RoBERTa, 0.224.
- Strongest valid review-level ρ(AU, H): ordinary RoBERTa, 0.184.
- Strongest valid aspect-level ρ(AU, H): RoBERTa food, 0.299.
- Ordinary RoBERTa is currently the strongest balanced configuration.
