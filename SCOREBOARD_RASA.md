# Rasa scoreboard (tuned pace per sid, large-v3 WER, proxy MOS)

> WER is whisper-large-v3 on first-sentence evals — relative within a sid only; brx/doi/san decoded with the Hindi model. Sids flagged judge-fail keep length_scale 1.0 (see audit/asr_judge.json). proxy_mos is a signal heuristic, not a listening test.

| sid | voice | length | WER | recheck | mos* | cps |
|---|---|---|---|---|---|---|
| 10 | vits-rasa-mai-male-alt | 1.0 | 0.0 | 0.0556 | 4.72 | 11.45 |
| 16 | vits-rasa-pan-male | 1.05 | 0.0 | 0.15 | 4.85 | 14.4 |
| 17 | vits-rasa-san-male | 1.0 | 0.0 | 0.0 | 4.83 | 11.3 |
| 18 | vits-rasa-tam-female | 1.0 | 0.0 | 0.0 | 4.93 | 16.78 |
| 7 | vits-rasa-doi-male | 1.05 | 0.0435 | 0.087 | 4.86 | 12.94 |
| 2 | vits-rasa-bn-female-alt | 1.0 | 0.0625 | 0.125 | 4.77 | 14.29 |
| 12 | vits-rasa-mr-female-alt | 1.0 | 0.0625 | 0.0625 | 4.98 | 15.09 |
| 8 | vits-rasa-kn-female-alt | 1.0 | 0.0769 | 0.0769 | 4.89 | 12.59 |
| 9 | vits-rasa-kn-male-alt | 1.0 | 0.0769 | 0.0769 | 4.93 | 13.69 |
| 14 | vits-rasa-ne-female | 1.0 | 0.0833 | 0.0833 | 4.85 | 13.63 |

Judge-fail sids (default pace, needs ears): 0 vits-rasa-asm-female, 1 vits-rasa-asm-male, 3 vits-rasa-bn-male-alt, 4 vits-rasa-brx-female, 5 vits-rasa-brx-male, 6 vits-rasa-doi-female, 11 vits-rasa-mal-female, 13 vits-rasa-mr-male-alt, 15 vits-rasa-pan-female, 19 vits-rasa-te-female-alt
