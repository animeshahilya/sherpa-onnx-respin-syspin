# SYSPIN scoreboard (tuned configs, large-v3 WER, proxy MOS)

> WER is whisper-large-v3 on short eval passages — relative within a voice only; bho/hne/mai/mag decoded with the Hindi model. proxy_mos is a signal heuristic, not a listening test.

| voice | length | WER | recheck | mos* | cps | dur_s |
|---|---|---|---|---|---|---|
| vits-syspin-en-female | 0.95 | 0.0 | 0.0 | 4.49 | 11.96 | 7.44 |
| vits-syspin-en-male | 1.0 | 0.0 | 0.0 | 4.55 | 14.88 | 5.98 |
| vits-syspin-bho-female | 1.0 | 0.0667 | 0.0667 | 4.49 | 9.6 | 7.4 |
| vits-syspin-bn-male | 1.0 | 0.0714 | 0.1429 | 4.56 | 11.79 | 6.14 |
| vits-syspin-bn-female | 1.0 | 0.1429 | 0.1429 | 4.58 | 10.74 | 6.71 |
| vits-syspin-te-female | 0.95 | 0.1538 | 0.1538 | 4.6 | 11.67 | 7.02 |
| vits-syspin-hne-female | 0.95 | 0.1875 | 0.1875 | 4.49 | 11.14 | 6.28 |
| vits-syspin-hne-male | 0.9 | 0.1875 | 0.25 | 4.49 | 10.1 | 6.93 |
| vits-syspin-bho-male | 0.9 | 0.2 | 0.2 | 4.5 | 10.41 | 6.82 |
| vits-syspin-hi-female | 1.05 | 0.2 | 0.2 | 4.44 | 10.59 | 6.33 |
| vits-syspin-mag-female | 1.1 | 0.2 | 0.2 | 4.52 | 7.6 | 9.08 |
| vits-syspin-gu-female | 1.05 | 0.2857 | 0.2857 | 4.68 | 5.52 | 11.41 |
| vits-syspin-te-male | 1.05 | 0.3077 | 0.3846 | 4.53 | 11.52 | 7.12 |
| vits-syspin-hi-male | 1.0 | 0.3333 | 0.3333 | 4.5 | 10.34 | 6.48 |
| vits-syspin-kn-male | 1.0 | 0.3333 | 0.3333 | 4.49 | 11.07 | 7.5 |
| vits-syspin-mai-female | 1.0 | 0.3333 | 0.4667 | 4.45 | 10.04 | 6.67 |
| vits-syspin-mr-male | 1.05 | 0.3571 | 0.3571 | 4.51 | 9.64 | 7.16 |
| vits-syspin-mai-male | 0.9 | 0.4 | 0.4667 | 4.38 | 7.73 | 8.66 |
| vits-syspin-kn-female | 1.0 | 0.4167 | 0.4167 | 4.69 | 9.28 | 8.94 |
| vits-syspin-gu-male | 1.1 | 0.5 | 0.5 | 4.1 | 6.44 | 9.78 |
| vits-syspin-mr-female | 0.95 | 0.5 | 0.5 | 4.5 | 9.76 | 7.07 |
| vits-syspin-mag-male | 1.05 | 0.6 | 0.6 | 4.44 | 8.61 | 8.02 |

Worst WER voices (tune first / review frontend): vits-syspin-gu-male (0.5), vits-syspin-mr-female (0.5), vits-syspin-mag-male (0.6)
