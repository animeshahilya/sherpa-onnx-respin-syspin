# Rasa speaking style per speaker

Mean Whisper character error rate, 3 sentences x 2 takes. Chosen: lowest, unless ALEXA (current) is within 2 points. Assamese (0, 1) not measurable by Whisper: ALEXA.

| sid | voice | ALEXA | BOOK | INDICTTS | NEWS | WIKI | chosen |
|---|---|---|---|---|---|---|---|
| 2 | bn_IN-tithi-medium | 24.5% | 21.9% | 18.6% | 21.8% | 19.9% | INDICTTS |
| 3 | bn_IN-anirban-medium | 22.3% | 27.3% | 23.0% | 28.0% | 25.9% | ALEXA |
| 4 | brx_IN-mainao-medium | 59.2% | 57.3% | 61.7% | 55.8% | 58.9% | NEWS |
| 5 | brx_IN-sansuma-medium | 57.6% | 58.2% | 59.6% | 55.7% | 60.9% | ALEXA |
| 6 | doi_IN-sheetal-medium | 36.4% | 37.7% | 35.8% | 35.1% | 38.6% | ALEXA |
| 7 | doi_IN-vijay-medium | 38.1% | 44.1% | 39.8% | 41.1% | 37.1% | ALEXA |
| 8 | kn_IN-spoorthi-medium | 15.4% | 14.2% | 10.8% | 11.8% | 9.6% | WIKI |
| 9 | kn_IN-chetan-medium | 14.1% | 13.4% | 8.4% | 14.2% | 14.6% | INDICTTS |
| 10 | mai_IN-shravan-medium | 36.0% | 34.7% | 35.9% | 35.2% | 32.0% | WIKI |
| 11 | ml_IN-aparna-medium | 56.4% | 66.5% | 42.1% | 38.8% | 108.6% | NEWS (unreliable: Whisper hallucinated) |
| 12 | mr_IN-mrunal-medium | 16.3% | 17.4% | 16.3% | 18.8% | 14.0% | WIKI |
| 13 | mr_IN-tejas-medium | 20.3% | 20.3% | 15.3% | 23.5% | 15.3% | INDICTTS |
| 14 | ne_NP-prerana-medium | 28.5% | 28.1% | 40.9% | 24.2% | 23.4% | WIKI |
| 15 | pa_IN-simran-medium | 37.9% | 36.5% | 34.8% | 32.1% | 33.0% | NEWS |
| 16 | pa_IN-harpreet-medium | 28.4% | 32.7% | 46.0% | 34.5% | 32.9% | ALEXA (unreliable: Whisper hallucinated) |
| 17 | sa_IN-vedant-medium | 16.0% | 20.5% | 22.4% | 20.2% | 26.5% | ALEXA |
| 18 | ta_IN-kaveri-medium | 7.0% | 6.0% | 5.5% | 10.3% | 10.2% | ALEXA |
| 19 | te_IN-harini-medium | 18.8% | 24.7% | 17.5% | 22.7% | 20.6% | ALEXA |

STYLE_BY_SPEAKER = ["ALEXA", "ALEXA", "INDICTTS", "ALEXA", "NEWS", "ALEXA", "ALEXA", "ALEXA", "WIKI", "INDICTTS", "WIKI", "NEWS", "WIKI", "INDICTTS", "WIKI", "NEWS", "ALEXA", "ALEXA", "ALEXA", "ALEXA"]
