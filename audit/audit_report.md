# Frontend audit

## A. Vocab coverage (female voice per lang)

| lang | size | nat_digits | ascii_digits | danda | halant | space |
|---|---|---|---|---|---|---|
| hi | 116 | 10/10 | 10/10 | 1/1 | 1/1 | Y |
| bn | 81 | 0/10 | 0/10 | 1/1 | 1/1 | Y |
| te | 116 | 2/10 | 10/10 | 1/1 | 1/1 | Y |
| kn | 76 | 0/10 | 0/10 | 0/1 | 1/1 | Y |
| mr | 134 | 10/10 | 10/10 | 1/1 | 1/1 | Y |
| gu | 110 | 10/10 | 10/10 | 0/1 | 1/1 | Y |
| bho | 87 | 0/10 | 0/10 | 0/1 | 1/1 | Y |
| hne | 103 | 3/10 | 0/10 | 1/1 | 1/1 | Y |
| mai | 128 | 10/10 | 4/10 | 1/1 | 1/1 | Y |
| mag | 116 | 9/10 | 10/10 | 1/1 | 1/1 | Y |
| en | 94 | -- | 6/10 | -- | -- | Y |

## B. Dropped chars: before (raw) vs after (frontend)

- hi/dashboard: before=0 [] | after=0 []
- hi/stress: before=1 ['U+0025'] | after=0 []
- bn/dashboard: before=0 [] | after=0 []
- bn/stress: before=11 ['U+0025', 'U+0031', 'U+0032', 'U+0035', 'U+09E6', 'U+09E7', 'U+09E8', 'U+09EA', 'U+09EB', 'U+201C', 'U+201D'] | after=0 []
- te/dashboard: before=0 [] | after=0 []
- te/stress: before=4 ['U+0025', 'U+0C67', 'U+0C68', 'U+0C6B'] | after=0 []
- kn/dashboard: before=0 [] | after=0 []
- kn/stress: before=8 ['U+0025', 'U+0030', 'U+0031', 'U+0032', 'U+0034', 'U+0035', 'U+201C', 'U+201D'] | after=0 []
- mr/dashboard: before=0 [] | after=0 []
- gu/dashboard: before=0 [] | after=0 []
- gu/stress: before=1 ['U+0025'] | after=0 []
- bho/dashboard: before=0 [] | after=0 []
- hne/dashboard: before=0 [] | after=0 []
- mai/dashboard: before=0 [] | after=0 []
- mag/dashboard: before=0 [] | after=0 []
- en/dashboard: before=0 [] | after=0 []
- en/stress: before=3 ['U+0024', 'U+0025', 'U+0032'] | after=0 []

## C. tokens.txt validation (sherpa-exact parse)

all 44 tokens files clean.
