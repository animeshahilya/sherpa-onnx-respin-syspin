#!/usr/bin/env python3
"""Give each Rasa speaker its own speaking style inside the shared model.

The Rasa model takes a style (emotion_id): ALEXA, BOOK, NEWS, WIKI... Our
exports froze one style, ALEXA, for all 20 speakers (`emotion_frozen_idx`),
because sherpa-onnx and the app pass only `sid`. Measured per speaker
(Whisper character error rate, 3 sentences x 2 renders per style), ALEXA is
not the clearest style for every speaker, so this replaces the constant with
a lookup by speaker: `emotion = STYLE_BY_SPEAKER[sid]`. Inputs, outputs and
everything else stay the same, so the app and sherpa-onnx need no change.

usage: build_rasa_styles.py <in.onnx> <out.onnx>
       build_rasa_styles.py --check <in.onnx>   # all-ALEXA table == original
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

STYLES = {'ALEXA': 0, 'BOOK': 3, 'INDICTTS': 9, 'NEWS': 10, 'WIKI': 16}

# sid -> style. Filled from the measurements (see README "Speaking styles").
STYLE_BY_SPEAKER = ['ALEXA', 'ALEXA', 'INDICTTS', 'ALEXA', 'NEWS', 'ALEXA', 'ALEXA', 'ALEXA', 'WIKI',
                    'INDICTTS', 'WIKI', 'NEWS', 'WIKI', 'INDICTTS', 'WIKI', 'NEWS', 'ALEXA', 'ALEXA',
                    'ALEXA', 'ALEXA']


def rewrite(model, table):
    g = model.graph
    users = [n for n in g.node if 'emotion_frozen_idx' in n.input]
    assert len(users) == 1 and users[0].op_type == 'Gather', users
    g.initializer.append(numpy_helper.from_array(np.array(table, np.int64), name='style_by_speaker'))
    g.node.insert(list(g.node).index(users[0]),
                  helper.make_node('Gather', ['style_by_speaker', 'sid'], ['style_for_speaker'],
                                   name='/style_by_speaker/Gather', axis=0))
    users[0].input[list(users[0].input).index('emotion_frozen_idx')] = 'style_for_speaker'
    keep = [i for i in g.initializer if i.name != 'emotion_frozen_idx']
    del g.initializer[:]
    g.initializer.extend(keep)
    return model


def check(path):
    import onnxruntime as ort
    original = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    rewritten = ort.InferenceSession(rewrite(onnx.load(path), [0] * 20).SerializeToString(),
                                     providers=['CPUExecutionProvider'])
    x = np.array([[0, 20, 0, 31, 0, 44, 0, 52, 0]], np.int64)
    for sid in (0, 8, 19):
        feeds = {'input': x, 'input_lengths': np.array([x.shape[1]], np.int64),
                 'scales': np.array([0.0, 1.0, 0.0], np.float32), 'sid': np.array([sid], np.int64)}
        a = original.run(None, feeds)[0]
        b = rewritten.run(None, feeds)[0]
        assert np.array_equal(a, b), (sid, float(np.abs(a - b).max()))
    print('identical with every speaker on ALEXA:', path)


if __name__ == '__main__':
    if sys.argv[1] == '--check':
        check(sys.argv[2])
    else:
        table = [STYLES[s] for s in STYLE_BY_SPEAKER]
        onnx.save(rewrite(onnx.load(sys.argv[1]), table), sys.argv[2])
        print('wrote', sys.argv[2], dict(enumerate(STYLE_BY_SPEAKER)))
