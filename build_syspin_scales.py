#!/usr/bin/env python3
"""Make the SYSPIN exports read noise_scale and noise_w from `scales`.

The SYSPIN ONNX exports take `scales` = [noise_scale, length_scale, noise_w]
but only use length_scale: noise_scale (0.667) and noise_w (1.0) were frozen
into the graph as constants. Apps that change them (the espeak-ng app's
speaking style) had no effect on these voices. This rewires the two constants
to scales[0] and scales[2]; at [0.667, x, 1.0] the output is unchanged.

usage: build_syspin_scales.py <in.onnx> <out.onnx>
       build_syspin_scales.py --check <in.onnx>   # same audio at the old values
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

FROZEN = {'/Constant_34_output_0': 0, '/duration_predictor/Constant_3_output_0': 2}


def rewrite(model):
    g = model.graph
    found = {n.output[0] for n in g.node if n.op_type == 'Constant' and n.output[0] in FROZEN}
    assert found == set(FROZEN), found
    for name, index in FROZEN.items():
        out = f'/scales_{index}'
        idx = helper.make_node('Constant', [], [out + '_idx'],
                               value=helper.make_tensor(out + '_idx', TensorProto.INT64, [], [index]))
        gather = helper.make_node('Gather', ['scales', out + '_idx'], [out], axis=0)
        for n in g.node:
            for i, x in enumerate(n.input):
                if x == name:
                    n.input[i] = out
        g.node.insert(0, gather)
        g.node.insert(0, idx)
    keep = [n for n in g.node if not (n.op_type == 'Constant' and n.output[0] in FROZEN)]
    del g.node[:]
    g.node.extend(keep)
    onnx.checker.check_model(model)
    return model


def seeded(model):
    for n in model.graph.node:
        if n.op_type == 'RandomNormalLike':
            n.attribute.append(helper.make_attribute('seed', 7.0))
    return model


def check(path):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    run = lambda m: ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
    original = run(seeded(onnx.load(path)))
    patched = run(seeded(rewrite(onnx.load(path))))
    x = np.array([[0, 20, 0, 31, 0, 44, 0, 52, 0, 23, 0]], np.int64)
    feed = lambda s: {'input': x, 'input_lengths': np.array([x.shape[1]], np.int64),
                      'scales': np.array(s, np.float32)}
    a = original.run(None, feed([0.667, 1.0, 1.0]))[0]
    b = patched.run(None, feed([0.667, 1.0, 1.0]))[0]
    assert a.shape == b.shape and np.abs(a - b).max() < 1e-4, (a.shape, b.shape)
    # and the two scales now act: no noise at all is deterministic
    quiet = [patched.run(None, feed([0.0, 1.0, 0.0]))[0] for _ in range(2)]
    assert np.array_equal(*quiet) and not np.array_equal(quiet[0][..., :a.shape[-1]], a[..., :quiet[0].shape[-1]])
    print('same audio at the old values, scales now used:', path)


if __name__ == '__main__':
    if sys.argv[1] == '--check':
        check(sys.argv[2])
    else:
        onnx.save(rewrite(onnx.load(sys.argv[1])), sys.argv[2])
        print('wrote', sys.argv[2])
