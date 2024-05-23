import motmetrics as mm
import numpy as np

acc = mm.MOTAccumulator(auto_id=True)
acc.update(
    [1, 2],                     # Ground truth objects in this frame
    [1, 2, 3],                  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],     # Distances from object 1 to hypotheses 1, 2, 3
        [0.5,  0.2,   0.3]      # Distances from object 2 to hypotheses 1, 2, 3
    ]
)
print(acc.events)
print(acc.mot_events)
frameid = acc.update(
    [1, 2],
    [1],
    [
        [0.2],
        [0.4]
    ]
)
print(acc.mot_events.loc[frameid])
frameid = acc.update(
    [1, 2],
    [1, 3],
    [
        [0.6, 0.2],
        [0.1, 0.6]
    ]
)
print(acc.mot_events.loc[frameid])
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
print(summary)
summary = mh.compute_many(
    [acc, acc.events.loc[0:1]],
    metrics=['num_frames', 'mota', 'motp'],
    names=['full', 'part'])
print(summary)
strsummary = mm.io.render_summary(
    summary,
    formatters={'mota' : '{:.2%}'.format},
    namemap={'mota': 'MOTA', 'motp' : 'MOTP'}
)
print(strsummary)
summary = mh.compute_many(
    [acc, acc.events.loc[0:1]],
    metrics=mm.metrics.motchallenge_metrics,
    names=['full', 'part'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)