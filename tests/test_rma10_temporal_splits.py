import pandas as pd
from scripts.rma10_validation import chronological_event_split, rolling_origin_event_splits


def synthetic():
    rows=[]
    # Tracks are independent. Each has 120 rows across one five-minute interval.
    for day in range(12):
        for track in range(3):
            tid=f'd{day}-t{track}'
            for i in range(120):
                rows.append({'track_id':tid,'frame_utc':pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(days=day,minutes=i*5),'horizonte_min':5})
    return pd.DataFrame(rows)


def test_rolling_folds_are_strictly_chronological_and_keep_tracks_whole():
    df=synthetic(); folds=rolling_origin_event_splits(df,n_last_events=4,min_rows=100)
    assert len(folds)>=2
    for f in folds:
        tr=set(df.loc[f['train_mask'],'track_id']); ca=set(df.loc[f['calibration_mask'],'track_id']); te=set(df.loc[f['test_mask'],'track_id'])
        assert not tr&ca and not tr&te and not ca&te
        assert df.loc[f['train_mask'],'frame_utc'].max() < df.loc[f['calibration_mask'],'frame_utc'].min()
        assert df.loc[f['calibration_mask'],'frame_utc'].max() < df.loc[f['test_mask'],'frame_utc'].min()


def test_chronological_split_has_no_temporal_leakage():
    df=synthetic(); s=chronological_event_split(df)
    tr=df.loc[s['train_mask']]; ca=df.loc[s['calibration_mask']]; te=df.loc[s['test_mask']]
    assert tr.frame_utc.max() < ca.frame_utc.min()
    assert ca.frame_utc.max() < te.frame_utc.min()
    assert not (set(tr.track_id)&set(ca.track_id))
    assert not (set(ca.track_id)&set(te.track_id))
