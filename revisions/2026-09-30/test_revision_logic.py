import numpy as np
from analyse import policy

# A premature alarm does not determine correct lead time; the later in-window
# emitted alert does. In another stay, a premature alarm suppresses the only
# in-window opportunity, so that event must count as missed.
a={'record_id':np.array([1,1,1,2,2,3,3]),'index_hour':np.array([0,7,8,0,5,0,7]),
   'y':np.array([0,1,1,0,1,0,0]),'lead_time_hours':np.array([np.nan,3.,2.,np.nan,2.,np.nan,np.nan])}
p=np.array([.9,.1,.9,.9,.9,.9,.1])
r=policy(a,np.ones(7,bool),p,.5)
assert r['n_event_stays']==2 and r['n_detected_stays']==1
assert r['event_sensitivity']==.5 and r['lead_median']==2 and r['lead_max']==2
assert r['event_free_stays_false_alert_percent']==100
assert np.isclose(r['false_episodes_per_alerted_stay'],1)
print('PASS: no credit for premature alarms, correct lead time, suppression-induced miss, and false-alert burden.')
