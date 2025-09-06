import pyBKT
import pandas as pd
from pyBKT.models import Model
model = Model(seed = 42, num_fits = 1)

df = pd.read_csv('d_afm_stoich.csv')
df = df.rename(columns={'outcome': 'correct'})
df = df.rename(columns={'kc_default': 'skill_name'})

model.fit(data=df)

print('Fitting...')
model.save('chem-bkt.p')
model.params().reset_index().to_csv('chem-bkt.csv', index=False)
print('Done!')
preds = model.predict(data=df)
preds.to_csv('chem-preds.csv', index=False)
