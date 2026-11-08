from mcap.reader import make_reader
import struct,json
from pathlib import Path
p=Path('drone_logs/flight_logs/20261107-201744_bag/bag/bag_0.recovered.mcap')
rows=[]
with p.open('rb') as f:
 for schema,ch,m in make_reader(f).iter_messages(topics=['/rosout']):
  d=m.data; endian='<' if d[1]&1 else '>'; pos=16
  def string():
   global pos
   pos=4+((pos-4+3)//4)*4
   n=struct.unpack_from(endian+'I',d,pos)[0];pos+=4
   s=d[pos:pos+n-1].decode(errors='replace');pos+=n;return s
  name=string();msg=string()
  if name=='refnav' or ('depth frames' in msg) or ('scan fit' in msg):rows.append({'t':m.log_time/1e9,'node':name,'message':msg})
out=Path('deploy-blocked-recovery/docs/evidence/blocked_recovery_20261108/real_bag_range_logs.json');out.write_text(json.dumps({'bag':str(p),'rows':rows},indent=2)+'\n')
for x in rows:
 if any(w in x['message'] for w in ['blocked','land','depth','observed','mission done']):print(x['node'],x['message'])
