"""Decode scalar MATLAB strings in this assignment's v7.3 MCOS layout.

This decoder is deliberately limited to the observed three-file representation.
It validates handles, string lengths, barcode format and channel identifiers.
"""
from pathlib import Path
import argparse,json,re
import h5py,numpy as np,pandas as pd

HERE=Path(__file__).resolve().parent

def decode(f,ref):
    handle=f[ref][()].ravel()
    if handle.dtype!=np.dtype('uint32') or len(handle)!=6 or int(handle[0])!=3707764736 or list(handle[1:4])!=[2,1,1] or int(handle[5])!=1:
        raise ValueError('Unsupported scalar MATLAB string handle')
    refs=f['#subsystem#/MCOS'][()].ravel();index=int(handle[4])+1
    if not 2<=index<len(refs)-1:raise ValueError('Invalid MCOS object index')
    payload=f[refs[index]][()].ravel()
    if payload.dtype!=np.dtype('uint64') or len(payload)<6 or list(payload[:4])!=[1,2,1,1]:
        raise ValueError('Unsupported string payload')
    n=int(payload[4]);raw=payload[5:].astype('<u8').tobytes()
    if 2*n>len(raw):raise ValueError('Invalid UTF-16 string length')
    if any(raw[2*n:]):raise ValueError('Nonzero padding')
    return raw[:2*n].decode('utf-16-le'),int(handle[4])

def run(raw_dir):
    rows=[];b3_capacity=[]
    for batch,date in enumerate(['2017-05-12','2018-02-20','2018-04-12'],1):
        matches=list(raw_dir.glob(date+'*.mat'))
        if len(matches)!=1:raise ValueError('Expected one MAT per batch')
        with h5py.File(matches[0],'r') as f:
            b=f['batch'];object_ids=[]
            for i in range(b['barcode'].size):
                barcode,obj=decode(f,b['barcode'][i,0]);channel,channel_obj=decode(f,b['channel_id'][i,0])
                if not re.fullmatch(r'(?i)EL\d{12}',barcode) or not channel.isdigit():raise ValueError('Invalid identifier')
                # Two disjoint scalar object ranges also validate field-to-object mapping.
                assert obj==i+1 and channel_obj==b['barcode'].size+i+1
                assert 1<=int(channel)<=48
                rows.append({'cell_id':f'b{batch}c{i}','batch':batch,'barcode':barcode.upper(),'channel_id':int(channel),'barcode_object_id':obj})
                if batch==3:
                    s=f[b['summary'][i,0]]
                    b3_capacity.append({'cell_id':f'b3c{i}','last_QD':float(s['QDischarge'][()].ravel()[-1])})
    d=pd.DataFrame(rows);assert len(d)==139 and d.cell_id.is_unique
    for _,g in d.groupby('batch'):assert g.barcode.is_unique and g.channel_id.is_unique
    duplicates=d[d.barcode.duplicated(keep=False)].sort_values('barcode')
    d.to_csv(HERE/'results/cell_identity.csv',index=False)
    duplicates.to_csv(HERE/'results/duplicate_barcodes.csv',index=False)
    report={'cells_decoded':len(d),'unique_normalized_barcodes':d.barcode.nunique(),
            'duplicate_barcode_cells':len(duplicates),'cross_batch_duplicate_barcodes':int((d.groupby('barcode').batch.nunique()>1).sum()),
            'B3_channel46_cell_ids':d[(d.batch==3)&(d.channel_id==46)].cell_id.tolist(),
            'scope':'Barcode metadata identity; not an audit of physical chain of custody'}
    (HERE/'results/identity_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    # Map the published sequential B3 rules without changing the 44-cell main score.
    quality=pd.DataFrame(b3_capacity).merge(d[d.batch==3][['cell_id','channel_id']],on='cell_id',validate='one_to_one')
    quality['exclusion_reason']=''
    assert quality.iloc[37].channel_id==46
    quality.loc[37,'exclusion_reason']='published channel46 exclusion'
    remaining=quality[quality.exclusion_reason=='']
    end=remaining.index[remaining.last_QD>.885]
    quality.loc[end,'exclusion_reason']='published end capacity >0.885Ah'
    remaining=quality[quality.exclusion_reason=='']
    noisy=remaining.iloc[[2,39,40]].index
    quality.loc[noisy,'exclusion_reason']='published sequential indices 3,40,41'
    quality['retained']=quality.exclusion_reason==''
    assert quality.retained.sum()==40
    quality.to_csv(HERE/'results/B3_quality_cohort.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-dir',type=Path,required=True);run(p.parse_args().raw_dir)
