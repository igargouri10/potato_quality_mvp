"""Generate public-safe synthetic PDF/QC fixtures, not employer records."""
from pathlib import Path
import fitz
import pandas as pd

def generate():
    out=Path(__file__).resolve().parents[1]/'demo_data'
    out.mkdir(exist_ok=True)
    rows=[]
    for batch,tier,score,defects in [('1042',1,85,2),('1043',3,65,6),('1044',4,50,15)]:
        rows.append({'Batch':batch,'PO Number':'900001','PO Line Number':'10','PLANT NAME':'DEMO PLANT','Supplier Site Label':'SYNTHETIC FARM','TIER':tier,'ASAVG':score,'Total Defects':defects,'GREENING':0})
        doc=fitz.open(); page=doc.new_page()
        page.insert_text((50,60),f'SYNTHETIC DEMO - no company data\nPurchase Order: 900001\nLine: 10\nBatch Number: {batch}\nSite: DEMO PLANT\nExpected Need By Date: 01-Jan-26',fontsize=14)
        doc.save(out/f'batch_{batch}.pdf'); doc.close()
    pd.DataFrame(rows).to_csv(out/'synthetic_qc.csv',index=False)
    return out

if __name__=='__main__':
    print(generate())
