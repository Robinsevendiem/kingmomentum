import tushare as ts
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# Configuration
TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()
if not TOKEN:
    raise ValueError("请先设置环境变量 TUSHARE_TOKEN，再运行该脚本。")

ts.set_token(TOKEN)
pro = ts.pro_api()

etfs = [
    {'code': '513520.SH', 'name': '日经ETF', 'start_date': '20190612'},
    {'code': '513100.SH', 'name': '纳指ETF', 'start_date': '20130515'},
    {'code': '513020.SH', 'name': '港股科技ETF', 'start_date': '20220127'},
    {'code': '510180.SH', 'name': '180ETF', 'start_date': '20060518'},
    {'code': '588120.SH', 'name': '科创板ETF', 'start_date': '20230908'},
    {'code': '159915.SZ', 'name': '创业板ETF', 'start_date': '20111209'},
    {'code': '501018.SH', 'name': '南方原油(LOF)', 'start_date': '20160624'},
    {'code': '518880.SH', 'name': '黄金ETF', 'start_date': '20130729'},
    {'code': '511090.SH', 'name': '30年国债ETF', 'start_date': '20230613'},
]

report_lines = []
today = datetime.now().strftime('%Y%m%d')

def log_report(msg):
    print(msg)
    report_lines.append(msg)

def fetch_chunked_data(code, start_date, end_date):
    """
    Fetch data in chunks to avoid API limits.
    """
    all_data = []
    
    # Convert dates
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    # Chunk size in days (e.g., 365 days * 2 = 730 days ~ 2 years)
    # Tushare often limits to a few thousand rows. 2 years of trading days ~500 rows. Safe.
    chunk_days = 700 
    
    current_start = start_dt
    
    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)
        
        s_date = current_start.strftime('%Y%m%d')
        e_date = current_end.strftime('%Y%m%d')
        
        # log_report(f"    Fetching chunk: {s_date} to {e_date}...")
        
        try:
            # Fetch unadjusted
            df_raw = ts.pro_bar(ts_code=code, start_date=s_date, end_date=e_date, adj=None, asset='FD')
            
            # Fetch adjusted (qfq)
            df_adj = ts.pro_bar(ts_code=code, start_date=s_date, end_date=e_date, adj='qfq', asset='FD')
            
            if df_raw is not None and not df_raw.empty:
                # Process this chunk
                if df_adj is None or df_adj.empty:
                    df_adj = df_raw.copy()
                
                df_raw = df_raw.set_index('trade_date').sort_index()
                df_adj = df_adj.set_index('trade_date').sort_index()
                
                adj_cols = ['open', 'high', 'low', 'close']
                # Ensure columns exist before selecting
                existing_adj_cols = [c for c in adj_cols if c in df_adj.columns]
                
                df_adj_subset = df_adj[existing_adj_cols].rename(columns={c: f'adj_{c}' for c in existing_adj_cols})
                
                df_chunk = df_raw.join(df_adj_subset, how='left')
                df_chunk = df_chunk.reset_index()
                
                all_data.append(df_chunk)
            
            # Sleep briefly
            time.sleep(0.1)
            
        except Exception as e:
            log_report(f"    [ERROR] Failed chunk {s_date}-{e_date}: {e}")
        
        # Move to next chunk
        current_start = current_end + timedelta(days=1)
        
    if not all_data:
        return pd.DataFrame()
        
    # Concatenate all chunks
    final_df = pd.concat(all_data, ignore_index=True)
    # Remove duplicates just in case
    final_df.drop_duplicates(subset=['trade_date'], inplace=True)
    # Sort by date
    final_df.sort_values('trade_date', inplace=True)
    
    return final_df

log_report(f"Data Download Report - {today}")
log_report("========================================")

for etf in etfs:
    code = etf['code']
    name = etf['name']
    start_date = etf['start_date']
    
    log_report(f"\nProcessing {name} ({code})...")
    
    try:
        df_final = fetch_chunked_data(code, start_date, today)
        
        if not df_final.empty:
            min_date = df_final['trade_date'].min()
            max_date = df_final['trade_date'].max()
            count = len(df_final)
            
            log_report(f"  Data Range: {min_date} to {max_date}")
            log_report(f"  Total Records: {count}")
            
            req_start_dt = datetime.strptime(start_date, '%Y%m%d')
            real_start_dt = datetime.strptime(min_date, '%Y%m%d')
            
            diff_days = (real_start_dt - req_start_dt).days
            
            if diff_days > 20: # Buffer
                log_report(f"  [DISCREPANCY] Start date mismatch! Requested: {start_date}, Actual: {min_date}. Missing {diff_days} days.")
                log_report(f"  Reason: Tushare data might be missing for early days or listing date differs from trading start.")
            elif diff_days < -5:
                 log_report(f"  [INFO] Data starts earlier than requested? Actual: {min_date}")
            else:
                log_report(f"  [OK] Start date aligned.")
                
            filename = f"{code}_{name}_history.csv"
            df_final.to_csv(filename, index=False, encoding='utf-8-sig')
            log_report(f"  Saved to {filename}")
            
        else:
             log_report(f"  [ERROR] No data found for {code}.")

    except Exception as e:
        log_report(f"  [EXCEPTION] Failed to process {code}: {str(e)}")
    
    time.sleep(0.5)

with open('data_verification_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("\nProcessing complete. Check data_verification_report.txt for details.")
