import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob

# File paths
RECORD_DIR = 'record'
POSITION_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26持仓记录.csv')
# We need to find the history files in the root directory
ROOT_DIR = '.'

def load_data():
    # Load position record
    # The file has a header on the first line, but pandas might misinterpret if not careful.
    # The content I saw earlier: "1→,日期,ETF名称,净值,涨跌幅,累计收益"
    # It seems to be a standard CSV.
    df_pos = pd.read_csv(POSITION_FILE)
    
    # Parse dates
    df_pos['日期'] = pd.to_datetime(df_pos['日期'])
    
    # Create a dictionary of date -> held asset
    # Filter for valid assets (ignore '现金' if we only care about the 9 assets)
    # The user said "9 targets".
    holding_map = df_pos.set_index('日期')['ETF名称'].to_dict()
    
    return holding_map

def get_history_files():
    # Map asset names to filenames
    # Asset names in record: 创业板, 南方原油, 上证180, 30年国债, 港股科技, 纳指100, 日经ETF, 黄金ETF, 科创板
    # Filenames: *创业板ETF*, *南方原油*, *180ETF*, *30年国债ETF*, *港股科技ETF*, *纳指ETF*, *日经ETF*, *黄金ETF*, *科创板ETF*
    
    mapping = {
        '创业板': '159915.SZ_创业板ETF_history.csv',
        '南方原油': '501018.SH_南方原油(LOF)_history.csv',
        '上证180': '510180.SH_180ETF_history.csv',
        '30年国债': '511090.SH_30年国债ETF_history.csv',
        '港股科技': '513020.SH_港股科技ETF_history.csv',
        '纳指100': '513100.SH_纳指ETF_history.csv',
        '日经ETF': '513520.SH_日经ETF_history.csv',
        '黄金ETF': '518880.SH_黄金ETF_history.csv',
        '科创板': '588120.SH_科创板ETF_history.csv'
    }
    
    # Verify files exist
    valid_mapping = {}
    for name, filename in mapping.items():
        if os.path.exists(filename):
            valid_mapping[name] = filename
        else:
            print(f"Warning: File {filename} not found for {name}")
            
    return valid_mapping

def create_visualizations():
    holding_map = load_data()
    asset_files = get_history_files()
    
    # Create a subplot for each asset
    # 9 assets, maybe 3 columns x 3 rows? Or just one big column of 9 charts?
    # User said "Each target one chart". 
    # Let's do a single column of 9 charts to make them large enough.
    # Or separate HTML files? A single file is easier to deliver.
    
    figures = []
    
    for asset_name, filename in asset_files.items():
        print(f"Processing {asset_name}...")
        df_hist = pd.read_csv(filename)
        
        # Parse history dates
        # History format: 20111209 (int or str)
        df_hist['trade_date'] = pd.to_datetime(df_hist['trade_date'], format='%Y%m%d')
        df_hist = df_hist.sort_values('trade_date')
        
        # Filter range (2017-08-10 to 2026-02-26) to match the record scope
        start_date = pd.Timestamp('2017-08-10')
        end_date = pd.Timestamp('2026-02-26')
        mask = (df_hist['trade_date'] >= start_date) & (df_hist['trade_date'] <= end_date)
        df_hist = df_hist.loc[mask].copy()
        
        if df_hist.empty:
            print(f"No data for {asset_name} in range.")
            continue
            
        # Determine holding status for each day in history
        # We need to map history dates to holding status
        # holding_map keys are timestamps.
        
        # Add a 'held' column
        # Note: holding_map might have dates that are not in df_hist (e.g. weekends/holidays if record has them, or gaps)
        # But we only care about trading days in history.
        
        # Efficient way:
        # Check if asset_name is in holding_map for that date
        
        # Prepare lists for plotting
        dates = df_hist['trade_date'].tolist()
        prices = df_hist['close'].tolist() # Use 'close' or 'adj_close'? 'close' is usually fine for visualization unless splits happened. 'adj_close' is better for returns. Let's use 'close' as it's the visible price.
        
        # Identify holding segments
        # We want to plot the "Not Held" as dim (grey) and "Held" as bright (red/blue).
        # We can plot the whole line as Grey.
        # Then overlay Red segments.
        
        fig = go.Figure()
        
        # 1. Plot the full history in dim color
        fig.add_trace(go.Scatter(
            x=dates, 
            y=prices,
            mode='lines',
            name=f'{asset_name} (Not Held)',
            line=dict(color='lightgrey', width=2),
            hoverinfo='skip' # Don't show hover for the background line if we have overlay
        ))
        
        # 2. Identify and plot held segments
        # We need to group consecutive held days into segments
        # A segment is a continuous sequence of days where the asset was held.
        # If held on Day i and Day i+1, the line i->i+1 is held.
        
        held_indices = []
        for i, date in enumerate(dates):
            # Check if held on this date
            # We need to be careful with timestamp matching (hours/minutes)
            # holding_map keys have 00:00:00. df_hist dates also likely normalized.
            d = date.normalize() # Ensure 00:00:00
            if holding_map.get(d) == asset_name:
                held_indices.append(i)
        
        # Now find contiguous segments in held_indices
        # If we have [1, 2, 3, 5, 6], segments are [1,2,3] and [5,6].
        # For a segment [1,2,3], we plot line from index 1 to 3.
        # Wait, if held on Day 1, 2, 3.
        # Does it mean we held FROM Day 1 to Day 3?
        # If record says held on Day 1, we held it at Close of Day 1.
        # If record says held on Day 2, we held it at Close of Day 2.
        # So the return Day 1 -> Day 2 is captured.
        # So we should connect Day 1 to Day 2.
        # So for indices [1, 2, 3], we plot line 1-2-3.
        
        if held_indices:
            segments = []
            if not held_indices:
                pass
            else:
                current_segment = [held_indices[0]]
                for i in range(1, len(held_indices)):
                    if held_indices[i] == held_indices[i-1] + 1:
                        current_segment.append(held_indices[i])
                    else:
                        segments.append(current_segment)
                        current_segment = [held_indices[i]]
                segments.append(current_segment)
            
            # Create a single trace for all segments using None to break lines
            # This is more efficient than many traces
            held_x = []
            held_y = []
            
            for segment in segments:
                # We need at least 2 points to draw a line
                # If a segment is just 1 point (held for 1 day), we might want to show a marker
                # But usually holding is > 1 day.
                # If just 1 point, line won't show.
                # But if we bought on Day X and sold on Day X+1 (held 1 day), we have Day X and Day X+1 in indices?
                # If we sell on Day X+1 Open, Day X+1 is NOT held in the record (record is EOD).
                # So we only have Day X in indices.
                # So we have a single point.
                # We can add markers to the held trace to show single points.
                
                seg_dates = [dates[j] for j in segment]
                seg_prices = [prices[j] for j in segment]
                
                held_x.extend(seg_dates)
                held_y.extend(seg_prices)
                held_x.append(None) # Break line
                held_y.append(None)
                
            fig.add_trace(go.Scatter(
                x=held_x,
                y=held_y,
                mode='lines+markers', # Markers ensure single days are visible
                name=f'{asset_name} (Held)',
                line=dict(color='red', width=3),
                marker=dict(size=4),
                hovertemplate='%{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>'
            ))
            
            # Update the grey trace to also show hover info so user can see prices when not held
            fig.data[0].hoverinfo = 'all'
            fig.data[0].hovertemplate = '%{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>'

        # Add title and layout
        fig.update_layout(
            title=f'{asset_name} Price & Holding Periods',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white'
        )
        
        figures.append(fig)

    # Save to HTML
    output_file = 'strategy_visualization_new.html'
    with open(output_file, 'w') as f:
        f.write('<html><head><title>Strategy Visualization</title></head><body>')
        f.write('<h1>Trading Strategy Visualization</h1>')
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write('</body></html>')
    
    print(f"Visualization saved to {output_file}")

if __name__ == '__main__':
    create_visualizations()
