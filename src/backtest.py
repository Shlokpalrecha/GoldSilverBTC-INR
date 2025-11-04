# src/backtest.py
def backtest_pairs(df, entry_z=2.0, exit_z=0.5):
    """Simple backtest for mean reversion strategy."""
    df = df.copy()
    df['Position'] = 0
    df.loc[df['Zscore'] > entry_z, 'Position'] = -1
    df.loc[df['Zscore'] < -entry_z, 'Position'] = 1
    df.loc[df['Zscore'].abs() < exit_z, 'Position'] = 0

    df['PnL'] = df['Position'].shift(1) * df['Spread'].diff()
    df['CumulativeReturn'] = df['PnL'].cumsum()
    return df
