# USD Pair LP Optimizer & Backtester

This tool analyzes historical data for specific cryptocurrency pools (USD-pegged pairs) on decentralized exchanges (like Shadow Finance). It helps identify top-performing pools based on past liquidity and volume, and backtests different Liquidity Provider (LP) strategies to estimate potential past performance.

## Features

- Fetches top USD-paired pools from supported exchanges (e.g., Shadow).
- Filters out pairs that aren't between two stablecoins.
- Loads historical daily data (Price, Total Value Locked (TVL), Trading Volume, Fees collected) for selected pools.
- Backtests multiple LP strategies (defined by position width and rebalancing rules) using the historical data.
- Calculates key performance metrics (like daily price volatility and rebalance frequency).
- Determines average daily TVL and Volume for the backtest period.
- Generates plots showing daily price changes for each backtested pool.
- Outputs a summary table highlighting the best results found.

## Setup

Follow these steps to get the tool running on your computer:

1.  **Clone the repository:** Download the project code. If you have `git` installed, use:
    ```bash
    git clone <your-repo-url> # Replace <your-repo-url> with the actual URL
    cd lp-optimizer
    ```
    Otherwise, download the code ZIP file and extract it.
2.  **Create and activate a virtual environment:** This isolates the project's Python packages from others on your system. Open your terminal in the `lp-optimizer` directory and run:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    # On Windows Command Prompt use: .venv\Scripts\activate.bat
    # On Windows PowerShell use: .venv\Scripts\Activate.ps1
    ```
    You should see `(.venv)` appear at the start of your terminal prompt.
3.  **Install dependencies:** Install the required Python packages into the active virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional) Configure API Key:**
    If you have a TheGraph API key (useful for fetching lots of data reliably), you need to make it available to the script as an environment variable named `GRAPH_API_KEY`. How to set environment variables depends on your operating system (search online for "set environment variable [your OS]").

## Usage

Run the script from your terminal, **making sure you are in the project's root directory (`lp-optimizer`)** and your virtual environment is activated (`(.venv)` should be visible).

The main command format is: `python -m src.main [COMMAND] [ARGUMENTS]`

**Main Command (Primary Focus):**

*(Note: While other subcommands might exist from previous versions, `top-usd-pairs` is the current primary command.)*

-   `top-usd-pairs`: Finds top USD pairs on an exchange, runs backtests, and shows results.

**Common Arguments (Options you can add after `top-usd-pairs`):**

-   `--exchange [NAME]`: Specify the exchange name (e.g., `shadow`). **Required.**
-   `--days [NUMBER]`: How many past days of data to analyze (e.g., `90`). Default is 90.
-   `--investment [AMOUNT]`: Simulated investment size for calculations (e.g., `1000000` for $1M). Default is 1,000,000.
-   `--min-liquidity [AMOUNT]`: Ignores pools with less Total Value Locked (TVL) than this. Default: 100,000.
-   `--min-volume [AMOUNT]`: Ignores pools with less average daily trade volume than this. Default: 10,000.
-   `--top-n [NUMBER]`: Limits the analysis to the top N pools found. Default: 10.
-   `--plot`: Add this flag to generate and save price change plots to the `output/` folder.
-   `--fee-tier-override [OVERRIDES]`: Manually set fee percentages for specific pools (e.g., `0xPoolAddr:0.05,0xOtherPool:0.01`).

**Example:**

Analyze top Shadow exchange pools over the last 90 days with a $1M simulation, generating plots:

```bash
python -m src.main top-usd-pairs --exchange shadow --days 90 --investment 1000000 --plot
```

## Shadow USD* 90-Day Volatility Tracking

Below are the 90-day volatility plots and average daily statistics for the top 5 pools analyzed.

---

### 1. USDC.e / USDT

![USDC.e/USDT 90d Volatility Plot](output/USDC.e-USDT_volatility.png)

- **Avg Daily TVL:** $4.52M
- **Avg Daily Volume:** $4.77M

---

### 2. wstkscUSD / scUSD

![wstkscUSD/scUSD 90d Volatility Plot](output/wstkscUSD-scUSD_volatility.png)

- **Avg Daily TVL:** $155K
- **Avg Daily Volume:** $276K

---

### 3. USDC.e / scUSD

![USDC.e/scUSD 90d Volatility Plot](output/USDC.e-scUSD_volatility.png)

- **Avg Daily TVL:** $5.87M
- **Avg Daily Volume:** $4.15M

---

### 4. USDC.e / xUSD

![USDC.e/xUSD 90d Volatility Plot](output/USDC.e-xUSD_volatility.png)

- **Avg Daily TVL:** $572K
- **Avg Daily Volume:** $80K

---

### 5. USDC.e / EURC.e

![USDC.e/EURC.e 90d Volatility Plot](output/USDC.e-EURC.e_volatility.png)

- **Avg Daily TVL:** $131K
- **Avg Daily Volume:** $168K

---

## Project Structure

```
lp-optimizer/
├── data/                   # Directory for potential future data storage
├── output/                 # Default output for plots/logs
├── src/
│   ├── api/                # Subgraph API clients
│   │   ├── __init__.py
│   │   └── subgraph_client.py
│   ├── data/               # Data loading module
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── simulation/         # Backtesting engine and results
│   │   ├── __init__.py
│   │   ├── backtest_engine.py
│   │   ├── config.py
│   │   └── result_processor.py
│   ├── utils/              # Helper utilities
│   │   ├── __init__.py
│   │   └── helpers.py
│   ├── __init__.py
│   └── main.py             # CLI entry point
├── .env                    # Local environment variables (ignored by git)
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore configuration
├── requirements.txt        # Python dependencies
└── README.md
```