# Tiingo Cache Information Folder

This folder contains metadata and documentation for the Tiingo data cache.

## 📁 Folder Structure

```
src/data/cache/tiingo/
├── info/
│   ├── lists/              📋 Ticker lists for scripts
│   ├── documentation/      📖 Detailed documentation
│   ├── logs/              📝 Download and processing logs
│   └── README.md           📄 This file
├── *.csv                   📊 Actual data files
└── ...                    📁 Other cache files
```

## 📋 Files

### 📋 Lists Folder (`lists/`)
- **`tickers_to_download.txt`** - Complete list of 80 tickers for download
- **Purpose**: Clean ticker symbols for script processing
- **Format**: One ticker per line, organized by category
- **Usage**: Download scripts, batch processing

### 📖 Documentation Folder (`documentation/`)
- **`TICKER_LIST_COMPLETE.md`** - Complete ticker documentation
- **Purpose**: Human-readable descriptions and details
- **Format**: Markdown with categories, descriptions, and priority lists
- **Usage**: Reference, planning, understanding tickers

### 📝 Logs Folder (`logs/`)
- **Purpose**: Download logs, error reports, processing history
- **Usage**: Troubleshooting, audit trail, performance analysis

## 🎯 Usage

### For Scripts
```python
# Read ticker list
with open('src/data/cache/tiingo/info/lists/tickers_to_download.txt', 'r') as f:
    tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
```

### For Humans
- Read documentation in `documentation/TICKER_LIST_COMPLETE.md`
- Check logs in `logs/` for download status
- Use lists in `lists/` for script input

## 🔄 Maintenance

### Adding New Tickers
1. Update `lists/tickers_to_download.txt`
2. Update `documentation/TICKER_LIST_COMPLETE.md`
3. Run download script
4. Verify new data files

### Updating Documentation
- Keep both files in sync
- Update version numbers
- Add new categories as needed

## 📊 Current Status

- **Total Tickers**: 80
- **Categories**: 9 (TECH, FINANCE, HEALTHCARE, CONSUMER, ENERGY, INDUSTRIAL, ETFs, CRYPTO, OTHER)
- **Priority Tickers**: 13 (benchmarks, tech giants, financial services, crypto)
- **Last Updated**: January 28, 2026

---

*This folder keeps all cache-related metadata organized and accessible.*
