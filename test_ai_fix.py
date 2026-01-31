from src.core.ai_ensemble_strategy_v2 import AIEnsembleStrategyV2
from src.core.data_store import get_data_store

store = get_data_store()
ai = AIEnsembleStrategyV2()

print('Training...')
ai.train_ensemble(store.available_tickers[:20], '2005-01-01', '2014-12-31', use_cache=True)

print('\nGenerating signals...')
signals = ai.generate_signals(store.available_tickers[:20], '2015-01-01', '2015-12-31')

print(f'\nSignals generated: {len(signals)}')
if len(signals) > 0:
    print(f'LONG: {(signals["signal"] == 1).sum()}')
    print(f'SHORT: {(signals["signal"] == -1).sum()}')
    print('\n✅ SUCCESS - AI signals are now being generated!')
else:
    print('\n❌ Still no signals')
