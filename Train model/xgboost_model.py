import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import timedelta
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)

import xgboost as xgb

import matplotlib.pyplot as plt


# -----------------------------
# Walk-forward: expanding, non-overlapping test windows
# -----------------------------
def expanding_non_overlapping_validation(df, feature_cols, n_splits=4, min_train_size=252):
    """Create non-overlapping expanding walk-forward splits.
    Each fold: train = [0:train_end), test = [train_end:train_end+test_size)
    test_size chosen so that last fold ends at end of dataset.
    Returns metrics aggregated across folds and predictions per fold.
    """
    N = len(df)
    # Ensure we have enough data
    if N < min_train_size * 2:
        raise ValueError('Not enough data for walk-forward')

    # Determine test size by dividing remaining portion after min_train_size
    remaining = N - min_train_size
    test_size = max(1, remaining // n_splits)

    folds = []
    train_start = 0
    train_end = min_train_size

    while train_end + test_size <= N:
        test_start = train_end
        test_end = train_end + test_size
        folds.append((train_start, train_end, test_start, test_end))
        # expand train to include the just-tested period
        train_end = test_end

    metrics_clf = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
    metrics_reg = {'mae':[], 'rmse':[], 'r2':[], 'direction_acc':[]}
    fold_preds = []

    for i,(ts,te,ss,se) in enumerate(folds,1):
        train = df.iloc[ts:te].dropna()
        test = df.iloc[ss:se].dropna()
        if len(train) < min_train_size or len(test) == 0:
            continue

        X_train = train[feature_cols]
        y_train_clf = train['Next_Direction']
        y_train_reg = train['Next_Return']

        X_test = test[feature_cols]
        y_test_clf = test['Next_Direction']
        y_test_reg = test['Next_Return']

        # Scaling only on train
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train models
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        reg = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', n_jobs=-1)

        clf.fit(X_train_s, y_train_clf)
        reg.fit(X_train_s, y_train_reg)

        yhat_clf = clf.predict(X_test_s)
        yhat_proba = clf.predict_proba(X_test_s)[:,1] if hasattr(clf,'predict_proba') else None
        yhat_reg = reg.predict(X_test_s)

        # Store fold metrics
        metrics_clf['accuracy'].append(accuracy_score(y_test_clf, yhat_clf))
        metrics_clf['precision'].append(precision_score(y_test_clf, yhat_clf, zero_division=0))
        metrics_clf['recall'].append(recall_score(y_test_clf, yhat_clf, zero_division=0))
        metrics_clf['f1'].append(f1_score(y_test_clf, yhat_clf, zero_division=0))

        mae = mean_absolute_error(y_test_reg, yhat_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, yhat_reg))
        r2 = r2_score(y_test_reg, yhat_reg)
        dir_acc = np.mean((yhat_reg > 0) == (y_test_reg > 0))

        metrics_reg['mae'].append(mae)
        metrics_reg['rmse'].append(rmse)
        metrics_reg['r2'].append(r2)
        metrics_reg['direction_acc'].append(dir_acc)

        fold_preds.append({'index': X_test.index, 'clf': yhat_clf, 'proba': yhat_proba, 'reg': yhat_reg})

        print(f"--- Fold {i}/{len(folds)} ---")
        print(f"Train: {ts} to {te-1}, Test: {ss} to {se-1}, Test rows: {len(test)}")
        print(f"Classification Acc: {metrics_clf['accuracy'][-1]:.4f}, Direction Acc (reg): {dir_acc:.4f}")

    return metrics_clf, metrics_reg, fold_preds

# -----------------------------
# BACKTEST: realistic execution
# - Execute trades at NEXT OPEN after signal is produced
# - Only long strategy (buy/flat). SELL = flat (no short)
# - Apply transaction cost only when entering/exiting
# - Compound equity properly
# -----------------------------

def backtest_realistic(df, signals, entry_delay='next_open', transaction_cost=0.001):
    """df must be indexed by date and contain 'open' and 'close' and 'return'.
    signals is aligned to df.index and indicates model decision at end of day t using info up to t (i.e. predict t+1):
      1 -> buy at next open (enter long at open_{t+1})
      -1 or 0 -> exit/flat at next open
    We will simulate using next day's open and next day's close returns.
    """
    df = df.copy()
    signals = np.asarray(signals)

    equity = 1.0
    position = 0
    entry_price = np.nan
    trades = []
    daily_equity = []

    trade_returns = []
    daily_returns = [0.0]

    # We'll iterate by row index i representing day t; signal at t indicates action for t+1
    for i in range(len(df)-1):
        sig = signals[i]  # decision made at end of day i, executed at open of day i+1
        next_open = df['open'].iat[i+1]
        next_close = df['close'].iat[i+1]
        daily_strategy_return = 0
        # If signal indicates buy and we are flat -> enter at next_open
        if sig == 1 and position == 0:
            entry_price = next_open * (1 + transaction_cost)  # include slippage/fee on entry
            position = 1
            trades.append({'type':'BUY','date':df.index[i+1],'price':next_open})
            daily_strategy_return = -transaction_cost
        # If signal indicates exit (0 or -1) and we are long -> exit at next_open
        elif (sig == 0 or sig == -1) and position == 1:
            exit_price = next_open * (1 - transaction_cost)
            trade_ret = (exit_price - entry_price) / entry_price
            trade_returns.append(trade_ret)
            equity *= (1 + trade_ret)
            trades.append({'type':'SELL','date':df.index[i+1],'price':next_open,'ret':trade_ret})
            daily_strategy_return = trade_ret
            position = 0
            entry_price = np.nan
        # If holding position at day i+1, mark equity to reflect that day's return from open->close (approx)
        if position == 1:
            intraday_ret = (next_close - next_open) / entry_price  # relative to entry
            daily_strategy_return = intraday_ret
            # Conservative approach: update equity multiplicatively by that day's return (approx)
            equity *= (1 + (next_close - next_open) / max(entry_price, 1e-9))
        daily_returns.append(daily_strategy_return)
        daily_equity.append(equity)

    # If still holding at the end, close at last close (apply exit cost)
    if position == 1:
        last_price = df['close'].iat[-1] * (1 - transaction_cost)
        trade_ret = (last_price - entry_price) / entry_price
        equity *= (1 + trade_ret)
        trades.append({'type':'SELL','date':df.index[-1],'price':df['close'].iat[-1],'ret':trade_ret})
        trade_returns.append(trade_ret)
        position = 0
        daily_returns.append(trade_ret)
        daily_equity.append(equity)

    daily_returns = np.array(daily_returns[1:])
    total_return = equity - 1
    buy_hold_return = df['close'].iat[-1] / df['close'].iat[0] - 1

    completed_trades = [t for t in trades if t['type'] in ['SELL']]
    win_rate = np.mean([t['ret'] > 0 for t in completed_trades]) if completed_trades else 0

    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(completed_trades),
        'trades': trades,
        'daily_equity': daily_equity
    }

# -----------------------------
# MAIN
# -----------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='Dataset/Cleaned/GOOG_cleaned.csv')
    parser.add_argument('--save_models', action='store_true')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.source)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

        print(f"✓ Loaded {len(df)} rows of data")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

        excluded = ['Next_Return','Next_Direction','symbol']
        feature_cols = [c for c in df.columns if c not in excluded]

        # Expanding non-overlapping validation
        print('[1/3] Running expanding non-overlapping walk-forward validation...')
        clf_metrics, reg_metrics, fold_preds = expanding_non_overlapping_validation(df, feature_cols, n_splits=4, min_train_size=252)

        print('Cross-validation results (expanding folds):')
        for k,v in clf_metrics.items():
            print(f"Classifier {k}: {np.mean(v):.4f} (+/- {np.std(v):.4f})")
        for k,v in reg_metrics.items():
            print(f"Regressor {k}: {np.mean(v):.4f} (+/- {np.std(v):.4f})")

        # Train final models on train portion (80%) and test on last 20%
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx].dropna()
        test = df.iloc[split_idx:].dropna()

        X_train = train[feature_cols]
        y_train_clf = train['Next_Direction']
        y_train_reg = train['Next_Return']
        X_test = test[feature_cols]

        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        reg = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', n_jobs=-1)

        print('[2/3] Training final models on 80% train...')
        clf.fit(X_train_s, y_train_clf)
        reg.fit(X_train_s, y_train_reg)
        print('✓ Models trained successfully')

        # Feature importance
        fi = pd.DataFrame({'feature': feature_cols, 'importance': clf.feature_importances_}).sort_values('importance',ascending=False)
        print('Top 15 features:')
        print(fi.head(15).to_string(index=False))

        # Build signals on test set using threshold from train
        test_proba = clf.predict_proba(X_test_s)[:,1]
        train_proba = clf.predict_proba(X_train_s)[:,1]
        threshold = 0.45  # static threshold
        test_signals = np.where(test_proba >= threshold, 1, 0)  # 1=buy, 0=flat

        print('[3/3] Backtesting on held-out test set...')
        backtest_results = backtest_realistic(test, test_signals, transaction_cost=0.001)

        print('BACKTEST RESULTS (Test Set)')
        print(f"Strategy Return: {backtest_results['total_return']*100:+.2f}%")
        print(f"Buy & Hold Return: {backtest_results['buy_hold_return']*100:+.2f}%")
        print(f"Win Rate: {backtest_results['win_rate']*100:.2f}%")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Number of Trades: {backtest_results['num_trades']}")

        # Prediction for next day (last row)
        last_row = df.iloc[[-1]][feature_cols]
        last_date = df.index[-1]
        last_close = df.loc[last_date,'close']
        last_row_s = scaler.transform(last_row)

        pred_dir = clf.predict(last_row_s)[0]
        pred_proba = clf.predict_proba(last_row_s)[0]
        pred_reg = reg.predict(last_row_s)[0]

        confidence = max(pred_proba) * 100
        pred_price = last_close * (1 + pred_reg)

        action = '🟢 BUY' if pred_dir == 1 else '🔴 SELL/FLAT'
        print(''+'='*60)
        print('🔮 PREDICTION FOR TOMORROW')
        print('='*60)
        print(f"📅 Base Date: {last_date.date()}")
        print(f"💵 Current Close: ${last_close:.2f}")
        print('-'*60)
        print(f"🤖 AI Recommendation: {action}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print(f"🔮 Predicted Return: {pred_reg*100:+.2f}%")
        print(f"🎯 Target Price: ${pred_price:.2f}")
        print('-'*60)

        if 'daily_equity' in backtest_results:
            equity_curve = backtest_results['daily_equity']
            plot_data = pd.DataFrame({
                'AI_Strategy': equity_curve, 
                'Buy_Hold': (test['close'] / test['close'].iloc[0]).values
            }, index=test.index[:len(equity_curve)])

            plt.figure(figsize=(16, 8))
            
            # 1. Vẽ đường tài sản
            plt.plot(plot_data.index, plot_data['AI_Strategy'], label=f'AI Strategy (+{backtest_results["total_return"]*100:.1f}%)', color='blue', linewidth=2)
            plt.plot(plot_data.index, plot_data['Buy_Hold'], label=f'Buy & Hold (+{backtest_results["buy_hold_return"]*100:.1f}%)', color='gray', linestyle='--', alpha=0.6)
            
            # 2. Vẽ điểm Mua/Bán
            # Lấy danh sách trades từ kết quả backtest
            trades = backtest_results['trades']
            buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
            buy_prices = [t['price'] for t in trades if t['type'] == 'BUY']
            
            sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
            sell_prices = [t['price'] for t in trades if t['type'] == 'SELL']
            
            # Vì biểu đồ đang vẽ Equity (Tài sản) chứ không phải Giá (Price), 
            # ta cần map ngày giao dịch sang giá trị tài sản tương ứng để đặt mũi tên.
            # Cách đơn giản hơn: Vẽ 2 biểu đồ con (Subplots).
            
            plt.title('Performance Comparison: AI vs Buy & Hold')
            plt.legend()
            plt.grid(True)
            plt.show()

            # --- Biểu đồ 2: Điểm vào lệnh trên giá ---
            plt.figure(figsize=(16, 8))
            plt.plot(test.index, test['close'], label='Stock Price', color='black', alpha=0.5)
            
            plt.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='BUY Signal', zorder=5)
            plt.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='SELL Signal', zorder=5)
            
            plt.title('Trade Entries & Exits on Google Stock Price')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Vui lòng cập nhật hàm backtest_realistic để trả về 'daily_equity' trong dictionary kết quả.")
        
        if args.save_models:
            joblib.dump(clf,'xgb_classifier.joblib')
            joblib.dump(reg,'xgb_regressor.joblib')
            joblib.dump(scaler,'scaler.joblib')
            print('✓ Models saved')

        print('Done.')

    except FileNotFoundError:
        print(f"Error: Could not find file at {args.source}")
    except Exception as e:
        print('Error:', e)
        import traceback
        traceback.print_exc()
