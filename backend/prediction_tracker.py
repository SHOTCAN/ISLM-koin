"""
Prediction Tracker — Self-Evaluation & Auto-Tuning V1
=====================================================
Logs every prediction, compares against actual outcomes,
and continuously improves internal model accuracy.

Features:
  - Store predictions with timestamps
  - Compare predictions vs actual prices
  - Calculate MAE, directional accuracy, hit rate
  - Auto-tune model weights based on performance
  - Historical backtesting validation
  - Continuous improvement loop
"""

import json
import time
import os
import numpy as np
from datetime import datetime
from collections import defaultdict


# ============================================
# PREDICTION STORE
# ============================================

class PredictionStore:
    """Persist predictions and outcomes to JSON for backtesting."""

    def __init__(self, filepath='/tmp/prediction_history.json'):
        self.filepath = filepath
        self._data = self._load()

    def _load(self):
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {'predictions': [], 'evaluations': [], 'model_weights': {}, 'stats': {}}

    def _save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            print(f"[PRED-STORE] Save error: {e}")

    def add_prediction(self, prediction):
        self._data['predictions'].append(prediction)
        # Keep last 500 predictions
        if len(self._data['predictions']) > 500:
            self._data['predictions'] = self._data['predictions'][-500:]
        self._save()

    def add_evaluation(self, evaluation):
        self._data['evaluations'].append(evaluation)
        if len(self._data['evaluations']) > 500:
            self._data['evaluations'] = self._data['evaluations'][-500:]
        self._save()

    def update_weights(self, weights):
        self._data['model_weights'] = weights
        self._save()

    def update_stats(self, stats):
        self._data['stats'] = stats
        self._save()

    @property
    def predictions(self):
        return self._data.get('predictions', [])

    @property
    def evaluations(self):
        return self._data.get('evaluations', [])

    @property
    def model_weights(self):
        return self._data.get('model_weights', {})

    @property
    def stats(self):
        return self._data.get('stats', {})


# ============================================
# PREDICTION EVALUATOR
# ============================================

class PredictionEvaluator:
    """Compare predictions against actual outcomes."""

    @staticmethod
    def evaluate_single(prediction, actual_price):
        """
        Evaluate a single prediction against actual price.
        Returns evaluation metrics.
        """
        pred_price = prediction.get('predicted_price', 0)
        base_price = prediction.get('base_price', 0)
        if pred_price <= 0 or base_price <= 0:
            return None

        # Absolute error
        abs_error = abs(actual_price - pred_price)
        pct_error = (abs_error / pred_price) * 100

        # Directional accuracy
        pred_direction = 'up' if pred_price > base_price else 'down'
        actual_direction = 'up' if actual_price > base_price else 'down'
        direction_correct = pred_direction == actual_direction

        # Range accuracy (was actual within predicted range?)
        pred_high = prediction.get('predicted_high', pred_price * 1.05)
        pred_low = prediction.get('predicted_low', pred_price * 0.95)
        in_range = pred_low <= actual_price <= pred_high

        return {
            'timestamp': time.time(),
            'coin': prediction.get('coin', 'unknown'),
            'horizon': prediction.get('horizon', 'unknown'),
            'model': prediction.get('model', 'unknown'),
            'base_price': base_price,
            'predicted_price': pred_price,
            'actual_price': actual_price,
            'abs_error': round(abs_error, 2),
            'pct_error': round(pct_error, 4),
            'direction_correct': direction_correct,
            'in_range': in_range,
        }


# ============================================
# MODEL WEIGHT OPTIMIZER
# ============================================

class ModelWeightOptimizer:
    """Auto-tune model weights based on historical accuracy."""

    DEFAULT_WEIGHTS = {
        'gbm_monte_carlo': 0.40,
        'arima': 0.30,
        'ema_trend': 0.15,
        'mean_reversion': 0.15,
    }

    def __init__(self):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self._model_errors = defaultdict(list)  # model → [pct_errors]

    def record_error(self, model_name, pct_error):
        """Record prediction error for a model."""
        self._model_errors[model_name].append(pct_error)
        # Keep last 100 errors per model
        if len(self._model_errors[model_name]) > 100:
            self._model_errors[model_name] = self._model_errors[model_name][-100:]

    def optimize(self):
        """
        Recalculate weights based on inverse MAE.
        Models with lower error get higher weight.
        """
        if not self._model_errors:
            return self.weights

        # Calculate MAE for each model
        maes = {}
        for model, errors in self._model_errors.items():
            if errors:
                maes[model] = np.mean(errors)
            else:
                maes[model] = 10.0  # High default error

        # Inverse MAE weighting
        inv_maes = {m: 1.0 / (mae + 0.001) for m, mae in maes.items()}
        total_inv = sum(inv_maes.values())

        if total_inv > 0:
            for model in self.weights:
                if model in inv_maes:
                    self.weights[model] = round(inv_maes[model] / total_inv, 4)

        # Ensure all weights present and sum to ~1
        for model in self.DEFAULT_WEIGHTS:
            if model not in self.weights:
                self.weights[model] = self.DEFAULT_WEIGHTS[model]

        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: round(v / total, 4) for k, v in self.weights.items()}

        return self.weights

    def get_report(self):
        """Get human-readable weight report."""
        lines = ["📊 *Model Weights (Auto-Tuned):*"]
        for model, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            errors = self._model_errors.get(model, [])
            mae = np.mean(errors) if errors else 0
            bar_len = int(weight * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            lines.append(f"  {model}: {bar} {weight*100:.1f}% (MAE: {mae:.2f}%)")
        return "\n".join(lines)


# ============================================
# ENHANCED PREDICTOR (Multi-Model Blend)
# ============================================

class EnhancedPredictor:
    """
    Multi-model prediction blender with auto-tuning.
    Models: GBM Monte Carlo, ARIMA-like, EMA Trend, Mean Reversion.
    """

    @staticmethod
    def predict_arima_like(prices, horizon_hours=1):
        """Simple ARIMA-inspired prediction using differencing + AR."""
        if len(prices) < 30:
            return prices[-1]

        # First difference
        diff = np.diff(prices[-30:])

        # AR(3) — predict next diff from last 3 diffs
        if len(diff) >= 3:
            # Simple weighted average of recent diffs
            w = [0.5, 0.3, 0.2]
            next_diff = sum(d * w_ for d, w_ in zip(diff[-3:], w))
        else:
            next_diff = np.mean(diff)

        # Scale by horizon
        scale = horizon_hours  # rough scaling
        predicted = prices[-1] + (next_diff * scale)

        return max(predicted, prices[-1] * 0.8)  # floor at -20%

    @staticmethod
    def predict_ema_trend(prices, horizon_hours=1):
        """EMA-based trend extrapolation."""
        if len(prices) < 20:
            return prices[-1]

        # Calculate EMA trend
        ema_fast = EnhancedPredictor._ema(prices, 7)
        ema_slow = EnhancedPredictor._ema(prices, 21)

        # Trend momentum
        trend = (ema_fast / ema_slow - 1) * 100

        # Extrapolate
        pct_change = trend * 0.3 * horizon_hours  # dampened extrapolation
        predicted = prices[-1] * (1 + pct_change / 100)

        return max(predicted, prices[-1] * 0.8)

    @staticmethod
    def predict_mean_reversion(prices, horizon_hours=1):
        """Mean reversion model — price tends to return to moving average."""
        if len(prices) < 50:
            return prices[-1]

        ma_50 = np.mean(prices[-50:])
        current = prices[-1]

        # How far from mean?
        deviation = (current / ma_50 - 1) * 100

        # Revert partially toward mean
        reversion_speed = 0.1 * horizon_hours  # 10% per hour toward mean
        predicted = current + (ma_50 - current) * min(reversion_speed, 0.5)

        return max(predicted, current * 0.8)

    @staticmethod
    def _ema(data, period):
        if len(data) < period:
            return data[-1] if data else 0
        ema = float(np.mean(data[:period]))
        k = 2 / (period + 1)
        for val in data[period:]:
            ema = float(val) * k + ema * (1 - k)
        return ema

    @staticmethod
    def predict_blended(prices, weights, horizon_hours=1):
        """
        Run all models and blend predictions using optimized weights.
        Returns {prediction, models, confidence, range}.
        """
        if not prices or len(prices) < 10:
            return {'prediction': 0, 'confidence': 0}

        current = prices[-1]
        models = {}

        # Model 1: GBM Monte Carlo (from existing MarketProjector)
        try:
            returns = np.diff(np.log(prices[-50:])) if len(prices) >= 50 else np.diff(np.log(prices))
            mu = np.mean(returns)
            sigma = np.std(returns)
            dt = horizon_hours / 24  # fraction of day
            sims = []
            for _ in range(200):
                z = np.random.normal()
                sim_price = current * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
                sims.append(sim_price)
            models['gbm_monte_carlo'] = np.median(sims)
        except:
            models['gbm_monte_carlo'] = current

        # Model 2: ARIMA-like
        try:
            models['arima'] = EnhancedPredictor.predict_arima_like(prices, horizon_hours)
        except:
            models['arima'] = current

        # Model 3: EMA Trend
        try:
            models['ema_trend'] = EnhancedPredictor.predict_ema_trend(prices, horizon_hours)
        except:
            models['ema_trend'] = current

        # Model 4: Mean Reversion
        try:
            models['mean_reversion'] = EnhancedPredictor.predict_mean_reversion(prices, horizon_hours)
        except:
            models['mean_reversion'] = current

        # Blend using weights
        blended = 0
        total_w = 0
        for model_name, pred_price in models.items():
            w = weights.get(model_name, 0.25)
            blended += pred_price * w
            total_w += w

        if total_w > 0:
            blended /= total_w

        # Confidence based on model agreement
        all_preds = list(models.values())
        if all_preds:
            pred_std = np.std(all_preds)
            pred_mean = np.mean(all_preds) if np.mean(all_preds) > 0 else 1
            cv = (pred_std / pred_mean) * 100  # coefficient of variation
            confidence = max(10, min(95, 100 - cv * 10))
        else:
            confidence = 50

        # Prediction range (based on model spread)
        pred_low = min(all_preds) if all_preds else current * 0.95
        pred_high = max(all_preds) if all_preds else current * 1.05

        return {
            'prediction': round(blended, 2),
            'models': {k: round(v, 2) for k, v in models.items()},
            'confidence': round(confidence, 1),
            'range_low': round(pred_low, 2),
            'range_high': round(pred_high, 2),
            'horizon_hours': horizon_hours,
            'current_price': current,
            'change_pct': round(((blended / current) - 1) * 100, 2),
        }


# ============================================
# MAIN PREDICTION TRACKER
# ============================================

class PredictionTracker:
    """
    Complete self-evaluation system:
    1. Log predictions
    2. Compare with actual outcomes
    3. Calculate accuracy metrics
    4. Auto-adjust model weights
    5. Continuous improvement loop
    """

    def __init__(self):
        self.store = PredictionStore()
        self.optimizer = ModelWeightOptimizer()
        self.evaluator = PredictionEvaluator()
        self._last_eval_time = 0
        self._eval_interval = 3600  # evaluate every hour

        # Load saved weights
        saved_weights = self.store.model_weights
        if saved_weights:
            self.optimizer.weights = saved_weights

    def log_prediction(self, coin, horizon, base_price, prediction_result, model='blended'):
        """Log a new prediction for future evaluation."""
        entry = {
            'timestamp': time.time(),
            'eval_at': time.time() + horizon * 3600,  # when to evaluate
            'coin': coin,
            'horizon': f"{horizon}h",
            'model': model,
            'base_price': base_price,
            'predicted_price': prediction_result.get('prediction', 0),
            'predicted_high': prediction_result.get('range_high', 0),
            'predicted_low': prediction_result.get('range_low', 0),
            'confidence': prediction_result.get('confidence', 0),
            'models': prediction_result.get('models', {}),
            'evaluated': False,
        }
        self.store.add_prediction(entry)

    def evaluate_pending(self, current_prices):
        """
        Evaluate all pending predictions where eval_at has passed.
        current_prices: dict of {coin: current_price}
        """
        now = time.time()
        if now - self._last_eval_time < self._eval_interval:
            return 0

        self._last_eval_time = now
        evaluated = 0

        for pred in self.store.predictions:
            if pred.get('evaluated'):
                continue
            if now < pred.get('eval_at', now + 9999):
                continue

            coin = pred.get('coin', '').upper()
            actual = current_prices.get(coin, 0)
            if actual <= 0:
                continue

            # Evaluate main prediction
            result = self.evaluator.evaluate_single(pred, actual)
            if result:
                self.store.add_evaluation(result)

                # Record errors per model
                models = pred.get('models', {})
                for model_name, model_pred in models.items():
                    if model_pred > 0:
                        model_error = abs((actual - model_pred) / model_pred * 100)
                        self.optimizer.record_error(model_name, model_error)

                pred['evaluated'] = True
                evaluated += 1

        if evaluated > 0:
            # Re-optimize weights
            new_weights = self.optimizer.optimize()
            self.store.update_weights(new_weights)

            # Update stats
            self._update_stats()

        return evaluated

    def _update_stats(self):
        """Calculate aggregate accuracy stats."""
        evals = self.store.evaluations
        if not evals:
            return

        recent = evals[-100:]  # last 100 evaluations

        mae = np.mean([e['pct_error'] for e in recent])
        dir_acc = sum(1 for e in recent if e['direction_correct']) / len(recent) * 100
        range_hit = sum(1 for e in recent if e['in_range']) / len(recent) * 100

        stats = {
            'total_evaluated': len(evals),
            'recent_count': len(recent),
            'mae_pct': round(mae, 2),
            'directional_accuracy': round(dir_acc, 1),
            'range_hit_rate': round(range_hit, 1),
            'last_updated': datetime.now().isoformat(),
        }
        self.store.update_stats(stats)

    def get_accuracy_report(self):
        """Generate human-readable accuracy report."""
        stats = self.store.stats
        if not stats or stats.get('total_evaluated', 0) == 0:
            return (
                "📊 *PREDICTION ACCURACY*\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "⏳ Belum ada prediksi yang dievaluasi.\n"
                "_Sistem akan mulai tracking setelah prediksi pertama._"
            )

        weights_report = self.optimizer.get_report()

        # Accuracy rating
        mae = stats.get('mae_pct', 99)
        if mae < 2:
            rating = "🏆 EXCELLENT"
        elif mae < 5:
            rating = "✅ GOOD"
        elif mae < 10:
            rating = "⚠️ FAIR"
        else:
            rating = "❌ NEEDS IMPROVEMENT"

        return (
            f"📊 *PREDICTION ACCURACY REPORT*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📈 Total Evaluasi: {stats['total_evaluated']}\n"
            f"🎯 MAE: {stats['mae_pct']:.2f}% {rating}\n"
            f"↕️ Directional Accuracy: {stats['directional_accuracy']:.1f}%\n"
            f"📐 Range Hit Rate: {stats['range_hit_rate']:.1f}%\n\n"
            f"{weights_report}\n\n"
            f"🔄 Auto-tuning: ACTIVE\n"
            f"⏰ Last update: {stats.get('last_updated', 'N/A')}"
        )

    def predict(self, coin, prices, horizon_hours=1):
        """
        Make a blended prediction and log it.
        Returns prediction result dict.
        """
        result = EnhancedPredictor.predict_blended(
            prices, self.optimizer.weights, horizon_hours
        )

        # Log for future evaluation
        if result.get('prediction', 0) > 0:
            self.log_prediction(
                coin=coin,
                horizon=horizon_hours,
                base_price=prices[-1] if prices else 0,
                prediction_result=result,
            )

        return result
