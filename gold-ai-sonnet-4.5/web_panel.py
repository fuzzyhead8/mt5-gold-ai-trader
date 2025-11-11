"""
Web Trading Panel for Gold AI Sonnet 4.5
"""

from flask import Flask, render_template, jsonify, request
import asyncio
import threading
import json
from datetime import datetime
import os

from main import GoldAISonnet

app = Flask(__name__)
trader = None
trading_thread = None

def run_trader():
    """Run the trader in a separate thread"""
    global trader
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    trader = GoldAISonnet(symbol='XAUUSD', timeframe='H1')

    async def run():
        if await trader.initialize():
            await trader.start_trading()

    loop.run_until_complete(run())

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    if trader:
        status = asyncio.run(trader.get_system_status())
        return jsonify(status)
    return jsonify({'error': 'Trader not initialized'})

@app.route('/api/start', methods=['POST'])
def start_trading():
    """Start trading"""
    global trading_thread
    if trading_thread is None or not trading_thread.is_alive():
        trading_thread = threading.Thread(target=run_trader)
        trading_thread.daemon = True
        trading_thread.start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """Stop trading"""
    if trader and trader.is_running:
        asyncio.run(trader.stop_trading())
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_running'})

@app.route('/api/positions')
def get_positions():
    """Get current positions"""
    if trader:
        positions = []
        for ticket, pos in trader.positions.items():
            positions.append({
                'ticket': ticket,
                'symbol': pos.symbol,
                'direction': pos.direction,
                'volume': pos.volume,
                'entry_price': pos.entry_price,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'current_profit': pos.current_profit,
                'open_time': pos.open_time.isoformat()
            })
        return jsonify(positions)
    return jsonify([])

@app.route('/api/analysis')
def get_analysis():
    """Get latest market analysis"""
    if trader:
        # Get recent market data for analysis
        market_data = asyncio.run(trader.mt5_connector.get_historical_data(
            trader.symbol, trader.timeframe, 50
        ))

        if market_data is not None:
            analysis = asyncio.run(trader.nebula_assistant.analyze_market_conditions(
                trader.symbol, trader.timeframe, market_data
            ))
            return jsonify(analysis)

    return jsonify({'error': 'Analysis not available'})

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    from config import get_config
    config = get_config()
    return jsonify(config)

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)

    # Create dashboard template
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gold AI Sonnet 4.5 - Trading Panel</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .status-card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .status-item { text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; }
        .status-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .status-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
        .control-panel { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 5px; }
        .btn-start { background: #27ae60; color: white; }
        .btn-stop { background: #e74c3c; color: white; }
        .btn:hover { opacity: 0.8; }
        .positions-table { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .profit-positive { color: #27ae60; }
        .profit-negative { color: #e74c3c; }
        .analysis-panel { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .analysis-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .analysis-item { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Gold AI Sonnet 4.5</h1>
            <p>Reverse Engineered AuriON AI System EA - Trading Panel</p>
        </div>

        <div class="status-card">
            <h2>System Status</h2>
            <div id="status-grid" class="status-grid">
                <!-- Status items will be populated by JavaScript -->
            </div>
        </div>

        <div class="control-panel">
            <h2>Trading Controls</h2>
            <button id="start-btn" class="btn btn-start">Start Trading</button>
            <button id="stop-btn" class="btn btn-stop">Stop Trading</button>
            <span id="control-status">System stopped</span>
        </div>

        <div class="analysis-panel">
            <h2>AI Market Analysis</h2>
            <div id="analysis-grid" class="analysis-grid">
                <!-- Analysis items will be populated by JavaScript -->
            </div>
        </div>

        <div class="positions-table">
            <h2>Open Positions</h2>
            <table id="positions-table">
                <thead>
                    <tr>
                        <th>Ticket</th>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Volume</th>
                        <th>Entry Price</th>
                        <th>Stop Loss</th>
                        <th>Take Profit</th>
                        <th>Current P&L</th>
                    </tr>
                </thead>
                <tbody id="positions-body">
                    <!-- Position rows will be populated by JavaScript -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Update status every 5 seconds
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const grid = document.getElementById('status-grid');
                    grid.innerHTML = `
                        <div class="status-item">
                            <div class="status-value">${data.is_running ? '🟢' : '🔴'}</div>
                            <div class="status-label">Status</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value">${data.positions_count}</div>
                            <div class="status-label">Positions</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value">$${data.account_balance.toFixed(2)}</div>
                            <div class="status-label">Balance</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value">$${data.account_equity.toFixed(2)}</div>
                            <div class="status-label">Equity</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value">${data.trading_allowed ? '✅' : '❌'}</div>
                            <div class="status-label">Trading</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value">${data.symbol}</div>
                            <div class="status-label">Symbol</div>
                        </div>
                    `;
                })
                .catch(error => console.error('Error updating status:', error));
        }

        // Update analysis every 10 seconds
        function updateAnalysis() {
            fetch('/api/analysis')
                .then(response => response.json())
                .then(data => {
                    const grid = document.getElementById('analysis-grid');
                    grid.innerHTML = `
                        <div class="analysis-item">
                            <div class="status-value">${data.trend || 'N/A'}</div>
                            <div class="status-label">Trend</div>
                        </div>
                        <div class="analysis-item">
                            <div class="status-value">${data.volatility || 'N/A'}</div>
                            <div class="status-label">Volatility</div>
                        </div>
                        <div class="analysis-item">
                            <div class="status-value">${data.recommendation || 'N/A'}</div>
                            <div class="status-label">Signal</div>
                        </div>
                        <div class="analysis-item">
                            <div class="status-value">${data.momentum ? data.momentum.toFixed(4) : 'N/A'}</div>
                            <div class="status-label">Momentum</div>
                        </div>
                    `;
                })
                .catch(error => console.error('Error updating analysis:', error));
        }

        // Update positions every 3 seconds
        function updatePositions() {
            fetch('/api/positions')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('positions-body');
                    tbody.innerHTML = '';

                    data.forEach(pos => {
                        const row = document.createElement('tr');
                        const profitClass = pos.current_profit >= 0 ? 'profit-positive' : 'profit-negative';

                        row.innerHTML = `
                            <td>${pos.ticket}</td>
                            <td>${pos.symbol}</td>
                            <td>${pos.direction}</td>
                            <td>${pos.volume}</td>
                            <td>${pos.entry_price.toFixed(5)}</td>
                            <td>${pos.stop_loss.toFixed(5)}</td>
                            <td>${pos.take_profit.toFixed(5)}</td>
                            <td class="${profitClass}">$${pos.current_profit.toFixed(2)}</td>
                        `;
                        tbody.appendChild(row);
                    });
                })
                .catch(error => console.error('Error updating positions:', error));
        }

        // Control buttons
        document.getElementById('start-btn').addEventListener('click', function() {
            fetch('/api/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('control-status').textContent =
                        data.status === 'started' ? 'System started' : 'Already running';
                });
        });

        document.getElementById('stop-btn').addEventListener('click', function() {
            fetch('/api/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('control-status').textContent =
                        data.status === 'stopped' ? 'System stopped' : 'Not running';
                });
        });

        // Initial updates
        updateStatus();
        updateAnalysis();
        updatePositions();

        // Set up periodic updates
        setInterval(updateStatus, 5000);
        setInterval(updateAnalysis, 10000);
        setInterval(updatePositions, 3000);
    </script>
</body>
</html>
    """

    with open('templates/dashboard.html', 'w') as f:
        f.write(dashboard_html)

    print("Starting Gold AI Sonnet Web Panel...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
